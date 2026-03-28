import csv
import os
import re
import json
import torch
import numpy as np
from collections import defaultdict
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from openai import OpenAI
from text2vec import SentenceModel

# ------------------- 配置 --------------------
finetune_model_path = 'model_TCM/'                  # 微调模型路径
csv_output_path = './test_result.csv'               # CSV输出文件路径
json_output_path = './test_result.json'             # JSON输出文件路径
therapy_keywords_path = './therapy_keywords.txt'    # 疗法特征关键词文件
term_keywords_path = './term_keywords.txt'          # 医学术语关键词文件
safety_keywords_path = './safety_keywords.txt'      # 安全性关键词文件
TEXT2VEC_MODEL_NAME = 'text2vec-base-chinese'       # 文本向量模型名称
api_key = "sk-1f0648aee9f54f41b993ef98f646aca2"     # API Key

# 初始化API客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# ------------------- 模型加载配置 --------------------
print("正在加载模型和配置...")
config = PeftConfig.from_pretrained(finetune_model_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, 
    use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA可用: {cuda_available}")

if cuda_available:
    # GPU模式
    device = torch.device("cuda:3")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    # CPU模式
    device = torch.device("cpu")
    print("使用CPU")

# 根据设备类型设置不同的加载参数
if cuda_available:
    # GPU模式：使用自动设备映射和8位量化
    print("GPU模式：使用8位量化...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        trust_remote_code=True,
        use_flash_attention_2=False
    )
else:
    # CPU模式：不使用8位量化，使用float32
    print("CPU模式：使用float32...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map=None,                # CPU模式不设置device_map
        torch_dtype=torch.float32,      # CPU使用float32
        load_in_8bit=False,             # CPU不支持8位量化
        trust_remote_code=True,
        use_flash_attention_2=False
    )
    # 将模型移到CPU并转换为float32
    model = model.to(device)
    model = model.float()               # 确保所有参数都是float32

# 应用LoRA微调权重
if cuda_available:
    model = PeftModel.from_pretrained(
        model, 
        finetune_model_path,
        device_map="auto"
    )
else:
    model = PeftModel.from_pretrained(
        model, 
        finetune_model_path,
        device_map=None
    )
    model = model.to(device)
    model = model.float()

model = model.eval()

print("主模型加载完成!")

# ------------------- 文本向量模型加载 --------------------
print(f"正在加载文本向量模型: {TEXT2VEC_MODEL_NAME}")
try:
    text2vec_model = SentenceModel(TEXT2VEC_MODEL_NAME)
    if cuda_available:
        # 如果GPU可用，将模型移到GPU
        text2vec_model = text2vec_model.to(device)
    text2vec_model.eval()
    print("文本向量模型加载完成!")
except Exception as e:
    print(f"警告: 无法加载文本向量模型 {TEXT2VEC_MODEL_NAME}: {e}")
    text2vec_model = None

# ------------------- 文本向量工具函数 --------------------
def compute_text_similarity(text1, text2):
    """使用text2vec模型计算两个文本的相似度"""
    if text2vec_model is None:
        # 如果模型未加载，使用简单的Jaccard相似度作为备选
        words1 = set(re.findall(r'[\u4e00-\u9fff\w]+', text1))
        words2 = set(re.findall(r'[\u4e00-\u9fff\w]+', text2))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union
    
    try:
        # 使用text2vec模型计算相似度
        embeddings = text2vec_model.encode([text1, text2])
        
        # 计算余弦相似度
        from numpy import dot
        from numpy.linalg import norm
        cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        return float(cos_sim)
    except Exception as e:
        print(f"计算文本相似度时出错: {e}")
        # 出错时返回一个中间值
        return 0.5

def extract_key_sentences(text, max_sentences=5):
    """提取文本中的关键句子"""
    # 按标点分割句子
    sentences = re.split(r'[。！？；]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # 过滤掉包含某些关键词的句子（治疗相关的）
    key_indicators = ['治疗', '方案', '原则', '建议', '药物', '中药', '西药', '诊断', '康复', '预防']
    key_sentences = []
    
    for sentence in sentences:
        if any(indicator in sentence for indicator in key_indicators):
            key_sentences.append(sentence)
    
    # 如果没有找到关键词句子，返回前几个句子
    if not key_sentences:
        return sentences[:max_sentences]
    
    # 返回最多指定数量的关键句子
    return key_sentences[:max_sentences]

def compute_sentence_level_similarity(text1, text2):
    """基于句子级别的相似度计算"""
    sentences1 = extract_key_sentences(text1, max_sentences=5)
    sentences2 = extract_key_sentences(text2, max_sentences=5)
    
    if not sentences1 or not sentences2:
        return 0.0
    
    # 计算句子间的相似度矩阵
    similarity_matrix = []
    for sent1 in sentences1:
        row = []
        for sent2 in sentences2:
            similarity = compute_text_similarity(sent1, sent2)
            row.append(similarity)
        similarity_matrix.append(row)
    
    # 取最佳匹配的平均值
    if similarity_matrix:
        # 对于每个句子，取与另一文本中句子的最高相似度
        max_similarities = [max(row) for row in similarity_matrix if row]
        if max_similarities:
            return sum(max_similarities) / len(max_similarities)
    
    return 0.0

# ------------------- API调用函数 --------------------
def generate_reference_by_api(disease):
    prompt = f"""你是一位精通中医和西医的医生，请为{disease}制定一个中西医结合治疗方案。
请按照以下步骤思考:
1. 简要描述中医和西医分别对该疾病的理论基础与病因认识;
2. 分别从中医和西医的角度，分析针对该疾病的治疗原则与手段;
3. 分析对于该疾病中医和西医的药物与疗法特点对比;
4. 分析中医和西医对疾病特性的侧重，如慢性和急性如何治疗，有并发症如何治疗;
5. 分别从中医和西医的角度，分析针对该疾病的预防与康复理念;
6. 结合上述分析，提出一个综合的中西医结合治疗方案，包括治疗原则、中药处方、西药处方、生活方式建议、饮食建议等。"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位经验丰富的中西医结合专家，擅长制定规范、标准的治疗方案。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.5,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {e}")
        # 如果API调用失败，使用本地模型生成
        return generate_text(prompt, max_new_tokens=1024, temperature=0.5)

# ------------------- 关键词文件读取函数 --------------------
def load_keywords_from_file(file_path, default_keywords=None):
    """从文本文件加载关键词，每行一个关键词"""
    if default_keywords is None:
        default_keywords = []
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
                print(f"从 {file_path} 加载了 {len(keywords)} 个关键词")
                return keywords
        else:
            print(f"警告: 文件 {file_path} 不存在，使用默认关键词")
            return default_keywords
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return default_keywords

# ------------------- 文本处理函数 --------------------
def extract_json_from_text(text):
    """从文本中提取JSON"""
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return None

def generate_text(prompt, max_new_tokens=1024, temperature=0.7):
    """生成文本（本地模型）"""
    encoding = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 将输入移到正确的设备
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    
    generate_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.9,
        "temperature": temperature,
        "repetition_penalty": 1.1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    
    # 使用适当的上下文管理器
    with torch.no_grad():
        with torch.autocast('cuda' if cuda_available else 'cpu'):
            generate_ids = model.generate(**generate_input)
    
    full_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):].strip()
    return generated_text

# ------------------- 分层融合投票系统 --------------------
class HierarchicalVotingSystem:
    """分层融合投票系统"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 从文件加载疗法特征关键词库（通用的疗法关键词）
        self.therapy_keywords = load_keywords_from_file(therapy_keywords_path)
        
        # 从文件加载医学术语库
        self.medical_terms = load_keywords_from_file(term_keywords_path)
        
        # 从文件加载安全性关键词库
        self.safety_indicators = load_keywords_from_file(safety_keywords_path)
    
    def assess_content_quality(self, answers_dict):
        """内容质量评估 - 使用通用的疗法特征关键词库"""
        content_scores = {}
        
        for path, answer in answers_dict.items():
            scores = {
                "completeness": 0,
                "specificity": 0,
                "professionalism": 0,
                "safety": 0
            }
            
            # 1. 完整性评估
            structure_elements = [
                "诊断", "治疗原则", "具体方案", "中药", "西药",
                "监测", "注意事项", "生活方式", "饮食建议"
            ]
            for element in structure_elements:
                if element in answer:
                    scores["completeness"] += 1
            
            # 限制最高分
            scores["completeness"] = min(10, scores["completeness"])
            
            # 2. 特异性评估 - 使用通用的疗法特征关键词库
            keyword_count = 0
            for keyword in self.therapy_keywords:
                if keyword in answer:
                    keyword_count += 1
            
            # 归一化：每5个关键词得1分，最高10分
            scores["specificity"] = min(10, keyword_count // 5 + min(2, keyword_count))
            
            # 3. 专业性评估 - 使用医学术语库
            term_count = 0
            for term in self.medical_terms:
                if term in answer:
                    term_count += 1
            
            # 归一化：每3个专业术语得1分，最高10分
            scores["professionalism"] = min(10, term_count // 3 + min(2, term_count))
            
            # 4. 安全性评估
            safety_count = 0
            for indicator in self.safety_indicators:
                if indicator in answer:
                    safety_count += 1
            
            # 归一化：每2个安全性指标得1分，最高10分
            scores["safety"] = min(10, safety_count // 2 + min(3, safety_count))
            
            content_scores[path] = scores
        
        return content_scores
    
    def assess_consistency(self, answers_dict):
        """一致性评估 - 基于通用的疗法特征关键词"""
        # 提取每个答案的关键主张
        key_claims = {}
        
        for path, answer in answers_dict.items():
            claims = []
            
            # 提取治疗原则（不使用疾病关键词）
            principle_patterns = [
                r'治疗原则[：:]\s*(.*?)\n',
                r'核心治则[：:]\s*(.*?)\n',
                r'治疗思路[：:]\s*(.*?)\n',
                r'治疗方案[：:]\s*(.*?)\n',
                r'治疗目标[：:]\s*(.*?)\n'
            ]
            
            for pattern in principle_patterns:
                match = re.search(pattern, answer)
                if match:
                    claim = match.group(1).strip()
                    # 过滤掉过长的原则
                    if len(claim) < 50 and len(claim) > 5:
                        claims.append(f"原则:{claim}")
            
            # 提取包含疗法特征关键词的句子作为关键主张
            for keyword in self.therapy_keywords:
                # 查找包含关键词的句子
                keyword_pattern = r'[^。！？；]*?' + re.escape(keyword) + r'[^。！？；]*?[。！？；]'
                matches = re.findall(keyword_pattern, answer)
                for match in matches[:2]:  # 每个关键词最多取前2个匹配
                    if len(match) < 80 and len(match) > 10:
                        simplified = match[:50].strip()
                        claims.append(f"疗法:{simplified}")
            
            # 去重并限制数量
            unique_claims = list(set(claims))
            key_claims[path] = unique_claims[:10]  # 最多保留10个关键主张
        
        # 计算一致性矩阵            
        consistency_matrix = {}
        paths = list(answers_dict.keys())
        
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                path1, path2 = paths[i], paths[j]
                claims1 = set(key_claims[path1])
                claims2 = set(key_claims[path2])
                
                if claims1 and claims2:
                    intersection = len(claims1.intersection(claims2))
                    union = len(claims1.union(claims2))
                    similarity = intersection / union if union > 0 else 0
                else:
                    similarity = 0
                
                consistency_matrix[f"{path1}-{path2}"] = similarity
        
        # 计算每个路径的平均一致性
        consistency_scores = defaultdict(float)
        for pair, similarity in consistency_matrix.items():
            path1, path2 = pair.split("-")
            consistency_scores[path1] += similarity
            consistency_scores[path2] += similarity
        
        for path in paths:
            consistency_scores[path] = consistency_scores[path] / (len(paths) - 1) if len(paths) > 1 else 1
        
        return dict(consistency_scores), key_claims
    
    def assess_reference_alignment(self, disease, answers_dict):
        """参考标准对齐评估 - 使用text2vec-base-chinese模型"""
        # 使用API生成参考标准答案
        print("  调用DeepSeek API生成参考标准...")
        reference_answer = generate_reference_by_api(disease)
        
        alignment_scores = {}
        
        # 使用text2vec模型计算每个答案与参考标准的相似度
        print("  使用text2vec-base-chinese模型计算语义相似度...")
        for path, answer in answers_dict.items():
            # 1. 计算整体文本相似度
            overall_similarity = compute_text_similarity(answer, reference_answer)
            
            # 2. 计算句子级别的相似度
            sentence_similarity = compute_sentence_level_similarity(answer, reference_answer)
            
            # 3. 计算关键词匹配度（保留作为辅助指标）
            keyword_match = 0
            for keyword in self.therapy_keywords:
                if keyword in reference_answer and keyword in answer:
                    keyword_match += 1
            keyword_score = min(1.0, keyword_match / 10)  # 每10个关键词匹配得1分，最高1分
            
            # 综合评分：整体相似度占50%，句子相似度占30%，关键词匹配占20%
            combined_score = (
                overall_similarity * 0.5 +
                sentence_similarity * 0.3 +
                keyword_score * 0.2
            )
            
            # 归一化到0-10分（相似度是0-1，乘以10得到0-10）
            alignment_scores[path] = min(10, combined_score * 10)
            
            # 输出详细评估信息
            print(f"路径{path}: 整体相似度={overall_similarity:.3f}, "
                  f"句子相似度={sentence_similarity:.3f}, "
                  f"关键词匹配={keyword_match}, "
                  f"综合得分={alignment_scores[path]:.2f}")
        
        return alignment_scores, reference_answer
    
    def integrated_voting(self, disease, answers_dict, dynamic_route):
        """集成投票决策"""
        print(f"\n为疾病'{disease}'进行集成投票评估...")
        
        # 1. 内容质量评估（使用通用的关键词库）
        print("阶段1: 内容质量评估...")
        content_scores = self.assess_content_quality(answers_dict)
        
        for path, scores in content_scores.items():
            total = sum(scores.values())
            print(f"  路径{path}: 完整性={scores['completeness']}, 特异性={scores['specificity']}, "
                  f"专业性={scores['professionalism']}, 安全性={scores['safety']}, 总分={total}")
        
        # 2. 一致性评估
        print("\n阶段2: 一致性评估...")
        consistency_scores, key_claims = self.assess_consistency(answers_dict)
        print(f"  一致性分数: {consistency_scores}")
        
        # 3. 参考标准对齐评估
        print("\n阶段3: 参考标准对齐评估...")
        alignment_scores, reference_answer = self.assess_reference_alignment(disease, answers_dict)
        print(f"  对齐分数: {alignment_scores}")
        
        # 4. 综合评分
        print("\n阶段4: 综合评分计算...")
        final_scores = {}
        
        for path in answers_dict.keys():
            # 计算内容质量总分（归一化到0-10）
            content_total = sum(content_scores[path].values())  # 最高40
            content_normalized = content_total / 4.0  # 归一化到0-10
            
            # 一致性分数（0-1范围）乘以10得到0-10分
            consistency_normalized = consistency_scores[path] * 10
            
            # 对齐分数已经在0-10范围
            
            # 加权综合评分
            # 权重：内容质量40%，一致性30%，对齐30%
            weighted_score = (
                content_normalized * 0.4 +
                consistency_normalized * 0.3 +
                alignment_scores[path] * 0.3
            )
            
            # 动态路由奖励（如果该路径是动态路由推荐的，加1分）
            if path == dynamic_route:
                weighted_score += 1.0
                print(f"  路径{path}获得动态路由奖励+1分")
            
            final_scores[path] = round(weighted_score, 2)
        
        print("最终综合评分:", final_scores)
        
        # 选择最佳路径
        best_path = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_path]
        
        # 生成选择理由
        reasons = []
        
        # 分析每个路径的优势
        for path in answers_dict.keys():
            strengths = []
            if content_scores[path]['completeness'] > 7:
                strengths.append("结构完整")
            if content_scores[path]['specificity'] > 6:
                strengths.append("疗法特征丰富")
            if content_scores[path]['professionalism'] > 6:
                strengths.append("专业性强")
            if content_scores[path]['safety'] > 7:
                strengths.append("安全性高")
            if consistency_scores[path] > 0.6:
                strengths.append("共识性高")
            
            if strengths:
                reasons.append(f"路径{path}: {', '.join(strengths)}")
        
        selection_reason = f"路径{best_path}综合评分最高({best_score}分)。" + \
                          "其他路径优势：" + "；".join(reasons)
        
        return {
            "best_path": best_path,
            "best_score": best_score,
            "scores": {
                "content": content_scores,
                "consistency": consistency_scores,
                "alignment": alignment_scores,
                "final": final_scores
            },
            "key_claims": key_claims,
            "reference_answer": reference_answer,
            "selection_reason": selection_reason
        }

# ------------------- 动态路由决策系统 --------------------
class DynamicRoutingSystem:
    """动态路由决策系统"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def perceive_condition_features(self, disease):
        """病情特征感知"""
        prompt = f"""作为中西医结合专家，请分析以下疾病的特征：

疾病：{disease}

请从以下四个维度进行分类：
1. 病程：急性（发病急骤，进展快）或慢性（病程长，进展缓慢）
2. 病性：实证（邪气盛，正气不虚）、虚证（正气不足）、或虚实夹杂
3. 病势：危重（生命体征不稳定）、轻症（症状轻微）、或中等
4. 认知焦点：病原体明确（有明确感染源）、功能失调为主（器官功能紊乱）、或两者兼有

请严格按照以下JSON格式输出：
{{
  "course": "急性"或"慢性",
  "nature": "实证"或"虚证"或"虚实夹杂",
  "severity": "危重"或"轻症"或"中等",
  "focus": "病原体明确"或"功能失调为主"或"两者兼有"
}}"""
        
        response = generate_text(prompt, max_new_tokens=256, temperature=0.3)
        
        # 解析JSON响应
        features = extract_json_from_text(response)
        if not features:
            # 如果解析失败，使用默认特征
            features = {
                "course": "慢性",
                "nature": "虚实夹杂",
                "severity": "中等",
                "focus": "两者兼有"
            }
        
        return features
    
    def route_decision(self, features):
        """路由决策"""
        course = features.get("course", "慢性")
        severity = features.get("severity", "中等")
        nature = features.get("nature", "虚实夹杂")
        focus = features.get("focus", "两者兼有")
        
        # 决策规则
        if course == "急性" and severity == "危重":
            return "A"  # 先西后中
        elif course == "慢性" and nature == "虚证":
            return "B"  # 先中后西
        elif nature == "虚实夹杂" or focus == "两者兼有":
            return "C"  # 中西并行
        else:
            return "C"  # 默认中西并行
    
    def get_route_description(self, route):
        """获取路由描述"""
        descriptions = {
            "A": "先西后中（西医主导急症处理，中医辅助调理康复）",
            "B": "先中后西（中医主导整体调理，西医辅助监测干预）",
            "C": "中西并行（深度整合，对抗协同）"
        }
        return descriptions.get(route, "未知路径")

# ------------------- 路径答案生成器 --------------------
class PathAnswerGenerator:
    """路径答案生成器"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_path_a_answer(self, disease):
        """生成路径A（先西后中）答案"""
        prompt = f"""作为中西医结合专家，请为{disease}制定"先西后中"的治疗方案：

【方案要求】
1. 西医部分：制定西医主导的治疗方案，包括诊断、紧急处理、主要药物
2. 中医部分：制定中医辅助的调理方案，包括辨证分型、中药方剂、传统疗法
3. 衔接要点：明确中西医如何有序衔接，时间节点，注意事项

请按照以下结构回答：
### 一、西医主导治疗
1. 诊断与评估
2. 主要治疗方案
3. 关键药物及用法

### 二、中医辅助调理
1. 中医辨证分型
2. 中药方剂建议
3. 其他中医疗法

### 三、中西医衔接与整合
1. 治疗时间线
2. 注意事项
3. 预期疗效评估"""
        
        return generate_text(prompt, max_new_tokens=1024, temperature=0.7)
    
    def generate_path_b_answer(self, disease):
        """生成路径B（先中后西）答案"""
        prompt = f"""作为中西医结合专家，请为{disease}制定"先中后西"的治疗方案：

【方案要求】
1. 中医部分：制定中医主导的整体调理方案，包括辨证分型、治则治法、中药方剂
2. 西医部分：制定西医辅助的监测和干预方案，包括必要检查、指标监测、必要时干预
3. 长期管理：制定长期管理方案，包括阶段目标、随访计划

请按照以下结构回答：
### 一、中医主导调理
1. 辨证分析与治疗原则
2. 中药方剂及用法
3. 传统疗法建议

### 二、西医辅助支持
1. 必要检查与监测
2. 西医干预时机与方式
3. 疗效评估指标

### 三、长期管理与随访
1. 阶段治疗目标
2. 随访计划
3. 生活调理建议"""
        
        return generate_text(prompt, max_new_tokens=1024, temperature=0.7)
    
    def generate_path_c_answer(self, disease):
        """生成路径C（中西并行）答案"""
        prompt = f"""作为中西医结合专家，请为{disease}制定"中西并行"的整合治疗方案：

【方案要求】
1. 中西医理论互补：分别阐述中西医对该病的理论解释，分析如何互补
2. 协同治疗方案：设计中西药协同方案，避免简单叠加，强调有机整合
3. 分层分阶段实施：根据疾病不同阶段（急性期、恢复期、维持期）制定不同重点

请按照以下结构回答：
### 一、中西医理论互补分析
1. 西医病理机制
2. 中医病因病机
3. 两者互补点

### 二、协同治疗方案设计
1. 药物协同方案
2. 非药物疗法整合
3. 治疗时间线

### 三、分层分阶段实施
1. 急性期处理重点
2. 恢复期调理重点
3. 维持期管理重点"""
        
        return generate_text(prompt, max_new_tokens=1024, temperature=0.7)

# ------------------- 最终方案选择器 --------------------
class FinalPlanSelector:
    """最终方案选择器 - 直接选择最优路径对应的方案"""
    
    def __init__(self):
        pass
    
    def select_final_plan(self, disease, features, dynamic_route, voting_result, answers_dict):
        """选择最终方案：直接使用最优路径的答案"""
        best_path = voting_result['best_path']
        final_plan = answers_dict[best_path]
        
        # 添加方案说明头部
        header = f"""# {disease}中西医结合治疗方案

## 方案选择说明
1. **病情特征分析**：
   - 病程：{features.get('course', '未知')}
   - 病性：{features.get('nature', '未知')}
   - 病势：{features.get('severity', '未知')}
   - 认知焦点：{features.get('focus', '未知')}

2. **决策路径**：
   - 动态路由推荐：路径{dynamic_route}（{self._get_route_desc(dynamic_route)}）
   - 分层投票结果：路径{best_path}（综合评分：{voting_result['best_score']}分）
   - 选择理由：{voting_result['selection_reason']}

## 最终治疗方案（路径{best_path}）
"""
        
        return header + "\n" + final_plan
    
    def _get_route_desc(self, route):
        """获取路由描述"""
        descriptions = {
            "A": "先西后中（西医主导急症处理，中医辅助调理康复）",
            "B": "先中后西（中医主导整体调理，西医辅助监测干预）",
            "C": "中西并行（深度整合，对抗协同）"
        }
        return descriptions.get(route, "未知路径")

# ------------------- 主处理函数 --------------------
def process_disease(disease, routing_system, answer_generator, voting_system, selector):
    """处理单个疾病的完整流程"""
    print(f"\n{'='*60}")
    print(f"处理疾病: {disease}")
    print('='*60)
    
    # 步骤1: 病情特征感知和动态路由
    print("步骤1: 病情特征感知和动态路由...")
    features = routing_system.perceive_condition_features(disease)
    dynamic_route = routing_system.route_decision(features)
    route_desc = routing_system.get_route_description(dynamic_route)
    
    print(f"  病情特征: {features}")
    print(f"  动态路由推荐: 路径{dynamic_route} - {route_desc}")
    
    # 步骤2: 生成三个路径的答案
    print("\n步骤2: 生成三个路径的答案...")
    answers = {}
    
    print("  生成路径A答案...")
    answers["A"] = answer_generator.generate_path_a_answer(disease)
    
    print("  生成路径B答案...")
    answers["B"] = answer_generator.generate_path_b_answer(disease)
    
    print("  生成路径C答案...")
    answers["C"] = answer_generator.generate_path_c_answer(disease)
    
    # 检查答案长度
    for path, answer in answers.items():
        print(f"    路径{path}答案长度: {len(answer)} 字符")
    
    # 步骤3: 分层融合投票
    voting_result = voting_system.integrated_voting(disease, answers, dynamic_route)
    
    # 步骤4: 选择最终方案（直接选择最优路径的答案）
    print("\n步骤4: 选择最终方案...")
    final_plan = selector.select_final_plan(
        disease, features, dynamic_route, voting_result, answers
    )
    
    # 整理结果
    result = {
        "disease": disease,
        "features": features,
        "dynamic_route": dynamic_route,
        "dynamic_route_desc": route_desc,
        "answers": answers,
        "voting_result": voting_result,
        "final_plan": final_plan
    }
    
    # 输出摘要
    print(f"\n{'='*60}")
    print(f"疾病: {disease}")
    print(f"动态路由: 路径{dynamic_route} - {route_desc}")
    print(f"投票最佳: 路径{voting_result['best_path']} ({voting_result['best_score']}分)")
    print(f"最终方案长度: {len(final_plan)} 字符")
    print(f"方案预览: {final_plan[:200]}...")
    print('='*60)
    
    return result

# ------------------- 主程序 --------------------
def main():
    """主程序"""
    print("初始化系统组件...")
    
    # 初始化各个系统组件
    routing_system = DynamicRoutingSystem(model, tokenizer, device)
    answer_generator = PathAnswerGenerator(model, tokenizer, device)
    voting_system = HierarchicalVotingSystem(model, tokenizer, device)
    selector = FinalPlanSelector()
    
    print("系统初始化完成!")
    
    # 疾病列表
    diseases = ["高血压"]
    
    # CSV文件头部
    csv_header = [
        '疾病', 
        '病情特征', 
        '动态路由推荐', 
        '投票最佳路径', 
        '投票综合评分',
        '各路径内容质量分(A/B/C)', 
        '各路径一致性分(A/B/C)', 
        '各路径对齐分(A/B/C)',
        '最终治疗方案'
    ]
    
    # 检查文件是否存在
    write_header = not os.path.exists(csv_output_path)
    
    # 存储所有结果
    all_results = []
    
    # 打开CSV文件
    with open(csv_output_path, 'a', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if write_header:
            csv_writer.writerow(csv_header)
        
        # 处理每个疾病
        for disease in diseases:
            try:
                print(f"\n{'#'*60}")
                print(f"开始处理疾病: {disease}")
                print('#'*60)
                
                # 处理疾病
                result = process_disease(
                    disease, routing_system, answer_generator, 
                    voting_system, selector
                )
                
                all_results.append(result)
                
                # 准备CSV行数据
                voting_scores = result["voting_result"]["scores"]
                
                # 计算内容质量总分
                content_scores = voting_scores["content"]
                content_totals = {}
                for path in ["A", "B", "C"]:
                    if path in content_scores:
                        content_totals[path] = sum(content_scores[path].values())
                    else:
                        content_totals[path] = 0
                
                # 写入CSV
                csv_row = [
                    result["disease"],
                    json.dumps(result["features"], ensure_ascii=False),
                    f"{result['dynamic_route']} - {result['dynamic_route_desc']}",
                    result["voting_result"]["best_path"],
                    result["voting_result"]["best_score"],
                    f"{content_totals.get('A', 0):.1f}/{content_totals.get('B', 0):.1f}/{content_totals.get('C', 0):.1f}",
                    f"{voting_scores['consistency'].get('A', 0):.2f}/{voting_scores['consistency'].get('B', 0):.2f}/{voting_scores['consistency'].get('C', 0):.2f}",
                    f"{voting_scores['alignment'].get('A', 0):.2f}/{voting_scores['alignment'].get('B', 0):.2f}/{voting_scores['alignment'].get('C', 0):.2f}",
                    result["final_plan"]
                ]
                
                csv_writer.writerow(csv_row)
                print(f"已写入CSV: {disease}")
                
            except Exception as e:
                print(f"处理疾病 {disease} 时出错: {e}")
                import traceback
                traceback.print_exc()
                
                # 写入错误信息
                csv_writer.writerow([disease, "处理出错", "错误", "错误", "错误", "错误", "错误", "错误", str(e)])
    
    # 保存详细结果到JSON文件
    print(f"\n保存详细结果到JSON文件...")
    
    # 将结果转换为可JSON序列化的格式
    serializable_results = []
    for result in all_results:
        serializable_result = {
            "disease": result["disease"],
            "features": result["features"],
            "dynamic_route": result["dynamic_route"],
            "dynamic_route_desc": result["dynamic_route_desc"],
            "answers": result["answers"],
            "voting_result": result["voting_result"],
            "final_plan": result["final_plan"]
        }
        serializable_results.append(serializable_result)
    
    with open(json_output_path, 'a', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print("处理完成!")
    print(f"简略结果已写入: {csv_output_path}")
    print(f"详细结果已写入: {json_output_path}")
    print(f"处理疾病总数: {len(all_results)}")
    
    if all_results:
        print("\n统计信息:")
        for result in all_results:
            disease = result["disease"]
            dynamic = result["dynamic_route"]
            best = result["voting_result"]["best_path"]
            match = "✓" if dynamic == best else "✗"
            print(f"  {disease}: 动态路由={dynamic}, 投票最佳={best} {match}")
        
        # 计算动态路由准确率
        matches = sum(1 for r in all_results if r["dynamic_route"] == r["voting_result"]["best_path"])
        accuracy = matches / len(all_results) * 100
        print(f"\n动态路由与投票结果一致率: {accuracy:.1f}% ({matches}/{len(all_results)})")

# ------------------- 程序入口 --------------------
if __name__ == "__main__":
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存")
    
    main()
    
    # 再次清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("程序执行完毕，已清理GPU缓存!")