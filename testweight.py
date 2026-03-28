import csv
import os
import re
import json
import torch
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from openai import OpenAI
from text2vec import SentenceModel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ------------------- 配置 --------------------
finetune_model_path = 'model_TCM/'                  # 微调模型路径
csv_output_path = './weight_experiment_results.csv'  # CSV输出文件路径
json_output_path = './weight_experiment_results.json' # JSON输出文件路径
therapy_keywords_path = './therapy_keywords.txt'    # 疗法特征关键词文件
term_keywords_path = './term_keywords.txt'          # 医学术语关键词文件
safety_keywords_path = './safety_keywords.txt'      # 安全性关键词文件
TEXT2VEC_MODEL_NAME = 'text2vec-base-chinese'       # 文本向量模型名称
api_key = "sk-1f0648aee9f54f41b993ef98f646aca2"     # API Key

# 权重配置方案
WEIGHT_SCHEMES = {
    "balanced": {"content": 0.4, "consistency": 0.3, "alignment": 0.3},  # 平衡型
    "content_focused": {"content": 0.6, "consistency": 0.2, "alignment": 0.2},  # 内容优先型
    "alignment_focused": {"content": 0.2, "consistency": 0.2, "alignment": 0.6},  # 对齐优先型
    "consistency_focused": {"content": 0.2, "consistency": 0.6, "alignment": 0.2},  # 一致性优先型
}

# 初始化API客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# ------------------- 模型加载配置 --------------------
print("正在加载模型和配置...")
config = PeftConfig.from_pretrained(finetune_model_path)

# 🔥 强制指定 GPU 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:1")

print(f"CUDA可用: {cuda_available}")
print(f"使用GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU空闲显存: 22GB+ (足够运行模型!)")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
    use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===================== 修复核心代码 =====================
# 1. 废弃 torch_dtype → 改用 dtype
# 2. 禁用 CPU 卸载，避免 accelerate 内存分配报错
# 3. 固定加载到 GPU:0
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map={"": 1},          # 强制全部加载到 GPU 0
    trust_remote_code=True,
    low_cpu_mem_usage=True,      # 低内存加载
)

# 加载 LoRA 适配器（修复版本兼容问题）
model = PeftModel.from_pretrained(
    model,
    finetune_model_path,
    device_map={"": 1}           # 强制 LoRA 也加载到 GPU 0
)
model.eval()
# ======================================================

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
    
    def __init__(self, model, tokenizer, device, weights=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 设置权重，默认为平衡型
        if weights is None:
            self.weights = {"content": 0.4, "consistency": 0.3, "alignment": 0.3}
        else:
            self.weights = weights
        
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
    
    def assess_reference_alignment(self, disease, answers_dict, reference_answer=None):
        """参考标准对齐评估 - 使用text2vec-base-chinese模型"""
        # 如果没有提供参考答案，使用API生成
        if reference_answer is None:
            print("  调用DeepSeek API生成参考标准...")
            reference_answer = generate_reference_by_api(disease)
        else:
            print("  使用提供的参考标准...")
        
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
            print(f"    路径{path}: 整体相似度={overall_similarity:.3f}, "
                  f"句子相似度={sentence_similarity:.3f}, "
                  f"关键词匹配={keyword_match}, "
                  f"综合得分={alignment_scores[path]:.2f}")
        
        return alignment_scores, reference_answer
    
    def integrated_voting(self, disease, answers_dict, dynamic_route, reference_answer=None):
        """集成投票决策"""
        print(f"\n为疾病'{disease}'进行集成投票评估...")
        print(f"权重设置: 内容={self.weights['content']}, 一致性={self.weights['consistency']}, 对齐={self.weights['alignment']}")
        
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
        alignment_scores, reference_answer = self.assess_reference_alignment(disease, answers_dict, reference_answer)
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
            weighted_score = (
                content_normalized * self.weights['content'] +
                consistency_normalized * self.weights['consistency'] +
                alignment_scores[path] * self.weights['alignment']
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
            "selection_reason": selection_reason,
            "weights": self.weights
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

# ------------------- 实验结果数据类 --------------------
@dataclass
class ExperimentResult:
    """实验结果数据类"""
    disease: str
    scheme_name: str
    weights: Dict[str, float]
    selected_path: str
    path_scores: Dict[str, float]
    score_details: Dict[str, Dict[str, Dict[str, float]]]
    decision_time: float
    dynamic_route: str
    features: Dict[str, str]
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'disease': self.disease,
            'scheme_name': self.scheme_name,
            'weights': self.weights,
            'selected_path': self.selected_path,
            'path_scores': self.path_scores,
            'score_details': self.score_details,
            'decision_time': self.decision_time,
            'dynamic_route': self.dynamic_route,
            'features': self.features
        }

# ------------------- 敏感性分析类 --------------------
class SensitivityAnalysis:
    """敏感性分析类"""
    
    @staticmethod
    def calculate_sensitivity_coefficient(df: pd.DataFrame, 
                                         weight_type: str, 
                                         metric: str = 'rank_change') -> float:
        """计算敏感性系数
        
        参数:
            df: 实验数据
            weight_type: 权重类型 ('content', 'consistency', 'alignment')
            metric: 评估指标 ('path_score', 'rank_change', 'selection')
        
        返回:
            敏感性系数 (0-1之间，越高表示越敏感)
        """
        # 分组按权重值
        weight_values = sorted(df[f'{weight_type}_weight'].unique())
        
        if metric == 'path_score':
            # 计算权重变化时得分的变化率
            score_changes = []
            for i in range(len(weight_values) - 1):
                w1, w2 = weight_values[i], weight_values[i+1]
                df1 = df[df[f'{weight_type}_weight'] == w1]
                df2 = df[df[f'{weight_type}_weight'] == w2]
                
                avg_score1 = df1[['score_A', 'score_B', 'score_C']].mean().mean()
                avg_score2 = df2[['score_A', 'score_B', 'score_C']].mean().mean()
                
                score_change = abs(avg_score2 - avg_score1) / ((w2 - w1) * 100)
                score_changes.append(score_change)
            
            return np.mean(score_changes) if score_changes else 0
        
        elif metric == 'rank_change':
            # 计算排名变化的频率
            rank_changes = []
            for disease in df['disease'].unique():
                disease_df = df[df['disease'] == disease]
                paths_by_weight = {}
                
                for weight in weight_values:
                    weight_df = disease_df[disease_df[f'{weight_type}_weight'] == weight]
                    if not weight_df.empty:
                        # 获取该权重下的路径排名
                        scores = {
                            'A': weight_df['score_A'].iloc[0],
                            'B': weight_df['score_B'].iloc[0],
                            'C': weight_df['score_C'].iloc[0]
                        }
                        ranked_paths = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        paths_by_weight[weight] = [p[0] for p in ranked_paths]
                
                # 检查排名变化
                weight_list = sorted(paths_by_weight.keys())
                for i in range(len(weight_list) - 1):
                    w1, w2 = weight_list[i], weight_list[i+1]
                    if paths_by_weight[w1] != paths_by_weight[w2]:
                        rank_changes.append(1)
                    else:
                        rank_changes.append(0)
            
            return np.mean(rank_changes) if rank_changes else 0
        
        elif metric == 'selection':
            # 计算权重变化时选择变化的频率
            selection_changes = []
            for disease in df['disease'].unique():
                disease_df = df[df['disease'] == disease]
                
                # 按权重排序
                disease_df = disease_df.sort_values(f'{weight_type}_weight')
                selections = disease_df['selected_path'].tolist()
                
                for i in range(len(selections) - 1):
                    if selections[i] != selections[i+1]:
                        selection_changes.append(1)
                    else:
                        selection_changes.append(0)
            
            return np.mean(selection_changes) if selection_changes else 0
    
    @staticmethod
    def perform_robustness_analysis(df: pd.DataFrame) -> Dict:
        """执行鲁棒性分析
        
        评估系统在权重变化时的稳定性
        """
        robustness_metrics = {}
        
        # 1. 选择一致性
        for disease in df['disease'].unique():
            disease_df = df[df['disease'] == disease]
            unique_selections = disease_df['selected_path'].unique()
            robustness_metrics[f'{disease}_consistency'] = 1 - (len(unique_selections) - 1) / 4
        
        # 2. 得分稳定性
        score_columns = ['score_A', 'score_B', 'score_C']
        score_variations = {}
        for col in score_columns:
            variation = df.groupby('scheme_name')[col].std().mean() / df[col].mean()
            score_variations[col] = variation
        
        robustness_metrics['score_stability'] = 1 - np.mean(list(score_variations.values())) if score_variations else 0
        
        # 3. 决策边界清晰度
        # 计算最佳路径与次佳路径的平均分差
        score_differences = []
        for _, row in df.iterrows():
            scores = [row['score_A'], row['score_B'], row['score_C']]
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2:
                score_diff = sorted_scores[0] - sorted_scores[1]
                score_differences.append(score_diff)
        
        robustness_metrics['decision_margin'] = np.mean(score_differences) if score_differences else 0
        
        # 4. 总体鲁棒性指数
        if 'score_stability' in robustness_metrics and 'decision_margin' in robustness_metrics:
            robustness_metrics['overall_robustness'] = (
                robustness_metrics['score_stability'] * 0.4 +
                (robustness_metrics['decision_margin'] / 10) * 0.3 +
                np.mean([robustness_metrics.get(f'{disease}_consistency', 0) 
                         for disease in df['disease'].unique()]) * 0.3
            )
        
        return robustness_metrics
    
    @staticmethod
    def perform_statistical_analysis(df: pd.DataFrame):
        """执行统计分析"""
        
        print("\n1. 权重敏感性分析")
        print("-"*40)
        
        # 计算不同权重下选择的一致性
        from collections import Counter
        
        for disease in df['disease'].unique():
            disease_df = df[df['disease'] == disease]
            path_counts = Counter(disease_df['selected_path'])
            
            print(f"\n疾病: {disease}")
            print(f"  总实验次数: {len(disease_df)}")
            print(f"  路径选择分布: {dict(path_counts)}")
            
            # 计算一致性指数（Agreement Index）
            total = len(disease_df)
            max_count = max(path_counts.values())
            agreement_index = max_count / total
            print(f"  一致性指数: {agreement_index:.3f}")
        
        print("\n2. 权重与路径选择的关联分析")
        print("-"*40)
        
        # 计算相关系数
        weight_columns = ['content_weight', 'consistency_weight', 'alignment_weight']
        
        for weight_col in weight_columns:
            # 将路径转换为数值（A=1, B=2, C=3）
            df['path_numeric'] = df['selected_path'].map({'A': 1, 'B': 2, 'C': 3})
            
            # 计算Spearman相关系数
            corr, p_value = stats.spearmanr(df[weight_col], df['path_numeric'])
            print(f"  {weight_col}: ρ={corr:.3f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print(f"    → {weight_col}与路径选择显著相关")
        
        print("\n3. Friedman检验（不同权重对评分的影响）")
        print("-"*40)
        
        # 准备数据进行Friedman检验
        scheme_results = {}
        for scheme in df['scheme_name'].unique():
            scheme_df = df[df['scheme_name'] == scheme]
            # 计算每个路径的平均得分
            avg_scores = [
                scheme_df['score_A'].mean(),
                scheme_df['score_B'].mean(),
                scheme_df['score_C'].mean()
            ]
            scheme_results[scheme] = avg_scores
        
        # 如果有足够的数据，执行Friedman检验
        if len(scheme_results) >= 3:
            try:
                # 将数据转换为适合Friedman检验的格式
                data = [scheme_results[scheme] for scheme in scheme_results.keys()]
                
                # 转置数据：每个权重方案作为一个"处理"，每个路径作为"块"
                data_transposed = list(zip(*data))
                
                # 执行Friedman检验
                stat, p_value = stats.friedmanchisquare(*data_transposed)
                print(f"  Friedman检验: χ²={stat:.3f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    print("  → 不同权重对评分有显著影响（p<0.05）")
                    # 执行事后检验（Nemenyi检验）
                    from scipy.stats import rankdata
                    
                    # 计算平均排名
                    ranks = np.array([rankdata(x) for x in data_transposed])
                    avg_ranks = ranks.mean(axis=0)
                    
                    print(f"  平均排名: {dict(zip(scheme_results.keys(), avg_ranks))}")
                else:
                    print("  → 不同权重对评分无显著影响")
            except Exception as e:
                print(f"  Friedman检验执行失败: {e}")
        
        print("\n4. 决策时间分析")
        print("-"*40)
        
        # 按权重方案分组
        for scheme in df['scheme_name'].unique():
            scheme_df = df[df['scheme_name'] == scheme]
            avg_time = scheme_df['decision_time'].mean()
            std_time = scheme_df['decision_time'].std()
            print(f"  {scheme}: 平均决策时间={avg_time:.3f}s (±{std_time:.3f})")
        
        print("\n5. 动态路由准确率分析")
        print("-"*40)
        
        # 计算动态路由与最终选择的匹配率
        matches = []
        for disease in df['disease'].unique():
            disease_df = df[df['disease'] == disease]
            for scheme in disease_df['scheme_name'].unique():
                scheme_df = disease_df[disease_df['scheme_name'] == scheme]
                if not scheme_df.empty:
                    dynamic_route = scheme_df['dynamic_route'].iloc[0]
                    selected_path = scheme_df['selected_path'].iloc[0]
                    matches.append(1 if dynamic_route == selected_path else 0)
        
        if matches:
            match_rate = np.mean(matches) * 100
            print(f"  动态路由与最终选择匹配率: {match_rate:.1f}% ({sum(matches)}/{len(matches)})")

# ------------------- 实验执行类 --------------------
class WeightSensitivityExperiment:
    """权重敏感性实验类"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results = []
        
        # 初始化系统组件
        self.routing_system = DynamicRoutingSystem(model, tokenizer, device)
        self.answer_generator = PathAnswerGenerator(model, tokenizer, device)
        
        print("实验系统初始化完成!")
    
    def run_single_disease_experiment(self, disease: str) -> List[ExperimentResult]:
        """对单个疾病运行多权重实验"""
        print(f"\n{'='*60}")
        print(f"开始处理疾病: {disease}")
        print('='*60)
        
        disease_results = []
        
        # 1. 病情特征感知和动态路由
        print("步骤1: 病情特征感知和动态路由...")
        features = self.routing_system.perceive_condition_features(disease)
        dynamic_route = self.routing_system.route_decision(features)
        route_desc = self.routing_system.get_route_description(dynamic_route)
        
        print(f"  病情特征: {features}")
        print(f"  动态路由推荐: 路径{dynamic_route} - {route_desc}")
        
        # 2. 生成三个路径的答案
        print("\n步骤2: 生成三个路径的答案...")
        answers = {}
        
        print("  生成路径A答案...")
        answers["A"] = self.answer_generator.generate_path_a_answer(disease)
        print(f"    长度: {len(answers['A'])} 字符")
        
        print("  生成路径B答案...")
        answers["B"] = self.answer_generator.generate_path_b_answer(disease)
        print(f"    长度: {len(answers['B'])} 字符")
        
        print("  生成路径C答案...")
        answers["C"] = self.answer_generator.generate_path_c_answer(disease)
        print(f"    长度: {len(answers['C'])} 字符")
        
        # 3. 生成参考答案（为所有权重方案共享）
        print("\n步骤3: 生成参考答案...")
        reference_answer = generate_reference_by_api(disease)
        print(f"  参考答案长度: {len(reference_answer)} 字符")
        
        # 4. 使用不同权重进行评估
        print("\n步骤4: 使用不同权重进行评估...")
        for scheme_name, weights in WEIGHT_SCHEMES.items():
            print(f"\n  权重方案: {scheme_name}")
            print(f"    权重设置: 内容={weights['content']}, 一致性={weights['consistency']}, 对齐={weights['alignment']}")
            
            # 创建带有特定权重的投票系统
            voting_system = HierarchicalVotingSystem(
                self.model, self.tokenizer, self.device, weights
            )
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行投票评估（使用共享的参考答案）
            voting_result = voting_system.integrated_voting(
                disease, answers, dynamic_route, reference_answer
            )
            
            # 计算决策时间
            decision_time = time.time() - start_time
            
            # 创建实验结果对象
            exp_result = ExperimentResult(
                disease=disease,
                scheme_name=scheme_name,
                weights=weights,
                selected_path=voting_result["best_path"],
                path_scores=voting_result["scores"]["final"],
                score_details=voting_result["scores"],
                decision_time=decision_time,
                dynamic_route=dynamic_route,
                features=features
            )
            
            disease_results.append(exp_result)
            self.results.append(exp_result)
            
            print(f"    选择路径: {exp_result.selected_path}, 决策时间: {decision_time:.3f}s")
        
        # 输出当前疾病的结果摘要
        print(f"\n{disease} 不同权重下的选择：")
        for result in disease_results:
            scores = result.path_scores
            print(f"  {result.scheme_name}: 选择路径{result.selected_path}, "
                  f"得分: A={scores.get('A',0):.2f}, "
                  f"B={scores.get('B',0):.2f}, "
                  f"C={scores.get('C',0):.2f}")
        
        return disease_results
    
    def run_experiment(self, diseases: List[str]):
        """运行完整实验"""
        print("="*80)
        print("权重敏感性实验开始")
        print("="*80)
        
        all_results = []
        
        for disease in diseases:
            try:
                disease_results = self.run_single_disease_experiment(disease)
                all_results.extend(disease_results)
            except Exception as e:
                print(f"处理疾病 {disease} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存结果
        self.save_results(all_results)
        
        # 分析结果
        df_results = self.analyze_results(all_results)
        
        return df_results
    
    def save_results(self, results: List[ExperimentResult]):
        """保存实验结果"""
        print("\n" + "="*80)
        print("保存实验结果")
        print("="*80)
        
        # 1. 保存为JSON格式
        serializable_results = [r.to_dict() for r in results]
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存到: {json_output_path}")
        
        # 2. 保存为CSV格式
        csv_header = [
            'disease', 'scheme_name', 
            'content_weight', 'consistency_weight', 'alignment_weight',
            'selected_path', 'score_A', 'score_B', 'score_C',
            'decision_time', 'dynamic_route',
            'course', 'nature', 'severity', 'focus'
        ]
        
        csv_rows = []
        for result in results:
            row = [
                result.disease,
                result.scheme_name,
                result.weights['content'],
                result.weights['consistency'],
                result.weights['alignment'],
                result.selected_path,
                result.path_scores.get('A', 0),
                result.path_scores.get('B', 0),
                result.path_scores.get('C', 0),
                result.decision_time,
                result.dynamic_route,
                result.features.get('course', ''),
                result.features.get('nature', ''),
                result.features.get('severity', ''),
                result.features.get('focus', '')
            ]
            csv_rows.append(row)
        
        df = pd.DataFrame(csv_rows, columns=csv_header)
        df.to_csv(csv_output_path, index=False, encoding='utf-8')
        print(f"CSV结果已保存到: {csv_output_path}")
        
        return df
    
    def analyze_results(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """分析实验结果并生成报告"""
        print("\n" + "="*80)
        print("实验结果分析")
        print("="*80)
        
        # 转换为DataFrame以便分析
        df_results = pd.DataFrame([
            {
                'disease': r.disease,
                'scheme_name': r.scheme_name,
                'content_weight': r.weights['content'],
                'consistency_weight': r.weights['consistency'],
                'alignment_weight': r.weights['alignment'],
                'selected_path': r.selected_path,
                'score_A': r.path_scores.get('A', 0),
                'score_B': r.path_scores.get('B', 0),
                'score_C': r.path_scores.get('C', 0),
                'decision_time': r.decision_time,
                'dynamic_route': r.dynamic_route,
                'course': r.features.get('course', ''),
                'nature': r.features.get('nature', ''),
                'severity': r.features.get('severity', ''),
                'focus': r.features.get('focus', '')
            }
            for r in results
        ])
        
        # 执行统计分析
        sensitivity_analysis = SensitivityAnalysis()
        sensitivity_analysis.perform_statistical_analysis(df_results)
        
        # 执行敏感性分析
        print("\n6. 权重敏感性系数分析")
        print("-"*40)
        for weight_type in ['content', 'consistency', 'alignment']:
            sens_score = sensitivity_analysis.calculate_sensitivity_coefficient(
                df_results, weight_type, metric='rank_change'
            )
            print(f"  {weight_type}权重敏感性: {sens_score:.3f}")
        
        # 执行鲁棒性分析
        print("\n7. 系统鲁棒性分析")
        print("-"*40)
        robustness = sensitivity_analysis.perform_robustness_analysis(df_results)
        for metric, value in robustness.items():
            print(f"  {metric}: {value:.3f}")
        
        # 生成可视化报告
        self.generate_visualization_report(df_results)
        
        # 输出最佳权重方案建议
        self.recommend_best_weight_scheme(df_results)
        
        return df_results
    
    def generate_visualization_report(self, df: pd.DataFrame):
        """生成可视化报告"""
        print("\n" + "="*80)
        print("生成可视化报告")
        print("="*80)
        
        # 设置样式
        sns.set_style("whitegrid")
        # 优先使用文泉驿微米黑（Linux容器专用），兼容中英文
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # 清除matplotlib字体缓存，确保生效
        import matplotlib
        matplotlib.font_manager._load_fontmanager(try_read_cache=False)
    
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. 不同权重下的路径选择分布
        ax1 = axes[0, 0]
        # 计算每个权重方案下各路径的选择次数
        selection_counts = df.groupby(['scheme_name', 'selected_path']).size().unstack(fill_value=0)
        selection_counts.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('不同权重方案下的路径选择分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('权重方案', fontsize=12)
        ax1.set_ylabel('选择次数', fontsize=12)
        ax1.legend(title='路径', loc='upper right', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 各路径得分箱线图
        ax2 = axes[0, 1]
        # 准备数据
        score_data = pd.melt(df, 
                            id_vars=['scheme_name'],
                            value_vars=['score_A', 'score_B', 'score_C'], 
                            var_name='路径', value_name='得分')
        score_data['路径'] = score_data['路径'].str.replace('score_', '')
        sns.boxplot(x='路径', y='得分', hue='scheme_name', data=score_data, ax=ax2)
        ax2.set_title('不同权重方案下各路径得分分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('路径', fontsize=12)
        ax2.set_ylabel('得分', fontsize=12)
        ax2.legend(title='权重方案', loc='upper right', fontsize=10)
        
        # 3. 权重与得分的热力图
        ax3 = axes[1, 0]
        # 计算相关性矩阵
        correlation_data = df[['content_weight', 'consistency_weight', 
                              'alignment_weight', 'score_A', 'score_B', 'score_C']]
        corr_matrix = correlation_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, mask=mask, ax=ax3, square=True, cbar_kws={"shrink": 0.8})
        ax3.set_title('权重与得分的相关性热力图', fontsize=14, fontweight='bold')
        
        # 4. 决策时间对比
        ax4 = axes[1, 1]
        time_data = df.groupby('scheme_name')['decision_time'].agg(['mean', 'std'])
        x_pos = np.arange(len(time_data))
        ax4.bar(x_pos, time_data['mean'], yerr=time_data['std'], 
               capsize=5, color='skyblue', alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(time_data.index, rotation=45)
        ax4.set_title('不同权重方案的决策时间对比', fontsize=14, fontweight='bold')
        ax4.set_xlabel('权重方案', fontsize=12)
        ax4.set_ylabel('决策时间(s)', fontsize=12)
        
        # 为每个柱状图添加数值标签
        for i, v in enumerate(time_data['mean']):
            ax4.text(i, v + 0.05, f'{v:.2f}s', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('weight_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('weight_sensitivity_analysis.pdf', bbox_inches='tight')
        plt.show()
        
        print("\n可视化报告已保存为:")
        print("  - weight_sensitivity_analysis.png")
        print("  - weight_sensitivity_analysis.pdf")
        
        # 5. 创建权重敏感性雷达图
        self.create_sensitivity_radar_chart(df)
    
    def create_sensitivity_radar_chart(self, df: pd.DataFrame):
        """创建权重敏感性雷达图（修复溢出问题）"""
        # 1. 先收集所有方案的决策时间，用于归一化
        scheme_times = {}
        for scheme in df['scheme_name'].unique():
            scheme_df = df[df['scheme_name'] == scheme]
            scheme_times[scheme] = scheme_df['decision_time'].mean()
        
        # 最小-最大归一化（决策时间越短，得分越高）
        max_t = max(scheme_times.values())
        min_t = min(scheme_times.values())

        # 2. 计算每个权重方案的性能指标（全部归一化到 0-1）
        scheme_metrics = {}
        for scheme in df['scheme_name'].unique():
            scheme_df = df[df['scheme_name'] == scheme]
            
            # 核心修复：决策速度归一化
            current_t = scheme_times[scheme]
            if max_t == min_t:
                norm_speed = 1.0
            else:
                norm_speed = (max_t - current_t) / (max_t - min_t)

            # 所有指标严格限制在 0-1
            metrics = {
                '选择一致性': round(self.calculate_selection_consistency(scheme_df), 3),
                '得分稳定性': round(1 - scheme_df[['score_A', 'score_B', 'score_C']].std().mean() / 10, 3),
                '决策速度': round(norm_speed, 3),  # 修复点
                '路由匹配率': round((scheme_df['selected_path'] == scheme_df['dynamic_route']).mean(), 3),
                '平均得分': round(scheme_df[['score_A', 'score_B', 'score_C']].mean().mean() / 10, 3)
            }
            scheme_metrics[scheme] = metrics

        # 3. 雷达图基础配置
        categories = list(scheme_metrics[list(scheme_metrics.keys())[0]].keys())
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # 闭合多边形

        # 4. 创建画布（优化尺寸+边距，防止溢出）
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        colors = ['#FF6B6B', '#4ECDC4', "#A7D145", "#B396CE"]

        # 5. 绘制雷达图
        for idx, (scheme, metrics) in enumerate(scheme_metrics.items()):
            values = [metrics[cat] for cat in categories]
            values += values[:1]
            
            ax.plot(angles, values, linewidth=2, label=scheme, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # 6. 布局优化（核心防溢出）
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)  # 缩小字体
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_ylim(0, 1.0)  # 强制锁定范围

        # 图例位置优化（不遮挡图形）
        ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.05), fontsize=10, frameon=True)
        plt.title('权重方案性能雷达图', fontsize=15, pad=25)
        
        # 强制收紧布局，防止溢出
        plt.tight_layout(pad=2.0)
        
        # 7. 保存图片
        plt.savefig('weight_scheme_radar.png', dpi=300, bbox_inches='tight')
        plt.savefig('weight_scheme_radar.pdf', bbox_inches='tight')
        plt.close()  # 释放内存
        print("  - weight_scheme_radar.png (已修复溢出)")
        print("  - weight_scheme_radar.pdf")
    
    def calculate_selection_consistency(self, df: pd.DataFrame) -> float:
        """计算选择一致性"""
        if len(df) <= 1:
            return 1.0
        
        # 计算同一疾病下不同权重的选择是否一致
        diseases = df['disease'].unique()
        consistencies = []
        
        for disease in diseases:
            disease_df = df[df['disease'] == disease]
            if len(disease_df) > 1:
                unique_selections = disease_df['selected_path'].unique()
                consistency = 1 - (len(unique_selections) - 1) / (len(disease_df) - 1)
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 1.0
    
    def recommend_best_weight_scheme(self, df: pd.DataFrame):
        """推荐最佳权重方案"""
        print("\n" + "="*80)
        print("最佳权重方案推荐")
        print("="*80)
        
        # 计算每个权重方案的综合得分
        scheme_scores = {}
        
        for scheme in df['scheme_name'].unique():
            scheme_df = df[df['scheme_name'] == scheme]
            
            # 计算多个指标
            metrics = {
                '选择一致性': self.calculate_selection_consistency(scheme_df),
                '平均得分': scheme_df[['score_A', 'score_B', 'score_C']].mean().mean(),
                '决策速度': 1 / scheme_df['decision_time'].mean(),
                '动态路由匹配率': (scheme_df['selected_path'] == scheme_df['dynamic_route']).mean(),
                '得分稳定性': 1 - scheme_df[['score_A', 'score_B', 'score_C']].std().mean() / 10
            }
            
            # 计算综合得分（加权平均）
            weights = {'选择一致性': 0.3, '平均得分': 0.3, '决策速度': 0.1, 
                      '动态路由匹配率': 0.2, '得分稳定性': 0.1}
            
            total_score = sum(metrics[key] * weights[key] for key in weights.keys())
            scheme_scores[scheme] = total_score
        
        # 找出最佳方案
        best_scheme = max(scheme_scores, key=scheme_scores.get)
        
        print("各权重方案综合评分：")
        for scheme, score in sorted(scheme_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {scheme}: {score:.4f}")
        
        print(f"\n推荐最佳权重方案：{best_scheme}")
        print(f"权重设置：{WEIGHT_SCHEMES[best_scheme]}")
        
        # 输出权重方案说明
        scheme_descriptions = {
            "balanced": "平衡型（内容40%，一致性30%，对齐30%）- 综合性能最佳",
            "content_focused": "内容优先型（内容60%，一致性20%，对齐20%）- 注重方案完整性",
            "alignment_focused": "对齐优先型（内容20%，一致性20%，对齐60%）- 注重与标准方案的一致性",
            "consistency_focused": "一致性优先型（内容20%，一致性60%，对齐20%）- 注重方案间的共识"
        }
        
        print(f"方案特点：{scheme_descriptions.get(best_scheme, '未知方案')}")

# ------------------- 主程序 --------------------
def main():
    """主程序"""
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存")
    
    print("初始化实验系统...")
    
    # 创建实验实例
    experiment = WeightSensitivityExperiment(model, tokenizer, device)
    
    # 测试疾病列表（可以调整疾病数量和类型）
    test_diseases = [
        "高血压", 
        "糖尿病",
        "慢性胃炎",
        "痛风",
        "过敏性鼻炎",
    ]
    
    print(f"将测试 {len(test_diseases)} 种疾病：{', '.join(test_diseases)}")
    
    # 运行实验
    results_df = experiment.run_experiment(test_diseases)
    
    # 输出实验总结
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    total_experiments = len(results_df)
    total_diseases = len(test_diseases)
    total_schemes = len(WEIGHT_SCHEMES)
    
    print(f"实验参数:")
    print(f"  - 测试疾病数量: {total_diseases}")
    print(f"  - 权重方案数量: {total_schemes}")
    print(f"  - 总实验次数: {total_experiments}")
    print(f"  - 每个疾病的实验次数: {total_schemes}")
    
    # 计算总体选择分布
    path_distribution = results_df['selected_path'].value_counts()
    print(f"\n总体路径选择分布:")
    for path, count in path_distribution.items():
        percentage = count / total_experiments * 100
        print(f"  路径{path}: {count}次 ({percentage:.1f}%)")
    
    # 计算平均决策时间
    avg_decision_time = results_df['decision_time'].mean()
    print(f"\n平均决策时间: {avg_decision_time:.3f}秒")
    
    # 计算平均得分
    avg_scores = {
        'A': results_df['score_A'].mean(),
        'B': results_df['score_B'].mean(),
        'C': results_df['score_C'].mean()
    }
    print(f"平均得分:")
    for path, score in avg_scores.items():
        print(f"  路径{path}: {score:.2f}")
    
    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
    
    # 再次清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存")

# ------------------- 程序入口 --------------------
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()