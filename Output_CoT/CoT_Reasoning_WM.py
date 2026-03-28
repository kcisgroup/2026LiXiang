import csv
import os
import re
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


# -------------------模型加载配置-----------------------
# LoRA微调模型路径
finetune_model_path = 'model_WM/'

# 加载LoRA配置
config = PeftConfig.from_pretrained(finetune_model_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)

# 确定设备
device_map = "cuda:0" if torch.cuda.is_available() else "cpu"                   # 指定cuda设备
device = torch.device(device_map)                                               # 获取设备对象

# 加载基础模型并应用LoRA微调
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 在CUDA上使用float16
    load_in_8bit=torch.cuda.is_available(),                                     # 在CUDA上启用8位加载
    trust_remote_code=True,
    use_flash_attention_2=torch.cuda.is_available()                             # 在CUDA上启用flash attention
)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model = model.eval()


# -------------------思维链模板-----------------------
# 1.差异性分析思维链：
cot_prompt_prefix = """你是一位精通中医和西医的医生，请根据以下疾病，给出中西医结合治疗方案。
请按照以下步骤思考:
1. 简要描述中医和西医分别对该疾病的理论基础与病因认识;
2. 分别从中医和西医的角度，分析针对该疾病的治疗原则与手段;
3. 分析对于该疾病中医和西医的药物与疗法特点对比;
4. 分析中医和西医对疾病特性的侧重，如慢性和急性如何治疗，有并发症如何治疗;
5. 分别从中医和西医的角度，分析针对该疾病的预防与康复理念;
6. 结合上述分析，提出一个综合的中西医结合治疗方案，包括治疗原则、中药处方、西药处方、生活方式建议、饮食建议等。
"""

# 2.临床诊疗思维链：
# cot_prompt_prefix = """作为中西医结合专家，请系统化分析以下疾病并提供整合治疗方案。思考步骤应包含：
# 1. 中西医理论差异：对比两种医学体系对该病的病理机制解释；
# 2. 治疗路径分析：分别列出典型的中医治疗策略和西医临床路径；
# 3. 疗法特性比较：从作用机理、起效时间、副作用等维度对比；
# 4. 临床适应场景：针对不同病程阶段和并发症情况的适用性分析；
# 5. 预防康复体系：中西医在疾病管理周期中的不同干预重点；
# 6. 整合方案设计：基于上述分析制定阶梯式结合治疗方案。
# """

# 3.循证医学思维链：
# cot_prompt_prefix = """基于循证医学原则，对于以下患者病症给出中西医结合诊疗推导：
# 1. 理论依据：用现代医学和传统医学语言分别阐述疾病本质；
# 2. 证据评估：整理中西医各自的一线治疗方案及其证据；
# 3. 优劣对比：从疗效、安全性、经济性等角度比较；
# 4. 临床决策：根据疾病严重程度制定中西医协同治疗流程；
# 5. 全程管理：构建包含预防、治疗、康复的中西医结合干预；
# 6. 个体化方案：整合针灸、方剂、西药等多元治疗模块。
# """


# ------------------- 文件路径配置 -----------------------
output_file_path = './Output-CoT/CoT_output_WM_1_2.csv'
header = ['疾病', '治疗方法(中西医结合)']
write_header = not os.path.exists(output_file_path)


# ------------------- 疾病输入表单 -----------------------
diseases = ["伤寒", "温病", "哮证", "肺炎", "肺结核", "肺气肿", "胃痛", "水肿", "痿证", "头痛", "胸痛", "眩晕", "心悸",
            "失眠", "中风", "内伤发热", "高血压", "胃下垂", "小儿感冒", "小儿肺炎", "消化不良", "痈", "牛皮癣", "牙痛",
            "白内障", "咳嗽", "喘证", "呕吐", "泄泻", "月经不调", "产后腹痛", "病毒性肝炎", "便秘", "糖尿病", "痤疮",
            "湿疹", "骨折", "甲状腺功能亢进", "淋证", "缺铁性贫血", "颈椎病", "带状疱疹", "骨关节炎", "肾结石", "痛风",
            "过敏性鼻炎", "骨质疏松", "慢性胃炎", "冠心病", "红斑狼疮", "婴幼儿支气管炎"]


# ------------------- 文本处理函数 -----------------------
def clean_generated_text(generated_text):
    cleaned = BeautifulSoup(generated_text, "html.parser").get_text()               # 删除所有HTML标签及内容
    cleaned = re.sub(r'```.*', '', cleaned, flags=re.DOTALL)                        # 删除指定字符及其之后所有内容
    cleaned = re.sub(r'参考文献.*', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'来源.*', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'相关链接.*', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'作者.*', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'http.*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'Please.*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'End.*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'Reference.*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


# ------------------- 主处理逻辑 -----------------------
with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    if write_header:
        csv_writer.writerow(header)

    for disease in diseases:
        input_text = cot_prompt_prefix + f"疾病：{disease}。"

        # 编码输入
        encoding = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)

        # 配置生成参数
        generate_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 2048,
            "do_sample": True,                          # 是否进行采样，False表示使用贪心解码
            "top_k": 50,                                # Top-k 采样：只在概率最高的k个词中采样
            "top_p": 0.95,                              # Top-p (nucleus) 采样：只在概率累积达到p的词中采样
            "temperature": 0.3,                         # 温度参数：控制生成文本的随机性/创造性（越低越保守）
            "repetition_penalty": 1.3,                  # 重复惩罚：大于1的值会惩罚重复出现的token
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }

        print(f"--- 正在生成: {disease} ---")

        with torch.no_grad():
            generate_ids = model.generate(**generate_input)

        # 调试信息
        print(f"输出长度: {len(generate_ids[0])}")
        actual_new_tokens = len(generate_ids[0]) - len(input_ids[0])
        print(f"实际生成的新token数: {actual_new_tokens}")
        if actual_new_tokens > 0:
            print(f"Last generated token ID: {generate_ids[0][-1]}, EOS token ID: {tokenizer.eos_token_id}")

        # 解码和处理生成的文本
        text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        generated_text = text[len(input_text):].strip()                     # 从生成的文本中移除输入部分
        cleaned_generated_text = clean_generated_text(generated_text)       # 文本处理

        # 写入CSV文件
        csv_writer.writerow([disease, cleaned_generated_text])
        print(f"已完成: {disease}")

print(f"全部内容已写入: {output_file_path}")
