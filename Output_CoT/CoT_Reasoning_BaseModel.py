import csv
import os
import re
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


# -------------------模型加载配置-----------------------
# 基础模型路径
base_model_path = 'model/'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

# 确定设备
device_map = "cuda:3" if torch.cuda.is_available() else "cpu"               # 指定可用的cuda设备，cude不可用时使用cpu
device = torch.device(device_map)                                           # 获取设备对象

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 在CUDA上使用float16
    load_in_8bit=torch.cuda.is_available(),                                     # 在CUDA上启用8位加载
    trust_remote_code=True,
    use_flash_attention_2=torch.cuda.is_available()                             # 在CUDA上启用flash attention
)
model = model.eval()


# -------------------思维链模板-----------------------
cot_prompt_prefix = """你是一位精通中医和西医的医生，请根据以下疾病，给出中西医结合治疗方案。
请按照以下步骤思考:
1. 简要描述中医和西医分别对该疾病的理论基础与病因认识;
2. 分别从中医和西医的角度，分析针对该疾病的治疗原则与手段;
3. 分析对于该疾病中医和西医的药物与疗法特点对比;
4. 分析中医和西医对疾病特性的侧重，如慢性和急性如何治疗，有并发症如何治疗;
5. 分别从中医和西医的角度，分析针对该疾病的预防与康复理念;
6. 结合上述分析，提出一个综合的中西医结合治疗方案，包括治疗原则、中药处方、西药处方、生活方式建议、饮食建议等。
请用中文回答。
"""


# ------------------- 文件路径配置 -----------------------
output_file_path = './Output_CoT/CoT_output_Base_1_1.csv'
header = ['疾病', '治疗方法(中西医结合)']
write_header = not os.path.exists(output_file_path)


# ------------------- 疾病输入表单 -----------------------
diseases = ["高血压"]


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
    cleaned = re.sub(r'202.*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
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
            "max_new_tokens": 3072,
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
