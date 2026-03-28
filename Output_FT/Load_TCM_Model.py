import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# LoRA微调模型路径
finetune_model_path = 'model_TCM/'

# 加载LoRA配置
config = PeftConfig.from_pretrained(finetune_model_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 确定设备
device_map = "cuda:3" if torch.cuda.is_available() else "cpu"  # 修改为cpu或指定其他可用的cuda设备

# 加载基础模型并应用LoRA微调
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 确保在CPU上使用float32
    load_in_8bit=torch.cuda.is_available(),  # 仅在CUDA上启用8位加载
    trust_remote_code=True,
    use_flash_attention_2=torch.cuda.is_available()  # 仅在CUDA上启用flash attention
)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model = model.eval()

# 疾病名称列表
diseases = []

# 准备CSV输出
output_file_path = 'output_TCM.csv'
header = ['疾病', '治疗方法']

# 打开CSV文件并写入数据
with open(output_file_path, 'a', encoding='utf-8', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)  # 写入表头

    for disease in diseases:
        # 准备输入数据
        input_text = f"给出{disease}的传统中医治疗方法。"
        input_ids = tokenizer([input_text], return_tensors="pt", add_special_tokens=False).input_ids

        # 将输入数据移至GPU（如适用）
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        # 配置生成参数
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }
        generate_ids = model.generate(**generate_input)
        text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

        # 将生成的文本写入CSV文件
        csv_writer.writerow([disease, text])

print(f"生成的文本已写入CSV文件: {output_file_path}")
