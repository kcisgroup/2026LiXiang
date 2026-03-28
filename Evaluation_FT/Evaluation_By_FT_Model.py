import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
import numpy as np

# 使用验证集评估模型输出

# ================= 配置部分 =================
finetune_model_path = 'model_TCM/'
val_csv_path = './DataSets/val_TCM.csv'
# ===========================================

# 初始化模型和分词器
config = PeftConfig.from_pretrained(finetune_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    load_in_8bit=torch.cuda.is_available(),
    trust_remote_code=True,
    use_flash_attention_2=torch.cuda.is_available()
)
model = PeftModel.from_pretrained(model, finetune_model_path)
model.eval()

# 加载验证数据
val_data = pd.read_csv(val_csv_path)

# 初始化评估工具
rouge = Rouge()
smoothie = SmoothingFunction().method4

# 存储所有结果
all_results = []

with torch.no_grad():
    for idx, row in val_data.iterrows():
        # 生成预测
        disease = row['疾病']
        reference = row['治疗方法']
        
        input_text = f"给出{disease}的传统中医治疗方法。"
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.3,
            eos_token_id=tokenizer.eos_token_id
        )
        
        prediction = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

        # 预处理文本
        ref_words = list(jieba.cut(reference.strip()))
        pred_words = list(jieba.cut(prediction.strip()))

        # 计算Rouge
        try:
            rouge_scores = rouge.get_scores(' '.join(pred_words), ' '.join(ref_words))[0]
            rouge1 = rouge_scores['rouge-1']['f']
            rouge2 = rouge_scores['rouge-2']['f']
            rougeL = rouge_scores['rouge-l']['f']
        except:
            rouge1 = rouge2 = rougeL = 0.0

        # 计算BLEU-4
        try:
            bleu4 = sentence_bleu(
                [ref_words],
                pred_words,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie
            )
        except:
            bleu4 = 0.0

        # 计算F1（基于unigram）
        ref_set = set(ref_words)
        pred_set = set(pred_words)
        
        tp = len(ref_set & pred_set)
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0
        recall = tp / len(ref_set) if len(ref_set) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        all_results.append({
            'disease': disease,
            'f1': f1,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'bleu4': bleu4
        })

# 计算平均指标
avg_metrics = {
    'F1': np.mean([x['f1'] for x in all_results]),
    'Rouge-1': np.mean([x['rouge1'] for x in all_results]),
    'Rouge-2': np.mean([x['rouge2'] for x in all_results]),
    'Rouge-L': np.mean([x['rougeL'] for x in all_results]),
    'BLEU-4': np.mean([x['bleu4'] for x in all_results])
}

# 打印结果
print("\n================= 评估结果 =================")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存详细结果
result_df = pd.DataFrame(all_results)
result_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
print("\n详细结果已保存至 evaluation_results.csv")