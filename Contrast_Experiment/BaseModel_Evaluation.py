import csv
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
import numpy as np

# ================= 配置部分 =================
base_model_path = './model/'
val_csv_path = './DataSets/val_WM_2.csv'
progress_file = './progress_base.txt'                    # 进度记录文件
result_csv_path = './baseline_evaluation_WM.csv'    # 结果文件路径
# ===========================================

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    use_fast=False,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    load_in_8bit=torch.cuda.is_available(),
    trust_remote_code=True,
    use_flash_attention_2=torch.cuda.is_available()
).eval()

# 加载验证数据（添加空值过滤）
val_data = pd.read_csv(val_csv_path).dropna(subset=['Prompt', 'Completion']).reset_index(drop=True)

# 初始化进度跟踪（断点恢复逻辑）
processed_indices = set()
if os.path.exists(progress_file):
    with open(progress_file, 'r', encoding='utf-8') as f:
        processed_indices = set(int(line.strip()) for line in f if line.strip().isdigit())

# 初始化结果文件（保留已有结果）
if not os.path.exists(result_csv_path):
    with open(result_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['disease', 'f1', 'rouge1', 'rouge2', 'rougeL', 'bleu4'])
        writer.writeheader()

# 初始化评估工具
rouge = Rouge()
smoothie = SmoothingFunction().method4

with torch.no_grad():
    for idx, row in val_data.iterrows():
        # 断点恢复检查
        if idx in processed_indices:
            continue
            
        try:
            # 生成预测
            disease = row['Prompt']
            reference = row['Completion']
            
            input_text = f"{disease}"
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

            # 写入结果
            result_row = {
                'disease': disease,
                'f1': f1,
                'rouge1': rouge1,
                'rouge2': rouge2,
                'rougeL': rougeL,
                'bleu4': bleu4
            }
            
            with open(result_csv_path, 'a', newline='', encoding='utf-8-sig') as rf:
                writer = csv.DictWriter(rf, fieldnames=result_row.keys())
                writer.writerow(result_row)

            # 更新进度
            with open(progress_file, 'a', encoding='utf-8') as pf:
                pf.write(f"{idx}\n")
            processed_indices.add(idx)

        except Exception as e:
            print(f"处理索引 {idx} 时发生错误: {str(e)}")
            continue

# 计算平均指标（从结果文件读取）
if os.path.exists(result_csv_path):
    result_df = pd.read_csv(result_csv_path)
    avg_metrics = {
        'F1': result_df['f1'].mean(),
        'Rouge-1': result_df['rouge1'].mean(),
        'Rouge-2': result_df['rouge2'].mean(),
        'Rouge-L': result_df['rougeL'].mean(),
        'BLEU-4': result_df['bleu4'].mean()
    }
else:
    avg_metrics = {'F1': 0, 'Rouge-1': 0, 'Rouge-2': 0, 'Rouge-L': 0, 'BLEU-4': 0}

# 打印结果
print("\n================= 基线模型评估结果 =================")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")
print("\n详细结果已保存至", result_csv_path)

if os.path.exists(result_csv_path):
    result_df = pd.read_csv(result_csv_path)
    
    # 创建平均值行（处理可能存在的旧平均值）
    if result_df['disease'].str.contains('[Average]').any():
        result_df = result_df[~result_df['disease'].str.contains('[Average]')]
    
    avg_row = {
        'disease': '[Average]',
        'f1': avg_metrics['F1'],
        'rouge1': avg_metrics['Rouge-1'],
        'rouge2': avg_metrics['Rouge-2'],
        'rougeL': avg_metrics['Rouge-L'],
        'bleu4': avg_metrics['BLEU-4']
    }
    
    result_df = pd.concat([result_df, pd.DataFrame([avg_row])], ignore_index=True)
    result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')