import csv
import os
import time
from zhipuai import ZhipuAI
import pandas as pd
from rouge import Rouge
import jieba

# ================= 配置部分 =================
val_csv_path = './DataSets/val_WM_2.csv'
progress_file = './Contrast_Experiment/progress_chatglm.txt'                    
result_csv_path = './Contrast_Experiment/ChatGLM_evaluation_WM.csv'
api_key = "0a3da81236ab4c439dc9ca802a998663.VZFnQ1HCS1J4YjlD"

# 初始化API客户端
client = ZhipuAI(api_key=api_key)

# 加载验证数据
val_data = pd.read_csv(val_csv_path).dropna(subset=['Prompt', 'Completion']).reset_index(drop=True)

# 初始化进度跟踪
processed_indices = set()
if os.path.exists(progress_file):
    with open(progress_file, 'r', encoding='utf-8') as f:
        processed_indices = set(int(line.strip()) for line in f if line.strip().isdigit())

# 初始化结果文件
if not os.path.exists(result_csv_path):
    with open(result_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['disease', 'f1', 'rouge1', 'rouge2', 'rougeL'])
        writer.writeheader()

# 初始化评估工具
rouge = Rouge()

for idx, row in val_data.iterrows():
    if idx in processed_indices:
        continue
    
    try:
        disease = row['Prompt']
        reference = row['Completion']
        
        # API调用参数
        messages = [{"role": "user", "content": disease}]
        
        response = None
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=messages,
                    temperature=0.3,
                    top_p=0.95,
                    max_tokens=512
                )
                break
            except Exception as e:
                print(f"API调用失败（尝试 {retry_count+1}/{max_retries}）: {str(e)}")
                retry_count += 1
                time.sleep(2)
        
        # 响应解析方式
        if not response or not response.choices:
            prediction = ""
        else:
            prediction = response.choices[0].message.content

        # 预处理和评估逻辑
        ref_words = list(jieba.cut(reference.strip()))
        pred_words = list(jieba.cut(prediction.strip()))

        try:
            rouge_scores = rouge.get_scores(' '.join(pred_words), ' '.join(ref_words))[0]
            rouge1 = rouge_scores['rouge-1']['f']
            rouge2 = rouge_scores['rouge-2']['f']
            rougeL = rouge_scores['rouge-l']['f']
        except:
            rouge1 = rouge2 = rougeL = 0.0

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
            'rougeL': rougeL
        }
        
        with open(result_csv_path, 'a', newline='', encoding='utf-8-sig') as rf:
            writer = csv.DictWriter(rf, fieldnames=result_row.keys())
            writer.writerow(result_row)

        with open(progress_file, 'a', encoding='utf-8') as pf:
            pf.write(f"{idx}\n")
        processed_indices.add(idx)

    except Exception as e:
        print(f"处理索引 {idx} 时发生错误: {str(e)}")
        continue

# 后续统计和输出逻辑
if os.path.exists(result_csv_path):
    result_df = pd.read_csv(result_csv_path)
    avg_metrics = {
        'F1': result_df['f1'].mean(),
        'Rouge-1': result_df['rouge1'].mean(),
        'Rouge-2': result_df['rouge2'].mean(),
        'Rouge-L': result_df['rougeL'].mean()
    }
else:
    avg_metrics = {'F1': 0, 'Rouge-1': 0, 'Rouge-2': 0, 'Rouge-L': 0}

print("\n================= ChatGLM 评估结果 =================")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")
print("\n详细结果已保存至", result_csv_path)

if os.path.exists(result_csv_path):
    result_df = pd.read_csv(result_csv_path)
    
    if result_df['disease'].str.contains('[Average]').any():
        result_df = result_df[~result_df['disease'].str.contains('[Average]')]
    
    avg_row = {
        'disease': '[Average]',
        'f1': avg_metrics['F1'],
        'rouge1': avg_metrics['Rouge-1'],
        'rouge2': avg_metrics['Rouge-2'],
        'rougeL': avg_metrics['Rouge-L']
    }
    
    result_df = pd.concat([result_df, pd.DataFrame([avg_row])], ignore_index=True)
    result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')