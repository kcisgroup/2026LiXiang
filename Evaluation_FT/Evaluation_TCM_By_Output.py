import pandas as pd
import jieba
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 读取参考文件并建立疾病映射
ref_df = pd.read_csv('./DataSets/raw_TCM.csv')
disease_map = {
    row['instruct']: row['output']
    for _, row in ref_df.iterrows()
}

# 读取待评估文件
hyp_df = pd.read_csv('./Output-FT/FT_Output_TCM.csv')

# 初始化评估工具
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
metrics = {
    'rouge-1': [], 'rouge-2': [], 'rouge-l': [],
    'bleu-4': [], 'f1': []
}

unmatched_diseases = []  # 记录未匹配的疾病

# 初始化平滑函数
smoother = SmoothingFunction().method1  # 使用method1平滑

for _, row in hyp_df.iterrows():
    disease = row['疾病']
    hyp_text = str(row['治疗方法'])
    
    # 获取参考文本
    ref_text = disease_map.get(disease, None)
    if not ref_text:
        print(f"警告：未找到疾病 '{disease}' 的参考数据")
        unmatched_diseases.append(disease)
        continue

    # 中文分词器
    def chinese_tokenizer(text):
        return list(jieba.cut(text))
    
    # 计算ROUGE
    rouge_scores = scorer.score(ref_text, hyp_text)
    metrics['rouge-1'].append(rouge_scores['rouge1'].fmeasure)
    metrics['rouge-2'].append(rouge_scores['rouge2'].fmeasure)
    metrics['rouge-l'].append(rouge_scores['rougeL'].fmeasure)

    # 计算BLEU-4（使用平滑）
    ref_tokens = [chinese_tokenizer(ref_text)]
    hyp_tokens = chinese_tokenizer(hyp_text)
    bleu = sentence_bleu(
        ref_tokens, 
        hyp_tokens, 
        weights=(0.25, 0.25, 0.25, 0.25),  # 均衡分配1-4gram权重
        smoothing_function=smoother  # 应用平滑
    )
    metrics['bleu-4'].append(bleu)

    # 计算词级F1
    ref_words = set(chinese_tokenizer(ref_text))
    hyp_words = set(chinese_tokenizer(hyp_text))
    tp = len(ref_words & hyp_words)
    fp = len(hyp_words - ref_words)
    fn = len(ref_words - hyp_words)
    
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    metrics['f1'].append(f1)

# 输出诊断信息
print("\n===== 数据匹配检查 =====")
print(f"参考文件包含 {len(disease_map)} 种疾病")
print(f"待评估文件包含 {len(hyp_df)} 条数据")
print(f"成功匹配 {len(hyp_df)-len(unmatched_diseases)} 条，未匹配 {len(unmatched_diseases)} 条")
if unmatched_diseases:
    print("未匹配疾病示例：", set(unmatched_diseases[:5]))

# 计算平均结果
avg_results = {
    'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-4', 'Word F1'],
    'Score': [
        sum(metrics['rouge-1'])/max(1, len(metrics['rouge-1'])),
        sum(metrics['rouge-2'])/max(1, len(metrics['rouge-2'])),
        sum(metrics['rouge-l'])/max(1, len(metrics['rouge-l'])),
        sum(metrics['bleu-4'])/max(1, len(metrics['bleu-4'])),
        sum(metrics['f1'])/max(1, len(metrics['f1']))
    ]
}

# 输出平均结果
print("\n===== 评估结果 =====")
for metric, score in zip(avg_results['Metric'], avg_results['Score']):
    print(f"{metric:<8}: {score:.4f}")

# 保存平均结果到CSV
avg_df = pd.DataFrame(avg_results)
avg_df.to_csv('Evaluation_TCM_By_Output.csv', index=False, encoding='utf-8-sig')