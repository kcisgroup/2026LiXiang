import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import os
import torch
import jieba

# ---------------------------------------- 配置参数 ----------------------------------------
MODEL_NAME = 'text2vec-base-chinese'

FILE_PATHS = [
    './Output-CoT/CoT_output_WM_1_1.csv',
    './Output-CoT/CoT_output_WM_2_1.csv',
    './Output-CoT/CoT_output_WM_3_1.csv'
]
output_csv_path = 'semantic_consistency_WM_CoTvsCoT_1.csv'

DISEASE_COLUMN = '疾病'
TEXT_COLUMN = '治疗方法(中西医结合)'

# ---------------------------------------- 文本清洗函数 ----------------------------------------
def basic_text_clean(text):

    # 处理非字符串输入 (例如 NaN)
    if not isinstance(text, str):
        return ""

    # 移除特定的、较长的模板/评论/问题/附加内容块
    text = re.sub(r'###\s*添加评论[\s\S]*?提供"', '', text, flags=re.MULTILINE)
    text = re.sub(r'###\s*问题[\s\S]*?###', '', text, flags=re.MULTILINE)
    text = re.sub(r'##\s*医学术语表达技巧训练[\s\S]*?(\n##|\Z)', '', text, flags=re.MULTILINE)
    text = re.sub(r'##?\s*附件下载[\s\S]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'##?\s*作者简介[\s\S]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\s*医学声明[\s\S]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\s*注释[:：\s\S]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'###\s*参考文献[\s\S]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'###\s*(结论|结语)[\s\S]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'#+\s*(注意事项|重要提示|重要提示：|注意事项：)[\s\S]*', '', text, flags=re.MULTILINE)

    # 移除指令性、引导性、提示性短语
    text = re.sub(r'###\s*请提供详细答案', '', text)
    text = re.sub(r'请根据以上要求进行回答[。，]?', '', text)
    text = re.sub(r'请注意，这里提供的是.*?。', '', text)
    text = re.sub(r'以上内容仅供参考.*?。', '', text)
    text = re.sub(r'上述内容仅供参考.*?。', '', text)
    text = re.sub(r'最后祝愿.*?。', '', text)
    text = re.sub(r'感谢阅读.*?。', '', text)
    text = re.sub(r'\(点击回应键.*?\)', '', text)
    text = re.sub(r'我们欢迎你的进一步探讨！[👇👌]*', '', text)
    text = re.sub(r'如果您希望获得更多反馈.*?个性化。', '', text)
    text = re.sub(r'如果您正在寻找关于此主题的更多信息.*?。','', text)
    text = re.sub(r'如果您遇到了这种情况，最好立即就医.*?。','', text)
    text = re.sub(r'请记住，.*?。','', text)
    text = re.sub(r'重要的是，.*?。','', text)
    text = re.sub(r'需要注意的是，.*?。','', text)
    text = re.sub(r'此方案旨在.*?。','', text)
    text = re.sub(r'本文作者并不承担任何责任.*?。','', text)

    # 移除 Markdown 标题标记和可能的列表标记
    text = re.sub(r'^[#]+[ \t]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[\.\)][\s\t]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[a-zA-Z][\.\)][\s\t]*', '', text, flags=re.MULTILINE)

    # 移除网址
    text = re.sub(r'https?://\S+', '', text)

    # 移除表情符号
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 移除多余的空格和换行符，确保文本的紧凑性
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n+', '\n', text).strip()
     
    return text

# ---------------------------------------- 分词函数 ----------------------------------------
def segment_text_jieba(text):
    if not isinstance(text, str) or not text.strip():
        return "" 
    seg_list = jieba.cut(text, cut_all=False)
    return " ".join(seg_list)

# ---------------------------------------- 主程序 ----------------------------------------

# 1. 加载数据
dataframes = {}
for i, file_path in enumerate(FILE_PATHS):
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 {file_path}")
        exit()
    try:
        df = pd.read_csv(file_path)
        if DISEASE_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
             print(f"错误: 文件 {file_path} 缺少必需的列 ('{DISEASE_COLUMN}' 或 '{TEXT_COLUMN}')")
             exit()
        
        col_name = f'text_cot_{i+1}'
        df.rename(columns={TEXT_COLUMN: col_name}, inplace=True)
        df[col_name] = df[col_name].apply(basic_text_clean)
        df[col_name] = df[col_name].apply(segment_text_jieba)
        
        df.set_index(DISEASE_COLUMN, inplace=True)
        dataframes[f'cot_{i+1}'] = df[[col_name]]
    except Exception as e:
        print(f"加载、处理或分词文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        exit()

# 2. 合并数据
merged_df = pd.concat(dataframes.values(), axis=1, join='outer')

# 3. 加载 Sentence Transformer 模型
print(f"\n正在加载 Sentence Transformer 模型: {MODEL_NAME} ...")
try:
    is_hub_id = not os.path.isdir(MODEL_NAME) and '/' not in MODEL_NAME and '\\' not in MODEL_NAME
    is_local_dir = os.path.isdir(MODEL_NAME)

    if is_hub_id:
        print(f"模型 '{MODEL_NAME}' 看起来像一个Hub ID，将尝试从网络或缓存加载。")
    elif is_local_dir:
        print(f"模型 '{MODEL_NAME}' 是一个本地路径，将尝试从该路径加载。")
    else:
        print(f"错误: 模型路径 '{MODEL_NAME}' 既不是有效的Hugging Face Hub ID，也不是一个存在的本地目录。")
        exit()
    
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"加载模型 {MODEL_NAME} 时出错: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 4. 计算语义相似度
results = []
print("\n开始计算语义相似度...")
for disease, row in merged_df.iterrows():
    texts_sources = {
        'cot1': row.get(f'text_cot_1'),
        'cot2': row.get(f'text_cot_2'),
        'cot3': row.get(f'text_cot_3')
    }

    valid_texts_dict = {}
    for k, v in texts_sources.items():
        if pd.notna(v) and str(v).strip():
            valid_texts_dict[k] = str(v) 
    
    if len(valid_texts_dict) < 2:
        print(f"疾病 '{disease}': 有效文本（清洗和分词后）少于2个，无法计算相似度。")
        missing_cots = [k for k, v_original in texts_sources.items() if k not in valid_texts_dict]
        results.append({
            DISEASE_COLUMN: disease,
            'similarity_1_vs_2': np.nan,
            'similarity_1_vs_3': np.nan,
            'similarity_2_vs_3': np.nan,
            'average_similarity': np.nan,
            'consistency_std_dev': np.nan,
            'missing_cots': missing_cots,
            'error': 'Insufficient valid texts'
        })
        continue

    try:
        cot_keys_present = list(valid_texts_dict.keys())
        text_list_to_encode = list(valid_texts_dict.values())
        numpy_embeddings = model.encode(text_list_to_encode, show_progress_bar=False) 
        embeddings = torch.tensor(numpy_embeddings)
    except Exception as e:
        print(f"疾病 '{disease}': 生成嵌入向量时出错: {e}")
        missing_cots = [k for k, v_original in texts_sources.items() if k not in valid_texts_dict]
        results.append({
            DISEASE_COLUMN: disease,
            'similarity_1_vs_2': np.nan,
            'similarity_1_vs_3': np.nan,
            'similarity_2_vs_3': np.nan,
            'average_similarity': np.nan,
            'consistency_std_dev': np.nan,
            'error': f"Embedding error: {str(e)}",
            'missing_cots': missing_cots
        })
        continue

    similarities_scores_for_row = {}
    pairwise_scores_list = []
    all_possible_comparisons = [('cot1', 'cot2'), ('cot1', 'cot3'), ('cot2', 'cot3')]

    for c1_key, c2_key in all_possible_comparisons:
        sim_key_name = f'similarity_{c1_key[-1]}_vs_{c2_key[-1]}'
        if c1_key in cot_keys_present and c2_key in cot_keys_present:
            idx1 = cot_keys_present.index(c1_key)
            idx2 = cot_keys_present.index(c2_key)
            score = util.pytorch_cos_sim(embeddings[idx1], embeddings[idx2]).item()
            similarities_scores_for_row[sim_key_name] = score
            pairwise_scores_list.append(score)
        else:
            similarities_scores_for_row[sim_key_name] = np.nan

    valid_scores_for_stats = [s for s in pairwise_scores_list if not np.isnan(s)] 
    avg_sim_for_disease = np.mean(valid_scores_for_stats) if valid_scores_for_stats else np.nan
    std_dev_for_disease = np.std(valid_scores_for_stats) if len(valid_scores_for_stats) > 1 else (0.0 if len(valid_scores_for_stats) == 1 else np.nan)

    result_row = {DISEASE_COLUMN: disease}
    result_row.update(similarities_scores_for_row)
    result_row['average_similarity'] = avg_sim_for_disease
    result_row['consistency_std_dev'] = std_dev_for_disease
    result_row['missing_cots'] = [k for k, v_original in texts_sources.items() if k not in valid_texts_dict]
    results.append(result_row)

print("语义相似度计算完成。")

# 5. 结果汇总与展示
results_df = pd.DataFrame(results)
if 'average_similarity' in results_df.columns:
    results_df.sort_values(by='average_similarity', ascending=False, na_position='last', inplace=True)

print("\n--- 单个疾病的语义一致性结果 (前10行) ---")
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)
print(results_df.head(10).to_string(index=True))
if len(results_df) > 10:
    print(f"... (以及另外 {len(results_df)-10} 行)")


# ------------------- 计算并准备添加总平均相似度到DataFrame -------------------
print("\n--- 各文件对之间的总平均语义相似度 ---")

num_files = len(FILE_PATHS)
pairwise_overall_averages = {} # 存储文件名和对应的平均相似度值
grand_average_similarity = np.nan # 初始化总总平均相似度

summary_rows_data = [] # 用于存储将要添加到DataFrame的摘要行

# 确保results_df不为空
if not results_df.empty:
    # 比较文件1和文件2
    if num_files >= 2 and 'similarity_1_vs_2' in results_df.columns:
        avg_1_vs_2 = results_df['similarity_1_vs_2'].mean()
        if pd.notna(avg_1_vs_2): # 确保平均值有效
            pairwise_overall_averages['avg_1_vs_2'] = avg_1_vs_2
            print(f"文件1 ({os.path.basename(FILE_PATHS[0])}) vs 文件2 ({os.path.basename(FILE_PATHS[1])}) 的总平均相似度: {avg_1_vs_2:.4f}")
            summary_rows_data.append({
                DISEASE_COLUMN: f'SUMMARY: Avg Sim Files 1&2 ({os.path.basename(FILE_PATHS[0])} vs {os.path.basename(FILE_PATHS[1])})',
                'average_similarity': avg_1_vs_2 # 将摘要值放入 'average_similarity' 列
                # 其他列将自动填充为 NaN
            })
        else:
            print(f"文件1 vs 文件2: 无法计算有效平均值 (可能所有比较都是NaN)。")
    else:
        print(f"文件1 vs 文件2: 无法计算 (可能文件不足或对应列 'similarity_1_vs_2' 不存在)。")

    # 比较文件1和文件3
    if num_files >= 3 and 'similarity_1_vs_3' in results_df.columns:
        avg_1_vs_3 = results_df['similarity_1_vs_3'].mean()
        if pd.notna(avg_1_vs_3):
            pairwise_overall_averages['avg_1_vs_3'] = avg_1_vs_3
            print(f"文件1 ({os.path.basename(FILE_PATHS[0])}) vs 文件3 ({os.path.basename(FILE_PATHS[2])}) 的总平均相似度: {avg_1_vs_3:.4f}")
            summary_rows_data.append({
                DISEASE_COLUMN: f'SUMMARY: Avg Sim Files 1&3 ({os.path.basename(FILE_PATHS[0])} vs {os.path.basename(FILE_PATHS[2])})',
                'average_similarity': avg_1_vs_3
            })
        else:
            print(f"文件1 vs 文件3: 无法计算有效平均值。")
    elif num_files >= 3:
        print(f"文件1 vs 文件3: 无法计算 (对应列 'similarity_1_vs_3' 不存在)。")


    # 比较文件2和文件3
    if num_files >= 3 and 'similarity_2_vs_3' in results_df.columns:
        avg_2_vs_3 = results_df['similarity_2_vs_3'].mean()
        if pd.notna(avg_2_vs_3):
            pairwise_overall_averages['avg_2_vs_3'] = avg_2_vs_3
            print(f"文件2 ({os.path.basename(FILE_PATHS[1])}) vs 文件3 ({os.path.basename(FILE_PATHS[2])}) 的总平均相似度: {avg_2_vs_3:.4f}")
            summary_rows_data.append({
                DISEASE_COLUMN: f'SUMMARY: Avg Sim Files 2&3 ({os.path.basename(FILE_PATHS[1])} vs {os.path.basename(FILE_PATHS[2])})',
                'average_similarity': avg_2_vs_3
            })
        else:
            print(f"文件2 vs 文件3: 无法计算有效平均值。")
    elif num_files >= 3:
        print(f"文件2 vs 文件3: 无法计算 (对应列 'similarity_2_vs_3' 不存在)。")
    
    # 计算所有有效的文件对平均相似度的总平均值
    all_pairwise_scores_values = [score for score in pairwise_overall_averages.values() if pd.notna(score)]
    if all_pairwise_scores_values:
        grand_average_similarity = np.mean(all_pairwise_scores_values)
        print(f"\n所有思维链结果对比的总平均相似度: {grand_average_similarity:.4f}")
        summary_rows_data.append({
            DISEASE_COLUMN: 'SUMMARY: Grand Average (All Calculated File Pairs)',
            'average_similarity': grand_average_similarity
        })
    else:
        print("\n未能计算任何文件对之间的总平均相似度。")

else:
    print("结果DataFrame为空，无法计算总平均相似度。")

# 将摘要行添加到 results_df
if summary_rows_data:
    summary_df = pd.DataFrame(summary_rows_data)
    # 确保 summary_df 的列与 results_df 对齐，缺少的列会是 NaN
    # 我们只关心 DISEASE_COLUMN 和 average_similarity
    for col in results_df.columns:
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    summary_df = summary_df[results_df.columns] # 保证列顺序一致
    
    results_df = pd.concat([results_df, summary_df], ignore_index=True)

# ------------------------------------------------------------------------------------

# 6. 保存结果到 CSV
try:
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n语义相似度结果已保存到: {output_csv_path}")
except Exception as e:
    print(f"\n保存结果到CSV时出错: {e}")
    import traceback
    traceback.print_exc()