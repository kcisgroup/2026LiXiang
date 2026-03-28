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
    './Output-CoT/CoT_output_TCM_3_1.csv',
    './Output-CoT/CoT_output_WM_3_1.csv',
]
output_csv_path = 'TCMvsWM_Semantic_Consistency_3_1.csv'

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
    processed_text = " ".join(seg_list)
    return processed_text

# ---------------------------------------- 主程序 ----------------------------------------

# 1. 加载数据
dataframes = {}
file_types = ['TCM', 'WM']  # 对应两个文件的标识

for i, file_path in enumerate(FILE_PATHS):
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 {file_path}")
        exit()
    try:
        df = pd.read_csv(file_path)
        if DISEASE_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
            print(f"错误: 文件 {file_path} 缺少必需的列 ('{DISEASE_COLUMN}' 或 '{TEXT_COLUMN}')")
            exit()
        
        # 添加原始行号列 (CSV文件行号从1开始，数据从第2行开始)
        origin_row_col_name = f'origin_row_{file_types[i]}'
        df[origin_row_col_name] = df.index + 2 
        
        # 修改文本列名为对应类型并清洗、分词
        text_col_name = f'text_{file_types[i]}'
        df.rename(columns={TEXT_COLUMN: text_col_name}, inplace=True)
        
        df[text_col_name] = df[text_col_name].apply(basic_text_clean)
        df[text_col_name] = df[text_col_name].apply(segment_text_jieba)
        
        df.set_index(DISEASE_COLUMN, inplace=True)
        # 保留原始行号和处理后的文本列
        dataframes[file_types[i]] = df[[origin_row_col_name, text_col_name]] 

    except Exception as e:
        print(f"加载、处理或分词文件 {file_path} 时出错: {e}")
        exit()

# 2. 合并数据
merged_df = pd.concat(list(dataframes.values()), axis=1, join='outer')

# 3. 加载模型
print(f"\n正在加载 Sentence Transformer 模型: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

# 4. 计算语义相似度
results = []
print("\n开始计算语义相似度...")
for disease, row in merged_df.iterrows():
    text_tcm = row.get(f'text_{file_types[0]}')
    text_wm = row.get(f'text_{file_types[1]}')
    
    origin_row_tcm = row.get(f'origin_row_{file_types[0]}')
    origin_row_wm = row.get(f'origin_row_{file_types[1]}')

    # 过滤无效文本
    valid_tcm = pd.notna(text_tcm) and str(text_tcm).strip()
    valid_wm = pd.notna(text_wm) and str(text_wm).strip()
    
    current_result = {
        DISEASE_COLUMN: disease,
        f'origin_row_{file_types[0]}': origin_row_tcm,
        f'origin_row_{file_types[1]}': origin_row_wm,
        'similarity_TCM_vs_WM': np.nan,
        'missing_data_sources': [],
        'error_message': None
    }

    if not valid_tcm:
        current_result['missing_data_sources'].append(file_types[0])
    if not valid_wm:
        current_result['missing_data_sources'].append(file_types[1])

    if not valid_tcm or not valid_wm:
        current_result['error_message'] = '文本缺失'
        results.append(current_result)
        continue

    try:
        # 生成嵌入向量
        embeddings = model.encode([str(text_tcm), str(text_wm)], show_progress_bar=False)
        
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        current_result['similarity_TCM_vs_WM'] = similarity
        results.append(current_result)

    except Exception as e:
        current_result['error_message'] = str(e)
        results.append(current_result)
        print(f"计算疾病 '{disease}' 的相似度时出错: {e}")

print("语义相似度计算完成。")

# 5. 结果处理
results_df = pd.DataFrame(results)

# 计算总平均相似度
average_sim = results_df['similarity_TCM_vs_WM'].mean() # mean() 会自动忽略 NaN

results_df.sort_values(by='similarity_TCM_vs_WM', ascending=False, inplace=True, na_position='last')

# 创建总览行
summary_row_data = {
    DISEASE_COLUMN: '总体平均相似度',
    f'origin_row_{file_types[0]}': None, # 或 ""
    f'origin_row_{file_types[1]}': None, # 或 ""
    'similarity_TCM_vs_WM': average_sim,
    'missing_data_sources': None, # 或 []
    'error_message': None
}
summary_df = pd.DataFrame([summary_row_data])

# 将总览行添加到排序后的 DataFrame 底部
final_results_df = pd.concat([results_df, summary_df], ignore_index=True)

# 调整列顺序，将原始行号放在前面
cols_order = [f'origin_row_{file_types[0]}', f'origin_row_{file_types[1]}', DISEASE_COLUMN, 
              'similarity_TCM_vs_WM', 'missing_data_sources', 'error_message']
final_results_df = final_results_df[cols_order]


# 6. 保存结果
try:
    final_results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已成功保存至: {output_csv_path}")
    print(f"总体平均语义相似度: {average_sim:.4f}")
except Exception as e:
    print(f"\n保存结果到 {output_csv_path} 时出错: {e}")