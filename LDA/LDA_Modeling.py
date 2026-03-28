import pandas as pd
import jieba
import re
from gensim import corpora, models
import pyLDAvis.gensim_models
import pyLDAvis
import warnings
import random

# 忽略一些可能出现的警告信息
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 配置参数 ---
CSV_FILE_PATH = './Output_FT/FT_Output_TCM.csv'              # CSV文件名
TEXT_COLUMN = '治疗方法'                                     # 需要分析的文本列名
STOPWORDS_FILE_PATH = 'stopwords.txt'                       # 停用词文件路径
NUM_TOPICS = 5                                              # 提取的主题数量
NUM_WORDS_PER_TOPIC = 15                                    # 每个主题显示多少个关键词
OUTPUT_VIS_HTML = './LDA/lda_result_tcm.html'                         # 可视化结果保存的文件名

# --- 功能函数 ---

def load_stopwords(filepath):
    """从文件加载停用词，每行一个词"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 使用集合提高查找效率
            stopwords = {line.strip() for line in f if line.strip()}
        print(f"成功从 {filepath} 加载 {len(stopwords)} 个停用词。")
        return stopwords
    except FileNotFoundError:
        print(f"警告: 停用词文件 {filepath} 未找到。将使用空列表。")
        return set()

def preprocess_text(text, stopwords):
    """清洗、分词、去停用词"""
    # 1. 确保是字符串类型
    text = str(text)

    # 2. 清洗文本: 移除特定格式、Markdown、多余空格、网址、英文、数字等
    text = re.sub(r'###\s*请提供详细答案', '', text) # 移除特定头部
    text = re.sub(r'###\s*添加评论[\s\S]*?提供"', '', text, flags=re.MULTILINE) # 移除评论区
    text = re.sub(r'###\s*问题[\s\S]*?###', '', text, flags=re.MULTILINE) # 移除 ### 问题 ### 块
    text = re.sub(r'#\s*[\w\s]+', '', text) # 移除话题标签
    text = re.sub(r'https?://\S+', '', text) # 移除URL
    text = re.sub(r'[^\u4e00-\u9fa5\s]', '', text) # 仅保留中文字符和空格
    text = re.sub(r'\s+', ' ', text).strip() # 移除多余空格
    # text = re.sub(r'[a-zA-Z0-9]', '', text) # (上一行已包含此功能)

    # 3. 使用jieba进行分词
    words = jieba.lcut(text)

    # 4. 移除停用词和长度为1的词
    words = [word for word in words if word not in stopwords and len(word) > 1]
    return words

# --- 主程序 ---

# 1. 加载数据
print(f"开始加载数据: {CSV_FILE_PATH}")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"成功加载 {CSV_FILE_PATH}")
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"错误: 在CSV文件中未找到列名 '{TEXT_COLUMN}'")
    # 删除目标列中包含NaN值的行
    df.dropna(subset=[TEXT_COLUMN], inplace=True)
    texts_raw = df[TEXT_COLUMN].tolist()
    print(f"需要处理的文档数量: {len(texts_raw)}")
except FileNotFoundError:
    print(f"错误: CSV 文件 {CSV_FILE_PATH} 未找到。")
    exit()
except ValueError as e:
    print(f"错误: {e}")
    exit()
except Exception as e:
    print(f"加载数据时发生意外错误: {e}")
    exit()


# 2. 加载停用词
stopwords = load_stopwords(STOPWORDS_FILE_PATH)

# 3. 预处理文本数据
print("开始预处理文本...")
processed_texts = [preprocess_text(text, stopwords) for text in texts_raw]
# 移除预处理后可能产生的空列表
processed_texts = [text for text in processed_texts if text]
print(f"预处理完成。有效文档数量: {len(processed_texts)}")

if not processed_texts:
    print("错误: 预处理后没有有效的文档。请检查停用词表或输入数据。")
    exit()

# 4. 创建 Gensim 词典和语料库
print("开始创建词典和语料库...")
dictionary = corpora.Dictionary(processed_texts)
# 可选: 过滤掉词频过低或过高的词
# dictionary.filter_extremes(no_below=2, no_above=0.6) # 例如：去掉只出现1次的词，去掉在60%以上文档中都出现的词
corpus = [dictionary.doc2bow(text) for text in processed_texts]
print("词典和语料库创建完成。")

# 检查语料库是否为空
if not corpus or all(not doc for doc in corpus):
     print("错误: 创建词袋语料库后为空。请检查预处理步骤和过滤条件。")
     exit()


# 5. 训练 LDA 模型
print(f"开始训练 LDA 模型 (主题数={NUM_TOPICS})...")
# 设置随机种子以确保结果可复现
random.seed(42)
jieba.re_han_default = re.compile('([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)', re.U) # 兼容jieba和pyLDAvis

try:
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=100,        # 保证结果可复现
        update_every=1,          # 每处理完一个 chunk 更新一次模型
        chunksize=10,            # 每次处理的文档数，根据内存调整
        passes=20,               # 对整个语料库的训练迭代次数
        alpha='auto',            # 文档-主题分布的先验参数
        eta='auto',              # 主题-词语分布的先验参数 (也可设为固定值或'auto')
        per_word_topics=True     # 计算每个词属于哪个主题，为pyLDAvis准备
    )
    print("LDA 模型训练完成。")
except Exception as e:
    print(f"LDA 训练过程中发生错误: {e}")
    exit()


# 6. 打印主题结果
print(f"\n--- 每个主题下的前 {NUM_WORDS_PER_TOPIC} 个关键词 ---")
topics = lda_model.print_topics(num_words=NUM_WORDS_PER_TOPIC)
for i, topic in enumerate(topics):
    # topic[1] 是主题词的字符串表示
    print(f"主题 {i}: {topic[1]}")

# 7. (可选) 计算一致性得分 (Coherence Score)
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'\n模型一致性得分 (C_V): {coherence_lda:.4f}') # 分数越高通常表示主题质量越好

# 8. (可选) 准备并保存可视化结果
print(f"\n正在准备可视化结果 (保存至 {OUTPUT_VIS_HTML})...")
try:
    # 准备 pyLDAvis 数据
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, mds='mmds') # 使用 'mmds' 可能获得更好的降维效果
    # 保存为 HTML 文件
    pyLDAvis.save_html(vis_data, OUTPUT_VIS_HTML)
    print(f"可视化结果已保存至 {OUTPUT_VIS_HTML}。请在浏览器中打开查看。")
except Exception as e:
    print(f"可视化准备过程中发生错误: {e}")
    print("跳过可视化步骤。")

print("\n--- 脚本执行完毕 ---")