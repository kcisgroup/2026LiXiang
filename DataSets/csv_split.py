import pandas as pd

# 文件路径
input_file = "train_WM.csv"  # 原始文件名
sample_file = "val_WM.csv"  # 抽样后的文件

# 读取CSV文件
df = pd.read_csv(input_file)

# 检查原始数据是否足够抽取1000条
if len(df) < 3000:
    raise ValueError("数据不足，无法完成抽样！")

# 随机抽取1000条数据
sampled_df = df.sample(n=3000, random_state=42)  # random_state 确保可复现
remaining_df = df.drop(sampled_df.index)  # 删除抽取的数据

# 保存抽取数据到文件
sampled_df.to_csv(sample_file, index=False)

# 覆盖原始文件，仅保留剩余数据
remaining_df.to_csv(input_file, index=False)

print(f"已随机抽取数据到文件：{sample_file}")
print(f"原文件已更新，仅保留剩余数据。")
