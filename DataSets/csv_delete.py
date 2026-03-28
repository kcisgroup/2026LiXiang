import pandas as pd

# 读取CSV文件
df = pd.read_csv('raw_WM.csv')

# 定位并删除包含特定值的行
df = df[df['output'] != '生成失败']

# 保存修改后的CSV文件
df.to_csv('raw_WM.csv', index=False)