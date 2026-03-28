import pandas as pd

# 加载原始CSV文件
input_file = "raw_TM.csv"  # 替换为你的文件名
output_file = "train_TM.csv"  # 输出文件名

# 读取CSV文件
df = pd.read_csv(input_file)

# 创建一个空的列表来存储新的数据行
new_rows = []

# 为每一行添加三种不同的提问方式，并拼接问题和答案
for index, row in df.iterrows():
    question_1 = f"<s>Human: {row['instruct']}的中医治疗方法有哪些？</s>"
    question_2 = f"<s>Human: 请给出{row['instruct']}的中医治疗方法。</s>"
    question_3 = f"<s>Human: 如果我得了{row['instruct']}，在中医中如何治疗？</s>"
    answer = f"<s>Assistant: {row['output']}</s>"

    # 拼接问题和答案，并添加到新行列表中
    new_rows.append({"text": question_1 + answer})
    new_rows.append({"text": question_2 + answer})
    new_rows.append({"text": question_3 + answer})

# 将新行列表转换为DataFrame
expanded_df = pd.DataFrame(new_rows)

# 保存到新的CSV文件
expanded_df.to_csv(output_file, index=False)

print(f"转换完成，新文件已保存为：{output_file}")