import csv

# 打开CSV文件
with open('.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# 添加表头
header = ['', '']  # 自定义表头
rows.insert(0, header)

# 保存文件
with open('.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)