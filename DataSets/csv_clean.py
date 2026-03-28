import csv

input_file = 'raw_WM.csv'
output_file = 'raw_WM2.csv'

with open(input_file, newline='', encoding='utf-8') as infile, \
        open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # 清理字段内的换行符
        cleaned_row = [field.replace('\n', '').replace('\r', '').replace('*', '') for field in row]
        # 如果清理后的行不为空，则写入输出文件
        if any(field.strip() for field in cleaned_row):
            writer.writerow(cleaned_row)