import pandas as pd

def merge_treatment_methods(file1_path, file2_path, output_path):
    """
    合并两个 CSV 文件，根据 "疾病" 列匹配 "治疗方法"。

    Args:
        file1_path (str): 第一个 CSV 文件路径。
        file2_path (str): 第二个 CSV 文件路径。
        output_path (str): 输出 CSV 文件路径。
    """

    try:
        # 1. 读取 CSV 文件
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        # 检查列名是否存在
        if '疾病' not in df1.columns or '治疗方法' not in df1.columns:
          raise ValueError(f"第一个CSV文件（{file1_path}）缺少名为 '疾病' 或 '治疗方法' 的列")
        if '疾病' not in df2.columns or '治疗方法' not in df2.columns:
          raise ValueError(f"第二个CSV文件（{file2_path}）缺少名为 '疾病' 或 '治疗方法' 的列")

        # 2. 重命名第二个文件的列名
        df2_rename = df2.rename(columns={'治疗方法': '治疗方法2'})

        # 3. 合并两个 DataFrame，使用 '疾病' 列作为键
        merged_df = pd.merge(df1, df2_rename, on='疾病', how='left')


        # 4. 将合并后的 DataFrame 保存到新的 CSV 文件
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"文件合并成功！结果已保存至 {output_path}")

    except FileNotFoundError:
        print(f"错误：未找到文件")
    except ValueError as e:
        print(f"错误：{e}")
    except Exception as e:
         print(f"发生未知错误：{e}")

# 主程序
if __name__ == "__main__":
    file1_path = "sample_TCM.csv"  # 第一个 CSV 文件路径
    file2_path = "sample_WM.csv"  # 第二个 CSV 文件路径
    output_path = "sample_merged.csv"  # 输出 CSV 文件路径
    merge_treatment_methods(file1_path, file2_path, output_path)