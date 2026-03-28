import random
from http import HTTPStatus
import dashscope
import csv

# 配置DashScope API密钥（推荐从环境变量中读取）
dashscope.api_key = 'sk-211256a3ca4b4781b49139f509de7367'  # 请替换为你的API密钥


# 生成中医和西医治疗方法的函数
def get_treatment_advice(symptom, treatment_type):
    prompt_templates = [ f"请为{symptom}提供西医的推荐治疗方案。", f"如果一个患者得了{symptom}，西医的治疗方案是什么？", f"{symptom}的常见西医治疗方法有哪些？", ]
    prompt = random.choice(prompt_templates)
    messages = [{'role': 'user', 'content': prompt}]
    response = dashscope.Generation.call(
        'qwen2-7b-instruct',
        messages=messages,
        seed=random.randint(1, 10000),  # 随机种子，保持生成结果的多样性
        result_format='message',
        stream=False,
        output_in_full=True
    )

    if response.status_code == HTTPStatus.OK:
        full_content = response.output.choices[0]['message']['content']
    else:
        print(f"Error: {response.status_code}, {response.message}")
        full_content = "生成失败"
    return full_content


# 症状列表
symptoms = {}
# 生成并保存数据集
def generate_treatment_dataset():
    treatments = []

    for symptom in symptoms:
        # 获取西医治疗方法
        treatment_western = get_treatment_advice(symptom, '西医')
        print(f"西医治疗方法：{treatment_western}")

        # 将结果添加到列表中
        treatments.append([symptom, treatment_western])

    # 保存结果为CSV文件
    with open('../Datasets/treatments-WM-plus.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(treatments)

    print("数据集已保存。")


if __name__ == '__main__':
    generate_treatment_dataset()
