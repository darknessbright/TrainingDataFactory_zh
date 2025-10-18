# 划分训练集和验证集
import json
import pandas as pd
from rich import print
from sklearn.model_selection import train_test_split


def data_maker():
    # 加载本地Parquet数据集
    parquet_file_path = './mnt/instruction_dataset.parquet'
    dataset = pd.read_parquet(parquet_file_path)
    # 打印数据集的列名以验证
    print("Dataset Columns:", dataset.columns)
    # 将数据集转换为字典列表
    json_data = dataset.to_dict(orient='list')
    # 打印第一行以验证
    # print("First Row:", {key: json_data[key][0] for key in json_data})
    # 初始化列表以存储Alpaca格式数据
    alpaca_data = []
    # 处理数据集中的每个条目
    for i in range(len(json_data['instruction'])):
        instruction = json_data['instruction'][i]
        input_text = json_data['input'][i]
        original_output = json_data['output'][i]
        # 创建Alpaca格式条目
        alpaca_entry = {
            "instruction": instruction,
            "input": input_text,
            "output": original_output
        }
        alpaca_data.append(alpaca_entry)
    # 将数据集拆分为训练集和测试集（80%训练，20%测试）
    train_data, test_data = train_test_split(alpaca_data, test_size=0.2, random_state=42)
    # 以Alpaca格式保存训练集
    with open("./generated_data/alpaca_train_dataset.json", "w") as file:
        json.dump(train_data, file, indent=4)
    # 以Alpaca格式保存测试集
    with open("./generated_data/alpaca_test_dataset.json", "w") as file:
        json.dump(test_data, file, indent=4)
    print("Alpaca格式训练数据集已保存为'alpaca_train_dataset.json'")
    print("Alpaca格式测试数据集已保存为'alpaca_test_dataset.json'")
    return 'alpaca_train_dataset.json', 'alpaca_test_dataset.json'


if __name__ == "__main__":
    data_maker()