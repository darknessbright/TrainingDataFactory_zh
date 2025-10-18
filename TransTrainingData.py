# AI输出数据转化为标准化训练集
import json
import pandas as pd
from rich import print


# 加载本地Parquet数据集
parquet_file_path = './mnt/instruction_dataset.parquet'
dataset = pd.read_parquet(parquet_file_path)

# 打印数据集的列名以验证
print("Dataset Columns:", dataset.columns)

# 将数据集转换为字典列表
json_data = dataset.to_dict(orient='list')

# 打印第一行以验证
print("First Row:", {key: json_data[key][0] for key in json_data})

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

# 以Alpaca格式保存
with open("./generated_data/alpaca_dataset.json", "w") as file:
    json.dump(alpaca_data, file, indent=4)

print("Alpaca format dataset saved as 'alpaca_dataset.json'")
