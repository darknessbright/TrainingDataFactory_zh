# 利用AI批量生成训练集
import json
import os
import random
import time
import re
from typing import List, Dict
import ollama
# from openai import OpenAI
import logging
import backoff
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置全局任务计数器
class Public:
    all_tasks = 0
    now_tasks = 0


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从环境变量中获取 API 密钥
api_key = "？？？"

# 初始化 ollama
class OllamaMultiTurn:
    def __init__(self, model="qwen3:30b-a3b"):
        self.model = model
        self.chat_history = []

    def get_models(self):
        """获取Ollama支持的模型列表"""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.error(f"获取模型列表时出错: {str(e)}\n请检查本地ollama服务！")
            # 返回默认模型列表
            return []

    def send_message(self, message, temperature=0.7, num_ctx=4096, top_k=50, top_p=0.7):
        """发送消息并获取响应"""
        # 添加用户消息到历史记录
        self.chat_history.append({"role": "user", "content": message})

        # 调用Ollama API
        response = ollama.chat(
            model=self.model,
            messages=self.chat_history,
            stream=True,
            options=ollama.Options(
                temperature=0.7,    # 适度的创造性
                num_ctx=4096,       # 中等大小的上下文窗口
                top_k=50,
                top_p=0.7,
            ),
        )

        # 处理流式响应
        full_response = ""
        for chunk in response:
            content = chunk["message"]["content"]
            full_response += content
            print(f"{content}", end="", flush=True)
        print("\n" + "-"*50)

        # 添加模型回复到历史记录
        self.chat_history.append({"role": "assistant", "content": full_response})
        return full_response

    def reset_conversation(self):
        """重置对话历史记录"""
        self.chat_history = []
        print("✅ 对话已重置")

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_single_entry(text: str) -> Dict:
    prompt = f"""
    基于以下文本，生成1个用于指令数据集的高质量条目。条目应该直接关联到给定的文本内容，提出相关的问题或任务。
    请确保生成多样化的指令类型，例如：
    - 分析类："分析..."
    - 比较类："比较..."
    - 解释类："解释..."
    - 评价类："评价..."
    - 问答类："为什么..."

    文本内容：
    {text}

    请以下面的格式生成条目，确保所有字段都有适当的内容：
    {{
        "instruction": "使用上述多样化的指令类型之一，提出一个具体的、与文本相关的问题或任务",
        "input": "如果需要额外的上下文信息，请在这里提供，否则留空",
        "output": "对instruction的详细回答或任务的完成结果"
    }}
    确保所有生成的内容都与给定的文本直接相关，生成的是有效的JSON格式，并且内容高质量、准确、详细。
    """

    # 实例化ollama会话
    Chat = OllamaMultiTurn()
    try:
        response = Chat.send_message(prompt)
        # 使用ollama
        logger.info(f"API 响应: {response}")
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        # 使用OpenAI
        # logger.info(f"API 响应: {response.choices[0].message.content}")
        # json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if json_match:
            entry = json.loads(json_match.group())
            required_keys = ['instruction', 'input', 'output']
            if isinstance(entry, dict) and all(key in entry for key in required_keys):
                # 根据 input 是否为空来设置 text 字段
                if entry['input'].strip():
                    entry['text'] = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction: {entry['instruction']}\n### Input: {entry['input']}\n### Response: {entry['output']}"
                else:
                    entry['text'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.### Instruction: {entry['instruction']}\n### Input: {entry['input']}\n### Response: {entry['output']}"

                logger.info("成功生成完整条目")
                return entry
            else:
                logger.warning("JSON 解析成功，但缺少必要字段")
                return {}
        else:
            logger.error("无法从API响应中提取有效的JSON")
            return {}

    except Exception as e:
        logger.error(f"生成条目时发生错误: {str(e)}")
        return {}

def process_file(file_path: str, entries_per_file: int, progress_callback=None) -> List[Dict]:
    dataset = []

    try:
        text = read_file(file_path)
        for j in range(entries_per_file):
            logger.info(f"  生成第 {j + 1}/{entries_per_file} 个条目")
            # 报告进度
            if progress_callback:
                progress_callback(Public.now_tasks, Public.all_tasks * entries_per_file, "AI")
            entry = generate_single_entry(text)
            if entry and all(key in entry for key in ['instruction', 'input', 'output', 'text']):
                dataset.append(entry)
                logger.info(f"  成功生成 1 个完整条目")
            else:
                logger.warning(f"  跳过不完整的条目")
            Public.now_tasks += 1
            time.sleep(1)  # 在请求之间增加延迟到2秒
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生未知异常: {str(e)}")
    return dataset

def generate_dataset(folder_path: str, entries_per_file: int = 2, progress_callback=None) -> List[Dict]:
    dataset = []
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        # 获取当前文件夹中的所有文件
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    files = [filename for filename in all_files if filename.endswith(".txt")]
    Public.all_tasks = len(files)
    with ThreadPoolExecutor(max_workers=4) as executor:  # 调整 max_workers 数量以适应你的硬件资源
        futures = [executor.submit(process_file, file_path, entries_per_file, progress_callback) for file_path in files]
        for future in as_completed(futures):
            try:
                dataset.extend(future.result())
            except Exception as e:
                logger.error(f"处理未来任务时发生未知异常: {str(e)}")
    Public.all_tasks = 0
    Public.now_tasks = 0
    return dataset

def save_dataset_as_parquet(dataset: List[Dict], output_file: str):
    schema = pa.schema([
        ('instruction', pa.string()),
        ('input', pa.string()),
        ('output', pa.string()),
        ('text', pa.string())
    ])

    arrays = [
        pa.array([entry['instruction'] for entry in dataset]),
        pa.array([entry['input'] for entry in dataset]),
        pa.array([entry['output'] for entry in dataset]),
        pa.array([entry['text'] for entry in dataset])
    ]
    '''
    # 添加元数据信息
    metadata = {
        'dataset_description': 'Instruction dataset generated by AI',
        'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_entries': str(len(dataset)),
        'generator_version': '1.0'
    }
    '''
    table = pa.Table.from_arrays(arrays=arrays, schema=schema)
    pq.write_table(table, output_file)

def get_ollama_models():
    """获取Ollama模型列表的便捷函数"""
    ollama_client = OllamaMultiTurn()
    return ollama_client.get_models()

if __name__ == "__main__":
    input_folder = "./chunked_data"  # 指定输入文件夹路径
    output_file = "mnt/instruction_dataset.parquet"

    logger.info("开始生成数据集")
    dataset = generate_dataset(input_folder, entries_per_file=5)
    save_dataset_as_parquet(dataset, output_file)
    logger.info(f"数据集已生成并保存到 {output_file}")
    logger.info(f"共生成 {len(dataset)} 个有效条目")
