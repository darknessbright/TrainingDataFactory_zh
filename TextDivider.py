# 使用BERT对中文文本进行按语义切分
import torch
from transformers import BertTokenizer, BertModel
import re
import os
# 余弦相似度计算
from scipy.spatial.distance import cosine
# 导入 tqdm 库用于显示进度条
from tqdm import tqdm
import time


def get_sentence_embedding(sentence, model, tokenizer):
    """
    获取句子的嵌入表示
    参数:
        sentence (str): 输入句子
        model (BertModel): 预训练的BERT模型
        tokenizer (BertTokenizer): BERT分词器
    返回:
        numpy.ndarray: 句子的嵌入向量
    """
    # 使用分词器处理输入句子，并转换为模型输入格式
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # 使用模型获取输出，不计算梯度
    with torch.no_grad():
        outputs = model(**inputs)
    # 返回最后一层隐藏状态的平均值作为句子嵌入
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def split_text_by_semantic(text, max_length, similarity_threshold=0.5, model_path='./bert-base-chinese', progress_callback=None):
    """
    基于语义相似度对文本进行分块
    参数:
        text (str): 输入的长文本
        max_length (int): 每个文本块的最大长度（以BERT分词器的token为单位）
        similarity_threshold (float): 语义相似度阈值，默认为0.79
        progress_callback (function): 进度回调函数，用于报告进度
    返回:
        list: 分割后的文本块列表
    """
    # 加载BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    model = BertModel.from_pretrained('./bert-base-chinese')
    model.eval()  # 设置模型为评估模式

    # 按句子分割文本（使用常见的中文标点符号）
    sentences = re.split(r'(。|！|？|；|\n)', text)
    # 重新组合句子和标点
    sentences = [s + p for s, p in zip(sentences[::2], sentences[1::2]) if s]

    if not sentences:
        return []

    chunks = []
    current_chunk = sentences[0]
    # 获取当前chunk的嵌入表示
    current_embedding = get_sentence_embedding(current_chunk, model, tokenizer)

    total_sentences = len(sentences) - 1  # 第一句已作为初始chunk
    processed_sentences = 0

    # 设定对于过于短小的句子或标题，自动与后面的句子放一起
    last_sentence = ""

    # 使用 tqdm 包装迭代器，显示进度条
    for sentence in tqdm(sentences[1:], desc="文本切分进度", unit="句"):
        # 调用进度回调函数（如果提供）
        if progress_callback:
            progress_callback(processed_sentences, total_sentences, "divider")
        if last_sentence != "":
            sentence = last_sentence + sentence
            last_sentence = ""

        # 单一符号跳过，和超短句合并
        if len(sentence.replace(" ","").replace("\n","")) <= 2:      # 如果出现切分出来的单一符号，则跳过
            continue
        elif len(sentence.replace(" ","").replace("\n","")) <= 15:
            last_sentence = sentence
            continue

        # 获取当前句子的嵌入表示
        sentence_embedding = get_sentence_embedding(sentence, model, tokenizer)
        # 计算当前chunk和当前句子的余弦相似度
        similarity = 1 - cosine(current_embedding, sentence_embedding)

        # 如果相似度高于阈值且合并后不超过最大长度，则合并
        if similarity > similarity_threshold and len(tokenizer.tokenize(current_chunk + sentence)) <= max_length:
            current_chunk += sentence
            # 更新当前chunk的嵌入表示 (简单平均)
            current_embedding = (current_embedding + sentence_embedding) / 2
        else:
            # 否则，保存当前chunk并开始新的chunk
            chunks.append(current_chunk)
            current_chunk = sentence
            current_embedding = sentence_embedding

        processed_sentences += 1

    # 添加最后一个chunk
    if current_chunk:
        chunks.append(current_chunk)

    # 循环遍历分段，对于过于短小的分段，与后续分段一起合并
    _last_chunk = ""
    chunks2 = []
    for _chunk in chunks:
        if _last_chunk != "":
            _chunk = _last_chunk + _chunk
            _last_chunk = ""
        if len(_chunk) <= 20:
            _last_chunk = _chunk
            continue
        else:
            chunks2.append(_chunk)
    chunks = chunks2

    # 完成时报告100%进度
    if progress_callback:
        progress_callback(total_sentences, total_sentences, "divider")

    return chunks


def read_text_file(file_path):
    """
    读取文本文件
    参数:
        file_path (str): 文件路径
    返回:
        str: 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def save_chunks_to_files(chunks, output_dir):
    """
    将分割后的文本块保存到文件
    参数:
        chunks (list): 文本块列表
        output_dir (str): 输出目录路径
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将每个文本块保存为单独的文件
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(output_dir, f"chunk_{i + 1}.txt")
        with open(chunk_file_path, 'w', encoding='utf-8') as file:
            file.write(chunk)
        # print(f"已保存第 {i + 1} 个文本块到 {chunk_file_path}") # 避免在进度条运行时打印过多信息


# 主程序
if __name__ == "__main__":
    # 设置输入和输出路径
    input_file_path = './test_novel.txt'  # 替换为你的长文本文件路径
    output_dir = './chunked_data/'  # 替换为你希望保存文本块的目录路径

    # 读取长文本
    long_text = read_text_file(input_file_path)

    # 设置每个文本块的最大分词数量和相似度阈值
    max_length = 2048                   # 可根据需要调整
    similarity_threshold = 0.5          # 可根据需要调整

    print("开始文本切分...")
    # 分割长文本
    text_chunks = split_text_by_semantic(long_text, max_length, similarity_threshold)
    print(f"文本切分完成，共生成 {len(text_chunks)} 个文本块。")

    # 保存分割后的文本块到指定目录
    save_chunks_to_files(text_chunks, output_dir)
    print(f"所有文本块已保存到 {output_dir}")
