# 训练样本生成GUI
import flask
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
import webbrowser
import os
import shutil
import TextDivider
import AIWorker
import Training_Test_Maker
import threading
import time
import subprocess

# 获取脚本的绝对路径
my_path = os.path.dirname(os.path.realpath(__file__))

# 初始化 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated_data'
CHUNKED_FOLDER = 'chunked_data'  # 用于存放切分后的数据
shutil.rmtree('chunked_data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['CHUNKED_FOLDER'] = CHUNKED_FOLDER  # 配置切分数据文件夹
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB 最大上传大小

# 确保上传和生成文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs(CHUNKED_FOLDER, exist_ok=True)  # 创建切分数据文件夹

# 全局变量用于存储当前处理状态
current_processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': ''
}

# 全局变量用于存储ollama信息
ollama_info = {
    'available': False,
    'models': []
}

def logo():
    print("="*66)
    print("")
    print("    ==========     =====         ====##         ##=======     #===")
    print("       ##         #    ##       #     ##       ##           ##")
    print("      ##         #===##        #       ##     ##           ##")
    print("     ##         ###           #       ##     ##=======    ##")
    print("    ##         #   ##        #      ##      ##            ##")
    print("   ##         #      #==    #====##        ##              #===")
    print("")
    print("="*66)
    print("\n" + "-" * 66)
    print("Engineer：北风小漠（DarknessBright）  Email:darknessbright@126.com")
    print("-" * 66 + "\n")
    time.sleep(1)


def check_ollama_status():
    """检查ollama服务状态和可用模型"""
    global ollama_info
    try:
        # 检查ollama是否可用
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            ollama_info['available'] = True
            # 解析模型列表
            lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # 第一列是模型名
                    models.append(model_name)
            ollama_info['models'] = models
        else:
            ollama_info['available'] = False
            ollama_info['models'] = []
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        ollama_info['available'] = False
        ollama_info['models'] = []


# 首页路由
@app.route('/')
def index():
    """
    渲染训练数据工厂的主页面。
    """
    return render_template('index.html')


# 处理文件上传的路由
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    处理单个或多个文件上传，并进行文本切分。
    """
    global current_processing_status

    # 重置处理状态
    current_processing_status['is_processing'] = True
    current_processing_status['progress'] = 0
    current_processing_status['message'] = '开始处理文件...'

    uploaded_files_paths = []
    if 'file' in request.files:  # 单文件上传
        file = request.files['file']
        if file.filename != '':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            uploaded_files_paths.append(filename)

    if 'files[]' in request.files:  # 多文件上传
        files = request.files.getlist('files[]')
        for file in files:
            if file and file.filename != '':
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                uploaded_files_paths.append(filename)

    if not uploaded_files_paths:
        current_processing_status['is_processing'] = False
        return render_template('index.html', message='请至少上传一个文件！')

    # 获取样本切分配置
    try:
        max_length = int(request.form.get('max_length', 2048))
        similarity_threshold = float(request.form.get('similarity_threshold', 0.5))
    except ValueError:
        current_processing_status['is_processing'] = False
        return render_template('index.html', message='切分参数无效，请检查最大长度和相似度阈值！')

    # 获取大模型和Prompt配置
    model_choice = request.form.get('model_choice')         # 大模型选择
    prompt_text = request.form.get('prompt_text')           # 手动设置的提示词
    top_k = request.form.get('top_k')
    top_p = request.form.get('top_p')
    temperature = request.form.get('temperature')
    context_window = request.form.get('context_window')
    # print(model_choice,prompt_text,top_k,top_p,temperature, context_window)

    print(f"Uploaded files: {[os.path.basename(f) for f in uploaded_files_paths]}")
    print(f"Max Length for chunking: {max_length}")
    print(f"Similarity Threshold for chunking: {similarity_threshold}")
    print(f"Selected model: {model_choice}")
    print(f"Prompt: {prompt_text}")

    # 在后台线程中处理文件
    thread = threading.Thread(
        target=process_files_background,
        args=(uploaded_files_paths, max_length, similarity_threshold, model_choice, prompt_text, top_k, top_p, temperature)
    )
    thread.start()

    return render_template('index.html', message='文件上传成功，正在处理中...')


# 句子切分进度条
def progress_callback(processed, total, process_type):
    """进度回调函数"""
    global current_processing_status
    if total > 0:
        progress = int((processed / total) * 100)
        current_processing_status['progress'] = progress
        if process_type == "divider":
            current_processing_status['message'] = f'正在切分文本... {progress}%'
            # 通过SocketIO发送进度更新
            socketio.emit('progress_update', {
                'progress': progress,
                'message': f'正在切分文本... {progress}%'
            })
        elif process_type == "AI":
            current_processing_status['message'] = f'正在生成训练数据... {progress}%'
            # 通过SocketIO发送进度更新
            socketio.emit('progress_update', {
                'progress': progress,
                'message': f'正在生成训练数据... {progress}%'
            })
        time.sleep(0.01)  # 短暂延迟以确保消息发送


def process_files_background(uploaded_files_paths, max_length, similarity_threshold, model_choice, prompt_text, top_k, top_p, temperature):
    """在后台线程中处理文件"""
    global current_processing_status
    try:
        processed_chunk_files = []
        all_chunks = []
        for uploaded_file_path in uploaded_files_paths:
            # 更新状态
            filename = os.path.basename(uploaded_file_path)
            current_processing_status['message'] = f'正在处理文件: {filename}'
            socketio.emit('progress_update', {
                'progress': current_processing_status['progress'],
                'message': f'正在处理文件: {filename}'
            })

            # 读取文件内容
            long_text = TextDivider.read_text_file(uploaded_file_path)
            # 调用 TextDivider 进行文本切分，传入进度回调函数
            text_chunks = TextDivider.split_text_by_semantic(
                long_text,
                max_length,
                similarity_threshold,
                my_path+"/bert-base-chinese",
                progress_callback
            )

            # 为每个上传文件创建一个独立的切分输出目录
            original_filename_base = os.path.splitext(os.path.basename(uploaded_file_path))[0]
            output_chunk_dir = os.path.join(app.config['CHUNKED_FOLDER'], original_filename_base)
            os.makedirs(output_chunk_dir, exist_ok=True)

            # 保存切分后的文本块
            all_chunks = all_chunks + text_chunks                               # 在总变量中记录切分数据
            TextDivider.save_chunks_to_files(text_chunks, output_chunk_dir)     # 存储切分数据
            processed_chunk_files.append(output_chunk_dir)                      # 记录切分后的目录

        socketio.emit('processing_complete', {
            'message': '文件上传、切分成功并已开始生成数据！'
        })

        # 调用AIWorker进行训练数据生成
        input_folder = "./chunked_data"        # 指定输入文件夹路径
        output_file = "mnt/instruction_dataset.parquet"
        print("开始生成数据集...")
        dataset = AIWorker.generate_dataset(input_folder, entries_per_file=5, progress_callback=progress_callback)
        AIWorker.save_dataset_as_parquet(dataset, output_file)                  # 存储Q&A训练数据

        # 调用 Training_Test_Maker 生成最终的数据集（包含训练数据和测试数据）
        train_file, test_file = Training_Test_Maker.data_maker()

        # 处理完成
        current_processing_status['is_processing'] = False
        current_processing_status['progress'] = 100
        current_processing_status['message'] = '处理完成！'

        socketio.emit('processing_complete', {
            'message': '训练数据集、测试数据集生成完毕！',
            'train_file': os.path.basename(train_file),
            'test_file': os.path.basename(test_file)
        })

    except Exception as e:
        current_processing_status['is_processing'] = False
        error_msg = f'处理文件时发生错误: {str(e)}'
        current_processing_status['message'] = error_msg
        socketio.emit('processing_error', {'message': error_msg})


# 获取当前处理状态的路由
@app.route('/status')
def get_status():
    """获取当前处理状态"""
    global current_processing_status
    return flask.jsonify(current_processing_status)

# 获取ollama信息的路由
@app.route('/ollama_models')
def get_ollama_info():
    """
    返回ollama服务信息
    """
    # print(ollama_info)
    return flask.jsonify(ollama_info["models"])

# 下载生成文件的路由
@app.route('/download/<filename>')
def download_file(filename):
    """
    允许用户下载生成的文件。
    """
    return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)


# SocketIO事件处理
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')


@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开连接')
    # 当客户端连接时，发送当前的ollama信息
    emit('ollama_info_update', ollama_info["models"])


if __name__ == "__main__":
    logo()
    # 检查ollama状态
    check_ollama_status()
    print(f"Ollama available: {ollama_info['available']}")
    if ollama_info['models']:
        print(f"Available models: {', '.join(ollama_info['models'])}")
    # 在浏览器中打开指定的 URL
    webbrowser.open_new('http://localhost:5000')
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)
