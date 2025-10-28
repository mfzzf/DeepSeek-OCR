# DeepSeek-OCR Docker - 使用本地环境依赖
# 这个版本复制本地 py312 环境的包，避免版本冲突
# 使用 devel 版本以支持 flash-attn 编译

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖和 Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# 设置 Python 3.12 为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 升级 pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY DeepSeek-OCR-vllm/requirements_docker.txt /app/

# 先安装 PyTorch（flash-attn 和 flashinfer 编译需要）
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 安装编译所需的依赖
RUN pip install --no-cache-dir ninja packaging psutil

# 安装 flash-attn（需要从源码编译，耗时较长，约 5-10 分钟）
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 安装其他依赖（使用本地 py312 环境验证过的版本）
RUN pip install --no-cache-dir -r requirements_docker.txt

# 注意：FlashInfer 目前没有 torch 2.8 的预编译版本，暂时跳过安装
# 不影响功能，只会使用 PyTorch 原生的 top-p & top-k 采样实现（性能略低但完全可用）
# 等 FlashInfer 发布 torch 2.8 版本后可以添加：
# RUN pip install --no-cache-dir flashinfer -i https://flashinfer.ai/whl/cu124/torch2.8/

# 复制应用代码
COPY DeepSeek-OCR-vllm/ /app/

# 创建模型目录
RUN mkdir -p /app/models

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 设置环境变量
ENV HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    VLLM_USE_V1=1

# 启动命令
CMD ["python3", "openai_api_server.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

