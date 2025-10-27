#!/bin/bash

# DeepSeek-OCR OpenAI-Compatible API Server启动脚本

# 设置默认参数
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未检测到 nvidia-smi，请确保已安装 NVIDIA 驱动"
fi

echo "=========================================="
echo "DeepSeek-OCR API Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "=========================================="
echo ""

# 启动服务器
python openai_api_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS"
