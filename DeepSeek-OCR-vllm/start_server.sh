#!/bin/bash
set -e

# DeepSeek-OCR Docker 容器启动脚本

echo "=========================================="
echo "Starting DeepSeek-OCR API Server"
echo "=========================================="
echo "Host: ${HOST:-0.0.0.0}"
echo "Port: ${PORT:-8000}"
echo "Log Level: ${LOG_LEVEL:-INFO}"
echo "vLLM Version: V1"
echo "=========================================="

# 执行 API 服务器
python3 openai_api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1

