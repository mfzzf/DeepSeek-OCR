#!/bin/bash

# DeepSeek-OCR Docker 构建脚本

set -e

echo "=========================================="
echo "DeepSeek-OCR Docker 构建"
echo "=========================================="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "错误: 未找到 Docker，请先安装 Docker"
    exit 1
fi

# 检查 NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "警告: NVIDIA Docker 可能未正确配置"
    echo "请确保已安装 nvidia-docker2 或 nvidia-container-toolkit"
fi

# 镜像名称和标签
IMAGE_NAME="deepseek-ocr-vllm"
IMAGE_TAG="${1:-latest}"

echo ""
echo "构建镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "基础镜像: vllm/vllm-openai:v0.6.3.post1 (CUDA 12.1/12.2)"
echo ""

# 构建镜像
docker build \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --progress=plain \
    .

echo ""
echo "=========================================="
echo "构建完成!"
echo "=========================================="
echo "镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "运行命令示例:"
echo ""
echo "  # 使用 Docker Compose (推荐)"
echo "  docker-compose up -d"
echo ""
echo "  # 使用 Docker 命令"
echo "  docker run -d \\"
echo "    --name deepseek-ocr-api \\"
echo "    --gpus all \\"
echo "    --shm-size 8g \\"
echo "    -p 8000:8000 \\"
echo "    -v \$(pwd)/DeepSeek-OCR-vllm/models:/app/models:ro \\"
echo "    -e LOG_LEVEL=INFO \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "=========================================="

