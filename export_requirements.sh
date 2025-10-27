#!/bin/bash

# 导出当前 py312 环境的依赖到 Docker requirements

set -e

echo "=========================================="
echo "导出 py312 环境依赖"
echo "=========================================="

# 激活 py312 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py312

# 导出所有依赖
echo "导出完整依赖列表..."
pip freeze > /home/ubuntu/projects/DeepSeek-OCR/DeepSeek-OCR-vllm/requirements_full.txt

# 只导出核心依赖（过滤掉一些不必要的）
echo "生成精简依赖列表..."
pip freeze | grep -E "^(torch|vllm|transformers|tokenizers|fastapi|uvicorn|pydantic|Pillow|requests|PyMuPDF|img2pdf|einops|easydict|addict|numpy)" > /home/ubuntu/projects/DeepSeek-OCR/DeepSeek-OCR-vllm/requirements_docker.txt

echo ""
echo "=========================================="
echo "导出完成！"
echo "=========================================="
echo "完整依赖: DeepSeek-OCR-vllm/requirements_full.txt"
echo "精简依赖: DeepSeek-OCR-vllm/requirements_docker.txt"
echo ""
echo "请检查生成的文件，然后更新 Dockerfile"
echo "=========================================="

