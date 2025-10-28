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

# 设置 Tensor Parallel Size 环境变量
# 如果未设置，则自动检测 GPU 数量
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    echo "TENSOR_PARALLEL_SIZE not set, will auto-detect GPU count"
    echo "Tensor Parallel Size: auto"
else
    export TENSOR_PARALLEL_SIZE
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
fi

echo "=========================================="

# NCCL 配置：用于多 GPU 通信
# 根据具体环境调整 NCCL 参数
if [ -n "$TENSOR_PARALLEL_SIZE" ] && [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    echo "Multi-GPU mode: Configuring NCCL..."
    # export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1        # 禁用 InfiniBand（如果没有 IB 网络）
    export NCCL_P2P_DISABLE=0        # 启用 P2P 通信（GPU 间直接通信）
    export NCCL_SHM_DISABLE=0        # 启用共享内存传输
    # 如果遇到 NCCL 错误，可尝试：
    # export NCCL_P2P_LEVEL=NVL      # 使用 NVLink（如果可用）
    # export NCCL_SHM_DISABLE=1      # 禁用共享内存（作为备选方案）
else
    echo "Single GPU mode or auto-detect: NCCL will be configured automatically if needed"
fi

# 执行 API 服务器
python3 openai_api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1

