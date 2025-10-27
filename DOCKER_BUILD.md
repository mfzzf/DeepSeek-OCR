# DeepSeek-OCR Docker 构建和使用指南

## 📋 前提条件

- Docker 已安装
- NVIDIA Docker runtime 已配置
- CUDA 12.1/12.2 驱动
- GPU 显存 >= 16GB (推荐 24GB+)

## 🚀 快速开始

### 方法 1: 使用 Docker Compose (推荐)

```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

### 方法 2: 使用 Docker 命令

```bash
# 1. 构建镜像
docker build -t deepseek-ocr-vllm:latest .

# 2. 运行容器
docker run -d \
  --name deepseek-ocr-api \
  --gpus all \
  --shm-size 8g \
  -p 8000:8000 \
  -v $(pwd)/DeepSeek-OCR-vllm/models:/app/models:ro \
  -e LOG_LEVEL=INFO \
  -e VLLM_USE_V1=1 \
  deepseek-ocr-vllm:latest

# 3. 查看日志
docker logs -f deepseek-ocr-api

# 4. 停止容器
docker stop deepseek-ocr-api
docker rm deepseek-ocr-api
```

## 🔧 配置选项

### 环境变量

- `HOST`: 服务绑定地址 (默认: 0.0.0.0)
- `PORT`: 服务端口 (默认: 8000)
- `WORKERS`: Worker 数量 (默认: 1)
- `LOG_LEVEL`: 日志级别 INFO/DEBUG (默认: INFO)
- `VLLM_USE_V1`: 使用 vLLM V1 引擎 (默认: 1)
- `CUDA_VISIBLE_DEVICES`: 指定 GPU (默认: 0)

### 启用调试日志

```bash
docker run -d \
  --name deepseek-ocr-api \
  --gpus all \
  --shm-size 8g \
  -p 8000:8000 \
  -v $(pwd)/DeepSeek-OCR-vllm/models:/app/models:ro \
  -e LOG_LEVEL=DEBUG \
  deepseek-ocr-vllm:latest
```

### 指定 GPU

```bash
# 使用特定 GPU
docker run -d \
  --gpus '"device=0"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  ...

# 使用多个 GPU
docker run -d \
  --gpus '"device=0,1"' \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  ...
```

## 📊 测试 API

### 健康检查

```bash
curl http://localhost:8000/health
```

### 列出模型

```bash
curl http://localhost:8000/v1/models
```

### OCR 请求示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Convert the document to markdown."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQ..."
            }
          }
        ]
      }
    ],
    "max_tokens": 8192,
    "temperature": 0.0
  }'
```

## 🔍 故障排查

### 查看容器日志

```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs -f deepseek-ocr-api
```

### 进入容器调试

```bash
# Docker Compose
docker-compose exec deepseek-ocr-api bash

# Docker
docker exec -it deepseek-ocr-api bash
```

### 检查 GPU

```bash
docker exec -it deepseek-ocr-api nvidia-smi
```

### 常见问题

#### 1. CUDA 版本不兼容

**错误**: `cuda>=12.8, please update your driver`

**解决**: Dockerfile 已设置为使用 `v0.6.3.post1` 镜像，兼容 CUDA 12.1/12.2

#### 2. 显存不足

**错误**: `CUDA out of memory`

**解决**: 
- 减少 `gpu_memory_utilization` (在 openai_api_server.py 中修改)
- 降低 `MAX_CROPS` (在 config.py 中修改)
- 使用更大显存的 GPU

#### 3. 模型加载失败

**错误**: `Model not found`

**解决**: 确保模型文件已正确挂载到容器中
```bash
ls -la ./DeepSeek-OCR-vllm/models/
```

## 📦 镜像信息

- **基础镜像**: vllm/vllm-openai:v0.6.3.post1
- **CUDA 支持**: 12.1/12.2
- **Python**: 3.10+
- **预装组件**: PyTorch, vLLM, FastAPI, Uvicorn

## 🔄 更新镜像

```bash
# 重新构建
docker-compose build --no-cache

# 或
docker build --no-cache -t deepseek-ocr-vllm:latest .
```

## 📝 生产环境建议

1. 使用特定版本标签而不是 `latest`
2. 配置资源限制 (CPU/内存)
3. 设置适当的重启策略
4. 配置日志轮转
5. 使用反向代理 (Nginx/Traefik)
6. 启用 HTTPS
7. 配置监控和告警

## 🌐 网络配置

如果需要在其他机器访问，确保防火墙已开放端口：

```bash
# Ubuntu/Debian
sudo ufw allow 8000

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

