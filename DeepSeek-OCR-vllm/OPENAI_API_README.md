# DeepSeek-OCR OpenAI-Compatible API

为 DeepSeek-OCR 添加了完整的 OpenAI 兼容接口支持，可以使用标准的 OpenAI SDK 调用。

## ✨ 特性

- ✅ 完全兼容 OpenAI Chat Completions API
- ✅ 支持多模态输入（Base64 和 URL 图像）
- ✅ 支持流式和非流式响应
- ✅ 支持 DeepSeek-OCR 所有特殊标记
- ✅ 开箱即用，零配置启动

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_api.txt
```

### 2. 启动服务器

```bash
# 方式1: 直接启动
python openai_api_server.py

# 方式2: 使用启动脚本
chmod +x start_server.sh
./start_server.sh

# 方式3: 自定义端口
python openai_api_server.py --port 8080
```

### 3. 测试 API

```bash
# 列出模型
curl http://localhost:8000/v1/models

# 测试 OCR（需要提供图像路径）
python test_openai_client.py path/to/image.jpg
```

## 📖 使用示例

### Python 示例

```python
from openai import OpenAI
import base64

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

# 读取并编码图像
with open("invoice.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

# 调用 API
response = client.chat.completions.create(
    model="deepseek-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Convert the document to markdown."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    }],
    temperature=0.0,
    max_tokens=4000
)

print(response.choices[0].message.content)
```

### cURL 示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "OCR this image."},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,..."}
        }
      ]
    }],
    "max_tokens": 2000
  }'
```

### 流式输出

```python
stream = client.chat.completions.create(
    model="deepseek-ocr",
    messages=[...],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 📚 文件说明

| 文件 | 说明 |
|------|------|
| `openai_api_server.py` | 主 API 服务器实现 |
| `test_openai_client.py` | 测试客户端脚本 |
| `requirements_api.txt` | API 服务器依赖 |
| `OPENAI_API_GUIDE.md` | 详细使用指南 |
| `start_server.sh` | 启动脚本 |

## 🔧 配置

在 `config.py` 中配置：

```python
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # 模型路径
BASE_SIZE = 1024                          # 全局视图分辨率
IMAGE_SIZE = 640                          # 切片分辨率
CROP_MODE = True                          # 启用动态切片
MAX_CONCURRENCY = 100                     # 最大并发数
```

## 📡 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | Chat Completions API |
| `/v1/models` | GET | 列出可用模型 |
| `/health` | GET | 健康检查 |
| `/` | GET | API 信息 |

## 🎯 DeepSeek-OCR Prompts

```python
# 文档转 Markdown
"<|grounding|>Convert the document to markdown."

# 通用 OCR
"<|grounding|>OCR this image."

# 纯文本
"Free OCR."

# 图表解析
"Parse the figure."

# 文本定位
"Locate <|ref|>text<|/ref|> in the image."
```

## ⚡ 性能优化

### GPU 内存

```python
# 在 openai_api_server.py 的 startup_event() 中调整
gpu_memory_utilization=0.75  # 降低显存使用
max_model_len=4096          # 减少最大序列长度
```

### 分辨率模式

```python
# config.py
# Tiny (快速): BASE_SIZE=512, IMAGE_SIZE=512, CROP_MODE=False
# Gundam (推荐): BASE_SIZE=1024, IMAGE_SIZE=640, CROP_MODE=True
```

## 🐛 故障排查

### 问题1: 端口已被占用

```bash
# 查看端口占用
lsof -i :8000

# 使用其他端口
python openai_api_server.py --port 8080
```

### 问题2: CUDA 内存不足

减少显存使用：

```python
gpu_memory_utilization=0.6
MAX_CROPS = 4  # 在 config.py 中
```

### 问题3: 图像加载失败

确保图像格式正确：

```python
# 支持的格式
['jpg', 'jpeg', 'png', 'bmp', 'gif']
```

## 📝 完整文档

查看 [OPENAI_API_GUIDE.md](OPENAI_API_GUIDE.md) 获取完整使用指南。

## 🌟 特别说明

本实现完全兼容 OpenAI SDK，你可以：

1. 使用任何支持 OpenAI API 的工具/库
2. 无缝切换到 DeepSeek-OCR（只需修改 `base_url`）
3. 在现有 OpenAI 应用中集成 OCR 能力

## 📄 许可证

与 DeepSeek-OCR 项目保持一致。
