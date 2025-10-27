# DeepSeek-OCR OpenAI-Compatible API Guide

本指南介绍如何使用 OpenAI 兼容接口调用 DeepSeek-OCR。

## 目录

- [快速开始](#快速开始)
- [API 端点](#api-端点)
- [使用示例](#使用示例)
- [支持的功能](#支持的功能)
- [参数说明](#参数说明)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 安装依赖

```bash
# 安装 API 服务器依赖
pip install -r requirements_api.txt

# 确保已安装 DeepSeek-OCR 的基础依赖
pip install -r requirements.txt
```

### 2. 配置模型路径

编辑 `config.py`，设置模型路径：

```python
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # 或本地模型路径
```

### 3. 启动服务器

```bash
# 默认启动（0.0.0.0:8000）
python openai_api_server.py

# 自定义主机和端口
python openai_api_server.py --host 0.0.0.0 --port 8080

# 查看所有选项
python openai_api_server.py --help
```

服务器启动后，你会看到：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Initializing DeepSeek-OCR engine...
INFO:     Engine initialized successfully!
```

### 4. 测试 API

```bash
# 使用测试脚本
python test_openai_client.py path/to/your/image.jpg

# 或使用 curl
curl http://localhost:8000/health
```

---

## API 端点

### 1. Chat Completions

**端点**: `POST /v1/chat/completions`

**功能**: 与 OpenAI Chat Completions API 兼容的 OCR 接口

**示例请求**:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Convert the document to markdown."},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            }
          }
        ]
      }
    ],
    "temperature": 0.0,
    "max_tokens": 4000
  }'
```

### 2. List Models

**端点**: `GET /v1/models`

**功能**: 列出可用模型

```bash
curl http://localhost:8000/v1/models
```

### 3. Health Check

**端点**: `GET /health`

**功能**: 检查服务器状态

```bash
curl http://localhost:8000/health
```

---

## 使用示例

### Python 客户端（推荐）

使用官方 OpenAI Python 库：

```python
from openai import OpenAI
import base64

# 初始化客户端
client = OpenAI(
    api_key="EMPTY",  # DeepSeek-OCR 不需要 API key
    base_url="http://localhost:8000/v1"
)

# 方法1: 使用 Base64 编码图像
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("invoice.jpg")

response = client.chat.completions.create(
    model="deepseek-ocr",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.0,
    max_tokens=4000
)

print(response.choices[0].message.content)
```

### 流式输出

```python
stream = client.chat.completions.create(
    model="deepseek-ocr",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "OCR this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ],
    stream=True,  # 启用流式输出
    max_tokens=4000
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### TypeScript/JavaScript 客户端

```typescript
import OpenAI from 'openai';
import fs from 'fs';

const client = new OpenAI({
  apiKey: 'EMPTY',
  baseURL: 'http://localhost:8000/v1'
});

// 读取并编码图像
const imageBuffer = fs.readFileSync('invoice.jpg');
const base64Image = imageBuffer.toString('base64');

const response = await client.chat.completions.create({
  model: 'deepseek-ocr',
  messages: [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Convert the document to markdown.' },
        {
          type: 'image_url',
          image_url: {
            url: `data:image/jpeg;base64,${base64Image}`
          }
        }
      ]
    }
  ],
  temperature: 0.0,
  max_tokens: 4000
});

console.log(response.choices[0].message.content);
```

### cURL 命令行

```bash
# 使用 base64 编码的图像
BASE64_IMAGE=$(base64 -i invoice.jpg)

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"deepseek-ocr\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"Convert the document to markdown.\"},
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"data:image/jpeg;base64,$BASE64_IMAGE\"
            }
          }
        ]
      }
    ],
    \"temperature\": 0.0,
    \"max_tokens\": 4000
  }"
```

---

## 支持的功能

### ✅ 已支持

- ✅ OpenAI Chat Completions API 格式
- ✅ 多模态输入（文本 + 图像）
- ✅ Base64 编码图像
- ✅ 图像 URL（HTTP/HTTPS）
- ✅ 流式响应（Server-Sent Events）
- ✅ 非流式响应
- ✅ 标准参数：temperature, max_tokens, top_p
- ✅ Token 使用统计
- ✅ DeepSeek-OCR 特殊标记（如 `<|grounding|>`）

### ❌ 暂不支持

- ❌ 多图像输入（当前仅支持单图像）
- ❌ 函数调用（tools/function calling）
- ❌ JSON mode
- ❌ 视觉详细度控制（detail 参数）
- ❌ logprobs

---

## 参数说明

### 请求参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | ✅ | - | 模型名称，使用 "deepseek-ocr" |
| `messages` | array | ✅ | - | 对话消息列表 |
| `temperature` | float | ❌ | 0.0 | 采样温度 (0.0-2.0) |
| `max_tokens` | int | ❌ | 8192 | 最大生成 token 数 |
| `max_completion_tokens` | int | ❌ | - | 同 max_tokens（新参数，优先级更高） |
| `top_p` | float | ❌ | 1.0 | 核采样参数 (0.0-1.0) |
| `stream` | boolean | ❌ | false | 是否启用流式输出 |
| `n` | int | ❌ | 1 | 生成的回复数量（当前仅支持 1） |

### Message 格式

#### 文本消息

```json
{
  "role": "user",
  "content": "Hello!"
}
```

#### 多模态消息

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Convert to markdown."},
    {
      "type": "image_url",
      "image_url": {"url": "data:image/jpeg;base64,..."}
    }
  ]
}
```

---

## DeepSeek-OCR 特殊 Prompt

DeepSeek-OCR 支持以下特殊提示词：

### 1. 文档转 Markdown（带布局）

```python
"<|grounding|>Convert the document to markdown."
```

### 2. 通用 OCR（带定位）

```python
"<|grounding|>OCR this image."
```

### 3. 纯文本提取（无布局）

```python
"Free OCR."
```

### 4. 图表解析

```python
"Parse the figure."
```

### 5. 图像描述

```python
"Describe this image in detail."
```

### 6. 文本定位

```python
"Locate <|ref|>Invoice Number<|/ref|> in the image."
```

---

## 配置选项

编辑 `config.py` 自定义行为：

```python
# 分辨率模式
BASE_SIZE = 1024      # 全局视图分辨率
IMAGE_SIZE = 640      # 切片分辨率
CROP_MODE = True      # 是否启用动态切片

# 切片控制
MIN_CROPS = 2         # 最小切片数
MAX_CROPS = 6         # 最大切片数（根据 GPU 内存调整）

# 性能
MAX_CONCURRENCY = 100 # 最大并发数
```

### 分辨率模式参考

| 模式 | BASE_SIZE | IMAGE_SIZE | CROP_MODE | 说明 |
|------|-----------|------------|-----------|------|
| Tiny | 512 | 512 | False | 快速，低质量 |
| Small | 640 | 640 | False | 平衡 |
| Base | 1024 | 1024 | False | 高质量 |
| Large | 1280 | 1280 | False | 最高质量 |
| **Gundam** | 1024 | 640 | **True** | **推荐**（动态切片） |

---

## 常见问题

### Q1: 如何处理大图像？

启用 `CROP_MODE = True`（Gundam 模式），系统会自动将大图切分处理。

### Q2: 如何提高吞吐量？

调整以下参数：

```python
# config.py
MAX_CONCURRENCY = 200  # 增加并发数
MAX_CROPS = 4          # 减少切片数（降低精度，提高速度）
```

### Q3: 如何使用公开 URL 的图像？

直接在 `image_url` 中使用 HTTP/HTTPS URL：

```python
{
  "type": "image_url",
  "image_url": {"url": "https://example.com/image.jpg"}
}
```

### Q4: 响应太慢怎么办？

1. 使用流式输出（`stream=True`）获得更快的首字节响应
2. 降低分辨率模式（使用 Tiny/Small）
3. 减少 `MAX_CROPS` 数量
4. 调整 GPU 利用率（默认 0.9，可降低到 0.75）

### Q5: 如何获取边界框信息？

使用 `<|grounding|>` 标记，响应会包含 `<|ref|>...<|/ref|><|det|>...<|/det|>` 格式的坐标信息：

```python
response = client.chat.completions.create(
    model="deepseek-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "<|grounding|>OCR this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)

# 解析坐标
import re
pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
matches = re.findall(pattern, response.choices[0].message.content)
for text, coords in matches:
    print(f"Text: {text}, Coords: {coords}")
```

### Q6: 支持哪些图像格式？

支持常见格式：JPG, PNG, JPEG, BMP, GIF（会自动转为 RGB）

### Q7: 如何在生产环境部署？

```bash
# 使用 gunicorn + uvicorn workers
gunicorn openai_api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300

# 或使用 Docker（需自行编写 Dockerfile）
```

---

## 性能优化建议

### GPU 内存优化

```python
# AsyncEngineArgs
gpu_memory_utilization=0.75  # 降低显存使用（默认 0.9）
max_model_len=4096          # 减少最大序列长度
```

### 批处理推荐

对于批量处理，使用多线程客户端：

```python
from concurrent.futures import ThreadPoolExecutor
import glob

def process_image(image_path):
    # ... OCR 逻辑
    pass

image_files = glob.glob("images/*.jpg")
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_image, image_files))
```

---

## 更新日志

### v1.0.0 (2025-01)

- ✅ 实现 OpenAI Chat Completions API
- ✅ 支持多模态输入（Base64 和 URL）
- ✅ 实现流式和非流式响应
- ✅ 添加健康检查和模型列表端点

---

## 许可证

与 DeepSeek-OCR 项目保持一致。

## 反馈与贡献

如有问题或建议，请提交 Issue 或 Pull Request。
