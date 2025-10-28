### **DeepSeek-OCR OpenAI 兼容 API 技术文档**

#### **1. 概述**

本文档旨在阐述 `openai_api_server.py` 如何构建一个与 OpenAI API 规范兼容的服务层，以便将底层的 DeepSeek-OCR 模型能力通过标准化的接口暴露出来。核心目标是实现“即插即用”，让熟悉 OpenAI API 的开发者能够无缝迁移。

#### **2. 实现 OpenAI 兼容性的核心设计**

该服务通过以下几个关键方面实现了与 OpenAI 的兼容：

##### **2.1. 兼容的 API Endpoints**

服务使用 FastAPI 框架搭建，并注册了两个核心的、与 OpenAI 完全一致的 API 路由：

*   `GET /v1/models`: 用于获取当前服务可用的模型列表。
    ```1:6:DeepSeek-OCR-vllm/openai_api_server.py
    @app.get("/v1/models")
    async def list_models() -> ModelList:
        """List available models."""
        return ModelList(
            data=[
                ModelInfo(
    ```
*   `POST /v1/chat/completions`: 用于创建聊天补全任务，是 API 的核心功能所在，支持流式（streaming）和非流式请求。
    ```1:2:DeepSeek-OCR-vllm/openai_api_server.py
    @app.post("/v1/chat/completions", response_model=None)
    async def create_chat_completion(request: ChatCompletionRequest):
    ```

##### **2.2. 兼容的数据结构 (Pydantic Models)**

代码使用 Pydantic 定义了与 OpenAI API 请求和响应体完全一致的数据模型。这确保了数据交互的标准化。

*   **请求模型**: `ChatCompletionRequest` (L96-L111) 精确定义了请求体，包含了 `model`, `messages`, `temperature`, `stream`, `max_tokens` 等标准字段。
    ```96:104:DeepSeek-OCR-vllm/openai_api_server.py
    class ChatCompletionRequest(BaseModel):
        """OpenAI Chat Completion API request."""
        model: str
        messages: List[ChatMessage]
        temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
        top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
        n: Optional[int] = Field(default=1, ge=1)
        stream: Optional[bool] = False
        max_tokens: Optional[int] = Field(default=8192, ge=1)
    ```
*   **多模态消息结构**: `ChatMessage`, `ContentPart`, `ImageUrl` (L77-L94) 等模型支持 OpenAI GPT-4 Vision 所采用的多模态输入格式，即在 `content` 字段中传入一个包含文本和图片 URL（或 Base64）的数组。
*   **响应模型**:
    *   `ChatCompletionResponse` (L128-L136): 用于标准的非流式响应。
    *   `ChatCompletionStreamResponse` (L145-L153): 用于流式响应的数据块（chunk）。
    *   `Usage` (L121-L126): 用于统计 `prompt_tokens`, `completion_tokens` 等用量信息。

##### **2.3. 兼容的多模态输入处理**

服务能够处理图像输入，兼容 OpenAI 的 `image_url` 格式。

*   **消息解析**: `parse_messages` 函数 (L195-L262) 负责解析输入的 `messages` 列表。它会遍历消息内容，提取文本和图片。
*   **图片加载**: `load_image_from_url` 函数 (L171-L193) 支持两种图片格式：
    1.  **URL**: `http://` 或 `https://` 开头的公开可访问图片链接。
    2.  **Base64**: `data:image/...;base64,...` 格式的内嵌图片数据。

解析成功后，图片会被送入 `DeepseekOCRProcessor` 进行预处理，转换成模型可接受的特征。

##### **2.4. 兼容的流式响应 (Server-Sent Events)**

当请求中 `stream=True` 时，API 会返回一个 `StreamingResponse`，其内容格式为 `text/event-stream`。

`generate_stream` 异步生成器函数 (L264-L395) 负责逐步生成符合 OpenAI SSE 格式的响应数据块。

1.  **起始块**: 发送一个包含 `role: "assistant"` 的初始数据块。
2.  **内容块**: vLLM 引擎每生成一部分文本，就将其包装在 `ChatCompletionStreamResponse` 中，并以 `data: {...}\n\n` 的格式发送给客户端。
3.  **结束块**: 生成结束后，发送一个 `finish_reason: "stop"` 的数据块，并附上 `usage` 统计信息。
4.  **最终信号**: 最后发送 `data: [DONE]\n\n` 标志流结束。

#### **3. API 使用示例**

你可以使用 `curl` 或任何支持 OpenAI API 的客户端来调用此服务。假设服务运行在 `localhost:8000`。

##### **示例 1: 标准文本问答 (非流式)**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": "Hello, what is DeepSeek-OCR?"
      }
    ]
  }'
```

##### **示例 2: 多模态图片识别 (Base64)**

```bash
# 将你的图片转为 Base64
# Linux: base64 -w0 my_image.png
# macOS: base64 -i my_image.png
IMAGE_B64=$(base64 -i my_image.png)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What does the text in this image say?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,'"$IMAGE_B64"'"
            }
          }
        ]
      }
    ],
    "max_tokens": 2048
  }'
```

##### **示例 3: 流式响应**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a long story."
      }
    ],
    "stream": true
  }'
```
你将会收到一系列 Server-Sent Events (SSE) 数据块。