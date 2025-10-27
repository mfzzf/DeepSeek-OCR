"""
OpenAI-compatible API server for DeepSeek-OCR using vLLM.

This server provides OpenAI-compatible endpoints for DeepSeek-OCR inference,
supporting multi-modal image input (URL and base64) and streaming responses.
"""

import asyncio
import base64
import io
import os
import time
import uuid
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

from deepseek_ocr import DeepseekOCRForCausalLM
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from config import MODEL_PATH, CROP_MODE

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Environment setup
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'

app = FastAPI(title="DeepSeek-OCR OpenAI-Compatible API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[AsyncLLMEngine] = None


# ===== OpenAI API Models =====

class ImageUrl(BaseModel):
    """Image URL or base64 encoded image."""
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ContentPart(BaseModel):
    """Content part for multi-modal messages."""
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion API request."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(default=8192, ge=1)
    max_completion_tokens: Optional[int] = None  # New parameter, takes precedence
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict] = None


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion API response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionStreamChoice(BaseModel):
    """Streaming completion choice."""
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI Chat Completion streaming response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """List of available models."""
    object: str = "list"
    data: List[ModelInfo]


# ===== Helper Functions =====

def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL or base64 string."""
    if url.startswith("data:image"):
        # Parse base64 encoded image
        header, base64_data = url.split(",", 1)
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
    elif url.startswith("http://") or url.startswith("https://"):
        # Load from URL
        import requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
    else:
        raise ValueError(f"Unsupported URL format: {url}")

    return image.convert("RGB")


def parse_messages(messages: List[ChatMessage]) -> tuple[str, Optional[Image.Image]]:
    """
    Parse OpenAI messages format into prompt and image.

    Returns:
        tuple: (prompt_text, image)
    """
    prompt_parts = []
    image = None

    for message in messages:
        role = message.role
        content = message.content

        # Handle string content
        if isinstance(content, str):
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Handle multi-modal content
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    # Load the image (only support one image for now)
                    if image is None:
                        try:
                            image = load_image_from_url(part.image_url.url)
                        except Exception as e:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Failed to load image: {str(e)}"
                            )

            if text_parts:
                combined_text = " ".join(text_parts)
                if role == "system":
                    prompt_parts.append(f"System: {combined_text}")
                elif role == "user":
                    # If there's an image, add the <image> token
                    if image is not None:
                        prompt_parts.append(f"<image>\n{combined_text}")
                    else:
                        prompt_parts.append(combined_text)
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {combined_text}")

    prompt = "\n".join(prompt_parts)
    return prompt, image


async def generate_stream(
    request_id: str,
    prompt: str,
    image_features: Optional[Dict],
    sampling_params: SamplingParams,
    model_name: str
) -> AsyncGenerator[str, None]:
    """Generate streaming responses."""
    created_time = int(time.time())

    # First chunk with role
    first_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={"role": "assistant", "content": ""},
                finish_reason=None
            )
        ]
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Generate request
    if image_features:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }
    else:
        request = {"prompt": prompt}

    printed_length = 0
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            printed_length = len(full_text)

            # Send text chunk
            if new_text:
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": new_text},
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk with finish_reason
    final_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Initialize the vLLM engine on startup."""
    global engine

    print("Initializing DeepSeek-OCR engine...")

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Engine initialized successfully!")


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models."""
    return ModelList(
        data=[
            ModelInfo(
                id="deepseek-ocr",
                created=int(time.time()),
                owned_by="deepseek-ai"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, StreamingResponse]:
    """
    Create a chat completion using OpenAI-compatible API.

    Supports:
    - Multi-modal image input (URL and base64)
    - Streaming and non-streaming responses
    - Standard OpenAI parameters
    """
    try:
        # Parse messages
        prompt, image = parse_messages(request.messages)

        # Process image if present
        image_features = None
        if image is not None:
            processor = DeepseekOCRProcessor()
            image_features = processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )

        # Setup sampling parameters
        max_tokens = request.max_completion_tokens or request.max_tokens

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822}
            )
        ]

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        # Generate request ID
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                generate_stream(request_id, prompt, image_features, sampling_params, request.model),
                media_type="text/event-stream"
            )

        # Non-streaming response
        if image_features:
            gen_request = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features}
            }
        else:
            gen_request = {"prompt": prompt}

        final_output = None
        async for request_output in engine.generate(gen_request, sampling_params, request_id):
            if request_output.outputs:
                final_output = request_output.outputs[0]

        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")

        # Calculate token counts (approximate)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(final_output.text.split())

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=final_output.text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DeepSeek-OCR OpenAI-Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek-OCR OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
