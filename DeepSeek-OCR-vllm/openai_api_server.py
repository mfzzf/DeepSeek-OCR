"""
OpenAI-compatible API server for DeepSeek-OCR using vLLM.

This server provides OpenAI-compatible endpoints for DeepSeek-OCR inference,
supporting multi-modal image input (URL and base64) and streaming responses.
"""

import os
import sys

# Set environment variable BEFORE any vllm imports
# Enable V1 engine (fixed compatibility with custom processors)
os.environ['VLLM_USE_V1'] = '1'

import asyncio
import base64
import io
import logging
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

# Setup logging
# Use DEBUG level to get more detailed logs for troubleshooting
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized at level: {LOG_LEVEL}")

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

# Use V1 AsyncLLMEngine (compatibility fixed)
from vllm import AsyncLLMEngine

from deepseek_ocr import DeepseekOCRForCausalLM
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, CROP_MODE, TOKENIZER
# Note: NoRepeatNGramLogitsProcessor is not compatible with V1 engine

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Environment setup
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

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
    usage: Optional[Usage] = None  # Include usage in final chunk


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
    url_type = "base64" if url.startswith("data:image") else "url"
    logger.info(f"Loading image from {url_type}")
    
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

    rgb_image = image.convert("RGB")
    logger.info(f"Image loaded successfully: size={rgb_image.size}, mode={rgb_image.mode}")
    return rgb_image


def parse_messages(messages: List[ChatMessage]) -> tuple[str, Optional[Image.Image]]:
    """
    Parse OpenAI messages format into prompt and image.

    Returns:
        tuple: (prompt_text, image)
    """
    logger.info(f"Parsing {len(messages)} messages")
    prompt_parts = []
    image = None

    for idx, message in enumerate(messages):
        role = message.role
        content = message.content
        logger.debug(f"Processing message {idx}: role={role}, content_type={type(content).__name__}")

        # Handle string content
        if isinstance(content, str):
            logger.debug(f"Message {idx} is string content, length={len(content)}")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Handle multi-modal content
        elif isinstance(content, list):
            logger.debug(f"Message {idx} is list content with {len(content)} parts")
            text_parts = []
            for part_idx, part in enumerate(content):
                logger.debug(f"Message {idx}, part {part_idx}: type={part.type}")
                if part.type == "text" and part.text:
                    logger.debug(f"Found text part, length={len(part.text)}")
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    logger.info(f"Found image_url in message {idx}, part {part_idx}")
                    # Load the image (only support one image for now)
                    if image is None:
                        try:
                            image = load_image_from_url(part.image_url.url)
                            logger.info(f"Successfully loaded image from message {idx}")
                        except Exception as e:
                            logger.error(f"Failed to load image: {str(e)}", exc_info=True)
                            raise HTTPException(
                                status_code=400,
                                detail=f"Failed to load image: {str(e)}"
                            )
                    else:
                        logger.warning(f"Skipping additional image in message {idx} (only one image supported)")

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
    logger.info(f"Message parsing complete: prompt_length={len(prompt)}, has_image={image is not None}")
    return prompt, image


async def generate_stream(
    request_id: str,
    prompt: str,
    image: Optional[Image.Image],
    image_features: Optional[Dict],
    sampling_params: SamplingParams,
    model_name: str
) -> AsyncGenerator[str, None]:
    """Generate streaming responses."""
    logger.info(f"[{request_id}] Starting streaming generation")
    logger.info(f"[{request_id}] Sampling params: temperature={sampling_params.temperature}, "
                f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
    
    start_time = time.time()
    created_time = int(start_time)

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
        logger.debug(f"[{request_id}] Streaming request with image features")
    else:
        request = {"prompt": prompt}
        logger.debug(f"[{request_id}] Streaming request without image")

    printed_length = 0
    token_count = 0
    iteration_count = 0
    full_text = ""
    async for request_output in engine.generate(request, sampling_params, request_id):
        iteration_count += 1
        if request_output.outputs:
            output_obj = request_output.outputs[0]
            full_text = output_obj.text if hasattr(output_obj, 'text') else str(output_obj)
            new_text = full_text[printed_length:]
            printed_length = len(full_text)

            # Log progress every 10 iterations
            if iteration_count % 10 == 0:
                logger.debug(f"[{request_id}] Streaming iteration {iteration_count}: "
                           f"total_length={printed_length}, new_text_length={len(new_text)}")

            # Send text chunk
            if new_text:
                token_count += 1
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

    # Final chunk with finish_reason and usage information
    elapsed_time = time.time() - start_time
    
    # Calculate token counts accurately
    try:
        # Count prompt tokens
        prompt_token_ids = TOKENIZER.encode(prompt, add_special_tokens=False)
        prompt_tokens = len(prompt_token_ids)
        
        # Add image tokens if image is present
        if image is not None and image_features is not None:
            if isinstance(image_features, dict) and 'pixel_values' in image_features:
                pixel_values = image_features['pixel_values']
                if hasattr(pixel_values, 'shape'):
                    if len(pixel_values.shape) >= 4:
                        num_image_patches = pixel_values.shape[0] if len(pixel_values.shape) == 4 else pixel_values.shape[1]
                        prompt_tokens += num_image_patches * 256
                        logger.debug(f"[{request_id}] Streaming: Added image tokens for {num_image_patches} patches")
        
        # Count completion tokens from generated text
        output_token_ids = TOKENIZER.encode(full_text, add_special_tokens=False)
        completion_tokens = len(output_token_ids)
        
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to accurately count tokens in streaming, using fallback: {e}")
        prompt_tokens = len(prompt.split())
        completion_tokens = len(full_text.split()) if full_text else 0
    
    total_tokens = prompt_tokens + completion_tokens
    
    throughput_stream = completion_tokens / elapsed_time if elapsed_time > 0 and completion_tokens > 0 else 0.0
    logger.info(f"[{request_id}] Streaming generation complete: "
                f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                f"total_tokens={total_tokens}, chunks={token_count}, iterations={iteration_count}, "
                f"elapsed={elapsed_time:.2f}s, throughput={throughput_stream:.2f} tokens/s")
    
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
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Initialize the vLLM engine on startup."""
    global engine

    logger.info("="*80)
    logger.info("Starting DeepSeek-OCR vLLM engine initialization...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Crop mode: {CROP_MODE}")
    logger.info("="*80)

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

    logger.info("Engine configuration:")
    logger.info(f"  - block_size: 256")
    logger.info(f"  - max_model_len: 8192")
    logger.info(f"  - tensor_parallel_size: 1")
    logger.info(f"  - gpu_memory_utilization: 0.9")
    logger.info(f"  - enforce_eager: False")
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("="*80)
    logger.info("✓ DeepSeek-OCR engine initialized successfully!")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global engine
    
    if engine is not None:
        logger.info("="*80)
        logger.info("Shutting down DeepSeek-OCR engine...")
        try:
            # Shutdown the engine to free GPU memory
            await engine.shutdown()
            logger.info("✓ Engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}", exc_info=True)
        finally:
            engine = None
            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("✓ GPU memory cleared")
            logger.info("="*80)


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


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion using OpenAI-compatible API.

    Supports:
    - Multi-modal image input (URL and base64)
    - Streaming and non-streaming responses
    - Standard OpenAI parameters
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.info(f"[{request_id}] Received chat completion request: model={request.model}, "
                f"stream={request.stream}, messages_count={len(request.messages)}")
    
    try:
        # Parse messages
        prompt, image = parse_messages(request.messages)

        # Process image if present
        image_features = None
        if image is not None:
            logger.info(f"[{request_id}] Processing image with crop_mode={CROP_MODE}")
            processor = DeepseekOCRProcessor()
            image_features = processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
            logger.info(f"[{request_id}] Image processing complete")

        # Setup sampling parameters
        max_tokens = request.max_completion_tokens or request.max_tokens
        logger.info(f"[{request_id}] Sampling parameters: temperature={request.temperature}, "
                    f"top_p={request.top_p}, max_tokens={max_tokens}")

        # Note: V1 engine does not support per-request logits processors
        # The NoRepeatNGramLogitsProcessor would need to be implemented differently for V1
        # For now, we disable it for V1 compatibility
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            skip_special_tokens=False,
        )

        # Handle streaming
        if request.stream:
            logger.info(f"[{request_id}] Initiating streaming response")
            return StreamingResponse(
                generate_stream(request_id, prompt, image, image_features, sampling_params, request.model),
                media_type="text/event-stream"
            )

        # Non-streaming response
        logger.info(f"[{request_id}] Starting non-streaming generation")
        start_time = time.time()
        if image_features:
            gen_request = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features}
            }
            logger.debug(f"[{request_id}] Non-streaming request with image features, prompt_length={len(prompt)}")
        else:
            gen_request = {"prompt": prompt}
            logger.debug(f"[{request_id}] Non-streaming request without image, prompt_length={len(prompt)}")

        final_output = None
        iteration_count = 0
        async for request_output in engine.generate(gen_request, sampling_params, request_id):
            iteration_count += 1
            logger.debug(f"[{request_id}] Generation iteration {iteration_count}, "
                        f"outputs_count={len(request_output.outputs) if request_output.outputs else 0}")
            if request_output.outputs:
                final_output = request_output.outputs[0]
                logger.debug(f"[{request_id}] Current output length: {len(final_output.text)}")

        if final_output is None:
            logger.error(f"[{request_id}] Generation failed: no output generated after {iteration_count} iterations")
            raise HTTPException(status_code=500, detail="Generation failed")
        
        # Log detailed output information
        logger.info(f"[{request_id}] Generation loop completed: iterations={iteration_count}")
        logger.debug(f"[{request_id}] final_output type: {type(final_output)}")
        logger.debug(f"[{request_id}] final_output attributes: {dir(final_output)}")
        
        # Get the actual text output
        output_text = final_output.text if hasattr(final_output, 'text') else str(final_output)
        logger.info(f"[{request_id}] Output text length: {len(output_text)}")
        
        # Log first 200 chars of output for debugging
        if len(output_text) > 0:
            preview = output_text[:200].replace('\n', '\\n')
            logger.debug(f"[{request_id}] Output preview: {preview}...")
        
        elapsed_time = time.time() - start_time
        
        # Calculate token counts accurately using tokenizer
        # Count prompt tokens
        try:
            prompt_token_ids = TOKENIZER.encode(prompt, add_special_tokens=False)
            prompt_tokens = len(prompt_token_ids)
            
            # Add image tokens if image is present
            if image is not None and image_features is not None:
                # Image features contain the actual image tokens
                # Each image can have multiple tokens depending on cropping
                if isinstance(image_features, dict) and 'pixel_values' in image_features:
                    # Estimate image tokens based on pixel_values shape
                    pixel_values = image_features['pixel_values']
                    if hasattr(pixel_values, 'shape'):
                        # Typically: (batch, num_patches, channels, height, width)
                        # or (num_patches, channels, height, width)
                        if len(pixel_values.shape) >= 4:
                            num_image_patches = pixel_values.shape[0] if len(pixel_values.shape) == 4 else pixel_values.shape[1]
                            # Each patch typically corresponds to multiple tokens
                            # For DeepSeek-OCR, this can vary based on the model architecture
                            prompt_tokens += num_image_patches * 256  # Approximate tokens per patch
                            logger.debug(f"[{request_id}] Added image tokens: num_patches={num_image_patches}")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to accurately count prompt tokens, using fallback: {e}")
            prompt_tokens = len(prompt.split())
        
        # Count completion tokens from actual token_ids
        if hasattr(final_output, 'token_ids') and final_output.token_ids:
            completion_tokens = len(final_output.token_ids)
            logger.debug(f"[{request_id}] Completion tokens from token_ids: {completion_tokens}")
        else:
            # Fallback: encode the output text
            try:
                output_token_ids = TOKENIZER.encode(output_text, add_special_tokens=False)
                completion_tokens = len(output_token_ids)
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to accurately count completion tokens, using fallback: {e}")
                completion_tokens = len(output_text.split()) if output_text else 0
        
        if completion_tokens == 0:
            logger.warning(f"[{request_id}] Generated empty response! "
                          f"output_text='{output_text}' (length={len(output_text)})")
            if hasattr(final_output, 'token_ids'):
                logger.debug(f"[{request_id}] token_ids: {final_output.token_ids}")
        
        throughput = completion_tokens / elapsed_time if elapsed_time > 0 and completion_tokens > 0 else 0.0
        logger.info(f"[{request_id}] Generation complete: "
                    f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                    f"elapsed={elapsed_time:.2f}s, throughput={throughput:.2f} tokens/s")

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=output_text
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
        logger.error(f"[{request_id}] Error during generation: {str(e)}", exc_info=True)
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
    import signal

    def signal_handler(signum, frame):
        """Handle shutdown signals to ensure proper cleanup."""
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        # Force GPU memory cleanup
        if engine is not None:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="DeepSeek-OCR OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info"
        )
    finally:
        # Ensure cleanup on exit
        print("Cleaning up GPU memory...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
