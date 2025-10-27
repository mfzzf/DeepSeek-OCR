"""
Test client for DeepSeek-OCR OpenAI-compatible API.

This script demonstrates how to use the OpenAI Python client
to interact with the DeepSeek-OCR API server.
"""

import base64
from openai import OpenAI


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_text_only():
    """Test text-only completion."""
    print("=" * 50)
    print("Test 1: Text-only completion")
    print("=" * 50)

    client = OpenAI(
        api_key="EMPTY",  # No API key needed
        base_url="http://localhost:8000/v1"
    )

    response = client.chat.completions.create(
        model="deepseek-ocr",
        messages=[
            {"role": "system", "content": "You are a helpful OCR assistant."},
            {"role": "user", "content": "Hello! Can you help me with OCR?"}
        ],
        temperature=0.0,
        max_tokens=100
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
    print()


def test_image_url():
    """Test OCR with image URL."""
    print("=" * 50)
    print("Test 2: OCR with image URL")
    print("=" * 50)

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    # Use a sample image URL (replace with your own)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = client.chat.completions.create(
        model="deepseek-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text can you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        temperature=0.0,
        max_tokens=500
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
    print()


def test_image_base64(image_path: str):
    """Test OCR with base64 encoded image."""
    print("=" * 50)
    print("Test 3: OCR with base64 image")
    print("=" * 50)

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    # Encode image
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="deepseek-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Convert the document to markdown."},
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
        max_tokens=2000
    )

    print(f"Response:\n{response.choices[0].message.content}")
    print(f"\nUsage: {response.usage}")
    print()


def test_streaming(image_path: str):
    """Test streaming response."""
    print("=" * 50)
    print("Test 4: Streaming response")
    print("=" * 50)

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    base64_image = encode_image(image_path)

    stream = client.chat.completions.create(
        model="deepseek-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this image."},
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
        max_tokens=2000,
        stream=True
    )

    print("Streaming response:")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


def test_document_to_markdown(image_path: str):
    """Test document to markdown conversion with grounding."""
    print("=" * 50)
    print("Test 5: Document to Markdown with grounding")
    print("=" * 50)

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    base64_image = encode_image(image_path)

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

    print(f"Response:\n{response.choices[0].message.content}")
    print(f"\nUsage: {response.usage}")
    print()


def test_list_models():
    """Test listing available models."""
    print("=" * 50)
    print("Test 6: List available models")
    print("=" * 50)

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"  - {model.id}")
    print()


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 50)
    print("DeepSeek-OCR OpenAI-Compatible API Test Client")
    print("=" * 50 + "\n")

    # Test 1: List models
    test_list_models()

    # Test 2: Text-only (may not work well for OCR model, but tests the API)
    # test_text_only()

    # Test 3-5: Image-based tests (require image path)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Using image: {image_path}\n")

        # Test with base64 image
        test_image_base64(image_path)

        # Test streaming
        test_streaming(image_path)

        # Test document to markdown
        test_document_to_markdown(image_path)
    else:
        print("Usage: python test_openai_client.py <image_path>")
        print("Example: python test_openai_client.py invoice.jpg")
        print("\nSkipping image-based tests (no image provided)")

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
