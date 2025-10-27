"""
快速演示脚本 - 测试 DeepSeek-OCR OpenAI API

这是一个最简单的测试脚本，用于验证 API 是否正常工作。
"""

import sys
import base64
from pathlib import Path


def test_connection():
    """测试服务器连接"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务器连接成功")
            return True
        else:
            print(f"❌ 服务器返回错误状态: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        print("提示: 请确保服务器已启动 (python openai_api_server.py)")
        return False


def test_ocr_with_image(image_path: str):
    """使用图像测试 OCR"""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 未安装 openai 库")
        print("请运行: pip install openai")
        return

    # 检查图像文件
    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return

    print(f"\n📷 正在处理图像: {image_path}")

    # 初始化客户端
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    # 读取并编码图像
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    print("⏳ 正在进行 OCR 识别...")

    try:
        # 调用 API
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
            max_tokens=2000
        )

        print("\n" + "=" * 50)
        print("OCR 结果:")
        print("=" * 50)
        print(response.choices[0].message.content)
        print("=" * 50)
        print(f"\nToken 使用: {response.usage.total_tokens} tokens")
        print("✅ OCR 完成!")

    except Exception as e:
        print(f"❌ OCR 失败: {e}")


def test_streaming(image_path: str):
    """测试流式输出"""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 未安装 openai 库")
        return

    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return

    print(f"\n📷 正在测试流式输出: {image_path}")

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    print("⏳ 流式输出开始...\n")
    print("=" * 50)

    try:
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
            max_tokens=1000,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n" + "=" * 50)
        print("✅ 流式输出完成!")

    except Exception as e:
        print(f"\n❌ 流式输出失败: {e}")


def main():
    print("\n" + "=" * 50)
    print("DeepSeek-OCR OpenAI API - 快速测试")
    print("=" * 50 + "\n")

    # 测试服务器连接
    if not test_connection():
        print("\n请先启动服务器:")
        print("  python openai_api_server.py")
        sys.exit(1)

    # 检查是否提供了图像路径
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python demo_quick_test.py <图像路径> [--stream]")
        print("\n示例:")
        print("  python demo_quick_test.py invoice.jpg")
        print("  python demo_quick_test.py invoice.jpg --stream")
        sys.exit(0)

    image_path = sys.argv[1]

    # 检查是否使用流式输出
    if len(sys.argv) > 2 and sys.argv[2] == "--stream":
        test_streaming(image_path)
    else:
        test_ocr_with_image(image_path)


if __name__ == "__main__":
    main()
