import ollama
import os


def remote_ocr(image_path, server_ip):
    """
    调用局域网内另一台电脑上的 Ollama 服务进行 OCR 识别
    """
    # 1. 检查本地图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误: B电脑本地路径 '{image_path}' 不存在。")
        return

    # 2. 配置 Ollama 客户端指向 A 电脑的 IP
    # 格式通常为 http://IP地址:端口
    ollama_host = os.getenv('OLLAMA_HOST')
    # os.environ["OLLAMA_HOST"] = f"http://{server_ip}:11434"

    print(f"🔗 正在连接到 Ollama 服务器: {ollama_host}")
    print(f"🚀 正在发送图片: {image_path} ...")

    try:
        # 3. 发起请求
        # 注意：这里使用的是 generate 接口，适用于处理图片输入
        response = ollama.generate(
            model='deepseek-ocr',
            prompt='请识别图片中的所有文字内容，保持原有格式。',
            images=[image_path],
            # host=ollama_host  # 关键：指定远程服务器地址
        )

        # 4. 输出结果
        if 'response' in response:
            print("\n" + "=" * 30)
            print("✅ 识别结果:")
            print(response['response'])
            print("=" * 30)
        else:
            print("⚠️ 未获取到有效识别结果。")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("💡 提示: 请检查 A 电脑是否设置了 OLLAMA_HOST=0.0.0.0 且防火墙已放行 11434 端口。")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 在这里填入 A 电脑的局域网 IP 地址
    TARGET_IP = "192.168.31.98"

    # 在这里填入 B 电脑上图片的路径
    # IMAGE_PATH = input("请输入图片路径: ").strip()
    # IMAGE_PATH = r"E:\project\PersonalAi\test\img1.png"
    IMAGE_PATH = r"E:\project\PersonalAi\test\img2.jpg"

    # 如果没输入路径，可以用默认值测试（记得修改）
    if not IMAGE_PATH:
        IMAGE_PATH = "test.png"

    remote_ocr(IMAGE_PATH, TARGET_IP)