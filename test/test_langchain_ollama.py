import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage
import os

# ================= 配置区域 =================
# A 电脑的局域网 IP 地址
A_COMPUTER_IP = "192.168.31.98"
# 模型名称 (必须与 A 电脑上一致)
MODEL_NAME = "deepseek-ocr"
# 图片路径
IMAGE_PATH = r"E:\project\PersonalAi\test\img1.png"


# ===========================================

def image_to_base64(image_path):
    """
    将图片转换为 Base64 字符串
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ 读取图片失败: {e}")
        return None


def run_ocr():
    # 1. 初始化 ChatOpenAI 客户端
    # 关键点：
    # 1. api_key 可以随便填，Ollama 不校验
    # 2. base_url 必须指向 A 电脑 IP，且必须加上 /v1 后缀
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=f"http://{A_COMPUTER_IP}:11434/v1",  # langchain-openai：必须以 /v1 结尾。
        api_key="ollama"  # langchain-openai 库强制要求传入 api_key 参数，否则会报错。可以填入任意字符串（如 "ollama" 或 "123"），它不会被验证
    )

    # 2. 处理图片
    print(f"📷 正在读取图片: {IMAGE_PATH} ...")
    base64_image = image_to_base64(IMAGE_PATH)

    if not base64_image:
        return

    # 3. 构建消息
    # 使用 HumanMessage 并传入 content 列表，包含文本和图片 URL
    message = HumanMessage(
        content=[
            {"type": "text", "text": "请识别图片中的所有文字内容，保持原有格式。"},
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64_image}"
            }
        ]
    )

    # 4. 发送请求
    print(f"🚀 正在向 A 电脑 ({A_COMPUTER_IP}) 发送 OCR 请求...")
    try:
        response = llm.invoke([message])
        print("\n--- ✅ 识别结果 ---")
        print(response.content)
        print("----------------")
    except Exception as e:
        print(f"\n❌ 调用失败: {e}")
        print("💡 提示：请检查 A 电脑 IP 是否正确，防火墙是否开放 11434 端口。")


if __name__ == "__main__":
    run_ocr()