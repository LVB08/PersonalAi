# coding: utf-8
import json
from openai import OpenAI
import dotenv
import os
import random

dotenv.load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")

# 1. 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 2. 定义工具
def search_weather(city: str) -> str:
    """查询天气工具"""
    # 模拟返回结果，实际可对接天气API
    num = random.randint(15, 35)
    return f"{city} 今天天气晴朗，气温 {num}°C。"


def calculator(expression: str) -> str:
    """计算器工具"""
    print(f"===========传入参数: {expression}============")
    try:
        # 简单模拟计算，生产环境需注意安全
        result = eval(expression)
        return str(result)
    except:
        return "计算错误"


# 工具注册表，让模型知道有哪些工具可用
tools_registry = {
    "search_weather": search_weather,
    "calculator": calculator
}


# 3. 定义系统提示词 (System Prompt) - 这是 Agent 的"人设"
SYSTEM_PROMPT = """
你是一个智能助手。你可以使用以下工具来帮助用户：
- search_weather(city: str): 查询指定城市的天气
- calculator(expression: str): 计算数学表达式

请按照以下格式回答：
Thought: 思考你下一步需要做什么
Action: 选择工具名称 (如果不需要工具则回答 "Final Answer")
Action Input: 工具的参数 (JSON格式)
Observation: (这是系统返回的结果，你不需要生成，只需读取)

如果你已经有了最终答案，请直接回答。
"""


# 4. 核心循环 (The Loop)
def run_agent(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    print(f"👤 用户: {user_input}")

    # 限制最大循环次数，防止死循环
    for _ in range(5):
        print(f"---------------------------------第{_}轮开始-----------------------------------------------")
        # 调用 LLM
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或 deepseek-chat 等
            messages=messages,
            temperature=0
        )

        ai_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_message})

        print("$$$$$$$$$$$$$$$===AI-开始===$$$$$$$$$$$$$$$")
        print(f"🤖 AI: {ai_message}")
        print("$$$$$$$$$$$$$$$===AI-结束===$$$$$$$$$$$$$$$")

        # 解析 Action (这里做简单的字符串解析，生产环境建议用 JSON Mode)
        if "Action:" in ai_message and "Action Input:" in ai_message:
            # 提取工具名和参数
            action_line = ai_message.split("Action:")[1].split("Action Input:")[0].strip()
            input_line = ai_message.split("Action Input:")[1].strip()

            tool_name = action_line.strip()
            try:
                tool_args = json.loads(input_line)
            except:
                tool_args = {}  # 简单容错

            # 执行工具
            if tool_name in tools_registry:
                # 这里演示简单的参数传递，实际可能需要更复杂的映射
                arg_val = list(tool_args.values())[0] if tool_args else ""
                observation = tools_registry[tool_name](arg_val)
                print("===========工具执行-开始===========")
                print(f"🛠️ 工具执行结果: {observation}")
                print("===========工具执行-结束===========")

                # 将观察结果加入对话历史，形成闭环
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                messages.append({"role": "user", "content": "Observation: 工具不存在"})
        else:
            # 如果没有 Action，说明任务完成
            print("*************最终回答-开始*************")
            print(f"✅ 最终回答: {ai_message}")
            print("*************最终回答-结束*************")
            break
        print(f"---------------------------------第{_}轮结束-----------------------------------------------")


# 5. 运行测试
if __name__ == "__main__":
    # 测试 1: 需要调用计算器
    run_agent("帮我算一下 123 * 456 等于多少？")

    # # 测试 2: 需要调用天气查询
    # run_agent("北京今天天气怎么样？")








