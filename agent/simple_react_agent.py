import json
from openai import OpenAI

# 1. 初始化客户端 (这里以兼容 OpenAI 格式的接口为例)
client = OpenAI(api_key="YOUR_API_KEY", base_url="YOUR_BASE_URL")


# 2. 定义工具 (Tools) - 这是 Agent 的"手脚"
def search_weather(city: str) -> str:
    """查询天气工具"""
    # 模拟返回结果，实际可对接天气API
    return f"{city} 今天天气晴朗，气温 25°C。"


def calculator(expression: str) -> str:
    """计算器工具"""
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
        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 或 deepseek-chat 等
            messages=messages,
            temperature=0
        )

        ai_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_message})

        print(f"🤖 AI: {ai_message}")

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
                print(f"🛠️ 工具执行结果: {observation}")

                # 将观察结果加入对话历史，形成闭环
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                messages.append({"role": "user", "content": "Observation: 工具不存在"})
        else:
            # 如果没有 Action，说明任务完成
            print(f"✅ 最终回答: {ai_message}")
            break


# 5. 运行测试
if __name__ == "__main__":
    # 测试 1: 需要调用计算器
    run_agent("帮我算一下 123 * 456 等于多少？")

    # 测试 2: 需要调用天气查询
    run_agent("北京今天天气怎么样？")