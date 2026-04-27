# coding: utf-8
"""
@Time: 2026-04-27
@Author: 怀风・Halcyon
@Description:
"""

import os
import json

from tools.get_stock_price import get_stock_price
from tools.calculate_tool import calculate_growth_rate
from tools.rag_tool import search_vector_db
from prompt_template import cus_prompt, system_prompt_react

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_agent


class StockReportAssistant:
    # 工具注册表，让模型知道有哪些工具可用
    tools_registry = {
        "calculate_growth_rate": calculate_growth_rate,
        "search_vector_db": search_vector_db,
        "get_stock_price": get_stock_price
    }
    max_try_num = 5

    def __init__(self, query):
        self.query = query
        self.llm = self.start_load_model()
        self.run()

    def start_load_model(self):
        """初始化模型"""
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        )  # temperature=0 让输出更稳定
        return llm

    def run(self):
        """运行主函数"""
        messages = [
            {"role": "system", "content": system_prompt_react},
            {"role": "user", "content": self.query}
        ]
        print(f"======用户提问: {self.query}======")
        for i in range(self.max_try_num):
            print(f"---------------第{i}轮开始---------------")
            ai_mes = self.llm.invoke(messages).content
            messages.append({"role": "assistant", "content": ai_mes})
            print(f"🤖 AI: {ai_mes}")

            if "Action:" in ai_mes and "Action Input:" in ai_mes:
                action_line = ai_mes.split("Action:")[1].split("Action Input:")[0].strip()
                input_line = ai_mes.split("Action Input:")[1].strip()
                tool_name = action_line.strip()
                try:
                    tool_args = json.loads(input_line)
                except:
                    tool_args = {}

                if tool_name in self.tools_registry:
                    arg_val = list(tool_args.values())[0] if tool_args else ""
                    observation = self.tools_registry[tool_name].invoke(arg_val)
                    print(f"🛠️ 工具执行结果: {observation}")

                    # 将观察结果加入对话历史，形成闭环
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    messages.append({"role": "user", "content": "Observation: 工具不存在"})
            else:
                # 如果没有 Action，说明任务完成
                print(f"✅ 最终回答: {ai_mes}")
                break
            print(f"---------------第{i}轮结束---------------")


if __name__ == "__main__":
    # query = "小米的财务情况？"
    # StockReportAssistant(query)

    print("=== 助手 CLI 模式 ===")
    print("输入 'exit' 或 'quit' 退出程序。")
    while True:
        try:
            user_input = input("\n[用户] 请提问: ").strip()  # 获取用户输入
            if user_input.lower() in ['exit', 'quit', 'q', '退出']:  # 退出逻辑
                print("再见！")
                break
            StockReportAssistant(user_input)
        except KeyboardInterrupt:
            print("\n\n检测到退出指令，再见！")
            break
        except Exception as e:
            print(f"\n[错误] 发生未知错误: {e}")

