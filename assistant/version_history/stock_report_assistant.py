# coding: utf-8
"""
@Time: 2026-04-27
@Author: 怀风・Halcyon
@Description:
"""

import os

from tools.get_stock_price import get_stock_price
from tools.calculate_tool import calculate_growth_rate
from tools.rag_tool import search_vector_db
from prompt_template import cus_prompt, system_prompt

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_agent


class StockReportAssistant:

    def __init__(self):
        self.store = {}
        self.prompt = cus_prompt
        self.tools = [calculate_growth_rate, search_vector_db, get_stock_price]  #
        self.llm = self.start_load_model()
        self.agent = self.create_agent()
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

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """根据session_id获取会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def create_agent(self):
        """创建agent"""
        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        agent_with_memory = RunnableWithMessageHistory(
            agent,
            self.get_session_history,
            input_messages_key="input",
            # history_messages_key="chat_history",
        )

        return agent_with_memory

    def run(self):
        """运行主函数"""
        while True:
            try:
                user_input = input("\n[用户] 您想咨询的问题？: ").strip()  # 获取用户输入
                if user_input.lower() in ['exit', 'quit', 'q', '退出']:  # 退出逻辑
                    print("再见！")
                    break
                response = self.agent.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "new_seesion"}}
                )
                print(f"\n[助手] 查询结果: {response}")

            except KeyboardInterrupt:
                print("\n\n检测到退出指令，再见！")
                break
            except Exception as e:
                print(f"\n[错误] 发生未知错误: {e}")


if __name__ == "__main__":
    StockReportAssistant()


