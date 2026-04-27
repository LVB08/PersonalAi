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
from prompt_template import cus_prompt, react_system_prompt

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

    def __init__(self):
        self.store = {}
        self.prompt = cus_prompt
        self.tools = [calculate_growth_rate, search_vector_db, get_stock_price]
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
            system_prompt=react_system_prompt
        )

        agent_with_memory = RunnableWithMessageHistory(
            agent,
            self.get_session_history,
            input_messages_key="input",
            # history_messages_key="chat_history",
        )

        return agent_with_memory

    def build_chain(self):
        # 创建基础链
        basechain = self.prompt | self.llm
        # 包装成带记忆的链
        chain_with_history = RunnableWithMessageHistory(
            runnable=basechain,
            get_session_history=self.get_session_history,
            input_messages_key="input",  # 输入消息的键名
            history_messages_key="chat_history"  # 历史消息的键名
        )
        return chain_with_history

    def run(self):
        """运行主函数"""
        response = self.agent.invoke(
            {"input": "华为的股价是多少"},
            config={"configurable": {"session_id": "new_seesion"}}
        )
        print(response)
        # print(f"🤖 AI: {response['messages'][-1].content}")

        response = self.chain.invoke(
            {"input": "华为的股价是多少"},
            config={"configurable": {"session_id": self.tmp_session_id}}
        )
        print(f"\n[助手] 分析结果: {response}")


if __name__ == "__main__":
    StockReportAssistant()


