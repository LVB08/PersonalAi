# coding: utf-8
"""
@Time: 2026-03-29
@Author: 怀风・Halcyon
@Description: 日志分析助手，支持多轮对话，用户可以连续粘贴多段日志，AI能结合上下文分析，并且输出标准的JSON报告
"""

import os

from OutputField import cus_parser
from PromptTemplate import cus_prompt

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class LogAnalazerBot():
    """日志分析助手"""

    def __init__(self):
        # self.log_text = log_text
        self.tmp_session_id = "tmp_session_id"
        self.store = {}  # 创建会话存储（多用户隔离）
        self.prompt = cus_prompt
        self.llm = self.start_load_model()
        self.parser = cus_parser
        self.chain = self.build_chain()
        self.main()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """根据session_id获取会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def start_load_model(self):
        # 初始化模型
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )  # temperature=0 让输出更稳定
        return llm

    def build_chain(self):
        # 创建基础链
        basechain = self.prompt | self.llm | self.parser | (lambda x: x.model_dump_json(indent=2))
        # 包装成带记忆的链
        chain_with_history = RunnableWithMessageHistory(
            runnable=basechain,
            get_session_history=self.get_session_history,
            input_messages_key="log_text",  # 输入消息的键名
            history_messages_key="history"  # 历史消息的键名
        )
        return chain_with_history


    def main(self):
        print("=== 日志分析助手 CLI 模式 ===")
        print("输入 'exit' 或 'quit' 退出程序。")
        print("请输入日志内容进行分析...\n")
        while True:
            try:
                user_input = input("\n[用户] 请输入日志片段: ").strip()  # 获取用户输入
                if user_input.lower() in ['exit', 'quit', 'q', '退出']:  # 退出逻辑
                    print("再见！")
                    break
                response = self.chain.invoke(
                    {"log_text": user_input},
                    config={"configurable": {"session_id": self.tmp_session_id}}
                )
                print(f"\n[助手] 分析结果: {response}")

            except KeyboardInterrupt:
                print("\n\n检测到退出指令，再见！")
                break
            except Exception as e:
                print(f"\n[错误] 发生未知错误: {e}")


if __name__ == '__main__':
    LogAnalazerBot()

"""
sample_log_text
'请分析这条日志：psql: error: connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused',
'服务启动后，日志里又出现了新的报错：OutOfMemoryError: Java heap space',
'修改配置重启后，现在日志里全是 INFO: Application started successfully，还需要关注什么吗？'
'第一个问题的分析结果，请重新输出一遍'
"""













