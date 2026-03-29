# coding: utf-8
# 日志分析助手，支持多轮对话，用户可以连续粘贴多段日志，AI能结合上下文分析，并且输出标准的JSON报告

import os
from dotenv import load_dotenv

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from langchain_openai import ChatOpenAI


# 创建会话存储（多用户隔离）
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 使用pydantic定义输出字段
class ResumeInfo(BaseModel):
    error_type: str = Field(description="错误类型")
    solution: str = Field(description="解决方案")
    fix_cmd: str = Field(description="用于修复错误的命令")
    serverity: str = Field(description="风险等级评估，high|medium|low")

# 初始化Pydantic输出解析器
parser = PydanticOutputParser(pydantic_object=ResumeInfo)

# 提示词模板，并创建包含历史占位符的提示模板
prompt_template = """
你是一位资深运维专家，请根据用户输入的日志片段，指出日志中错误原因并给出修复命令。

日志片段：
{log_text}

请严格按照以下格式输出：
{{
    "error_type": "错误类型",
    "solution": "解决方案",
    "fix_cmd": "用于修复错误的命令",
    "serverity": "风险等级评估，high|medium|low"
}}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    MessagesPlaceholder(variable_name="history"),  # 关键：历史消息占位符
    ("human", "{log_text}")
])

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
) # temperature=0 让输出更稳定

# 创建基础链
chain = prompt | llm | parser | (lambda x: x.model_dump_json(indent=2))

# 包装成带记忆的链
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="log_text",      # 输入消息的键名
    history_messages_key="history"   # 历史消息的键名
)

# 使用示例
session_id = "user_123"

# 第一轮对话
log_text1 = '请分析这条日志：psql: error: connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused'
response1 = chain_with_history.invoke(
    {"log_text": log_text1},
    config={"configurable": {"session_id": session_id}}
)
print(response1)
print("======="*5)

# 第一轮对话
log_text2 = '服务启动后，日志里又出现了新的报错：OutOfMemoryError: Java heap space'
response2 = chain_with_history.invoke(
    {"log_text": log_text2},
    config={"configurable": {"session_id": session_id}}
)
print(response2)
print("======="*5)
# 第一轮对话
log_text3 = '修改配置重启后，现在日志里全是 INFO: Application started successfully，还需要关注什么吗？'
response3 = chain_with_history.invoke(
    {"log_text": log_text3},
    config={"configurable": {"session_id": session_id}}
)
print(response3)
print("======="*5)
log_text4 = '第一个问题的分析结果，请重新输出一遍'
response4 = chain_with_history.invoke(
    {"log_text": log_text4},
    config={"configurable": {"session_id": session_id}}
)
print(response4)
print("======="*5)

# 10. 辅助函数：查看会话历史
def view_session_history(new_session_id: str):
    """查看指定会话的历史记录"""
    if new_session_id in store:
        history = store[new_session_id]
        print(f"会话 {new_session_id} 的历史记录:")
        for i, msg in enumerate(history.messages):
            msg_type = "用户" if msg.type == "human" else "助手"
            print(f"{i + 1}. [{msg_type}]: {msg.content[:100]}...")  # 只显示前100个字符
    else:
        print(f"会话 {session_id} 不存在")

# 展示会话历史
print("\n=== 查看会话历史 ===")
view_session_history(session_id)

