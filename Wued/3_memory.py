# coding: utf-8
# 记忆存储，多轮对话
# 现已不再支持，memory类型：ConversationBufferMemory、ConversationBufferWindowMemory、ConversationSummaryMemory、ConversationSummaryBufferMemory
# 1.2.x版本langchain建议使用 RunnableWithMessageHistory
# 扩展：使用 Redis 存储历史

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

# 1. 创建会话存储（多用户隔离）
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 2. 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
) # temperature=0 让输出更稳定

# 3. 创建包含历史占位符的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手。"),
    MessagesPlaceholder(variable_name="history"),  # 关键：历史消息占位符
    ("human", "{input}")
])

# 4. 创建基础链
chain = prompt | llm

# 5. 包装成带记忆的链
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",      # 输入消息的键名
    history_messages_key="history"   # 历史消息的键名
)

# 6. 使用示例
session_id = "user_123"

# 第一轮对话
response1 = chain_with_history.invoke(
    {"input": "你好，我叫小明"},
    config={"configurable": {"session_id": session_id}}
)
print(response1.content)
print("======="*5)
# 第二轮对话（自动包含历史）
response2 = chain_with_history.invoke(
    {"input": "美国成立多少年了？直接给答案即可。"},
    config={"configurable": {"session_id": session_id}}
)
print(response2.content)  # 会记得名字是"小明"
print("======="*5)
# 第三轮对话（自动包含历史）
response3 = chain_with_history.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": session_id}}
)
print(response3.content)  # 会记得名字是"小明"

print("++++++++++++"*5)
# 检查内部存储的历史
print("\n--- 存储的内部历史 ---")
for message in store[session_id].messages:
    print(type(message))
    print(message)
    print("++++++++++++" * 5)