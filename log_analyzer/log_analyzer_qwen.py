# coding: utf-8

import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# 1. 定义 Pydantic 模型 (指定输出格式)
class LogAnalysisReport(BaseModel):
    """日志分析报告模型"""
    error_type: str = Field(description="错误的类型或级别 (例如: NullPointerException, 500 Error)")
    root_cause: str = Field(description="导致错误的根本原因分析")
    fix_command: str = Field(description="建议的修复指令或代码片段")
    severity: str = Field(description="严重程度: High, Medium, Low")


# 2. 设置 LLM
# 请确保设置了环境变量 OPENAI_API_KEY
try:
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
except Exception as e:
    print(f"LLM初始化失败: {e}")

# 3. 定义记忆存储工厂函数
# 这里使用简单的内存字典存储，生产环境可替换为 Redis 或 数据库
store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 4. 构建提示词模板
# 注意：这里包含了一个 history 占位符，用于 RunnableWithMessageHistory 注入历史
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深的运维日志分析专家。请根据用户的日志片段，结合之前的对话上下文，分析错误原因并给出修复建议。

请严格按照以下JSON格式返回分析结果：
{{
    "error_type": "错误类型",
    "root_cause": "根本原因分析",
    "fix_command": "修复指令",
    "severity": "High/Medium/Low"
}}

以下是常见错误类型参考：
- ConnectionError: 连接相关错误
- FileNotFoundError: 文件未找到错误
- PermissionError: 权限错误
- SyntaxError: 语法错误
- MemoryError: 内存错误
- ValueError: 值错误
"""),
    MessagesPlaceholder(variable_name="history"),  # 这里会自动填入历史消息
    ("human", "请分析以下日志信息：\n{log_input}")
])

# # 4. 构建提示词模板
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个资深的运维日志分析专家。请根据用户的日志片段，结合之前的对话上下文，分析错误原因并给出修复建议。"
#              "请严格按照 Pydantic 模型要求的 JSON 格式输出，不要包含任何额外的 Markdown 代码块符号（如 ```json）"),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "请分析以下日志信息：\n{log_input}")
# ])

# 5. 定义输出解析器
parser = PydanticOutputParser(pydantic_object=LogAnalysisReport)

# 6. 构建基础链 (Chain)
# 将 提示词 + LLM + 解析器 串联
try:
    # 正确的链构建方式
    basic_chain = prompt | llm | parser | (lambda x: x.model_dump_json())
except Exception as e:
    print(f"链构建失败: {e}")

# 7. 包装 RunnableWithMessageHistory
# 这是实现"记忆"功能的核心
try:
    chain_with_memory = RunnableWithMessageHistory(
        basic_chain,
        get_session_history,
        input_messages_key="log_input",  # 用户输入的键
        history_messages_key="history"  # 提示词中定义的 history 占位符
    )
except Exception as e:
    print(f"记忆链初始化失败: {e}")
    chain_with_memory = None


# 8. 测试函数
def test_log_analysis():
    if chain_with_memory is None:
        print("链未成功初始化，跳过测试")
        return

    # 示例日志
    # sample_logs = [
    #     "2023-10-15 10:30:15 ERROR: Connection to database failed: Connection refused",
    #     "2023-10-15 10:31:22 WARNING: High memory usage detected (85%)",
    #     "2023-10-15 10:32:10 INFO: Application started successfully"
    # ]
    sample_logs = [
        '请分析这条日志：psql: error: connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused',
        '服务启动后，日志里又出现了新的报错：OutOfMemoryError: Java heap space',
        '修改配置重启后，现在日志里全是 INFO: Application started successfully，还需要关注什么吗？'
    ]

    print("=== 日志分析助手测试 ===")

    # 第一次调用
    print("\n第一次调用:")
    try:
        result1 = chain_with_memory.invoke(
            {"log_input": sample_logs[0]},
            config={"configurable": {"session_id": "test-session-1"}}
        )
        print(f"分析结果: {result1}")
    except Exception as e:
        print(f"第一次调用失败: {e}")

    # 第二次调用，测试记忆功能
    print("\n第二次调用（测试记忆功能）:")
    try:
        result2 = chain_with_memory.invoke(
            {"log_input": sample_logs[1]},
            config={"configurable": {"session_id": "test-session-1"}}
        )
        print(f"分析结果: {result2}")
    except Exception as e:
        print(f"第二次调用失败: {e}")

    # 第三次调用，继续测试
    print("\n第三次调用（继续测试记忆功能）:")
    try:
        result3 = chain_with_memory.invoke(
            {"log_input": sample_logs[2]},
            config={"configurable": {"session_id": "test-session-1"}}
        )
        print(f"分析结果: {result3}")
    except Exception as e:
        print(f"第三次调用失败: {e}")


# 9. 辅助函数：清除会话历史
def clear_session(session_id: str):
    """清除指定会话的历史记录"""
    if session_id in store:
        del store[session_id]
        print(f"会话 {session_id} 已清除")


# 10. 辅助函数：查看会话历史
def view_session_history(session_id: str):
    """查看指定会话的历史记录"""
    if session_id in store:
        history = store[session_id]
        print(f"会话 {session_id} 的历史记录:")
        for i, msg in enumerate(history.messages):
            msg_type = "用户" if msg.type == "human" else "助手"
            print(f"{i + 1}. [{msg_type}]: {msg.content[:100]}...")  # 只显示前100个字符
    else:
        print(f"会话 {session_id} 不存在")


if __name__ == "__main__":
    # 运行测试
    test_log_analysis()

    # 展示会话历史
    print("\n=== 查看会话历史 ===")
    view_session_history("test-session-1")