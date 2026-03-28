# coding: utf-8

import os
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 1. 定义 Pydantic 模型
class LogAnalysisReport(BaseModel):
    """日志分析报告模型"""
    error_type: str = Field(description="错误的类型或级别")
    root_cause: str = Field(description="导致错误的根本原因分析")
    fix_command: str = Field(description="建议的修复指令或代码片段")
    severity: str = Field(description="严重程度: High, Medium, Low")

# 2. 设置 LLM (请确保配置了环境变量)
llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

# 3. 定义记忆存储
store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. 构建提示词模板 (注意转义)
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个资深的运维日志分析专家。请根据用户的日志片段，结合之前的对话上下文，分析错误原因并给出修复建议。"),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "请分析以下日志信息：\n{log_input}")
# ])

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

# 5. 定义输出解析器
parser = PydanticOutputParser(pydantic_object=LogAnalysisReport)

# 6. 构建基础链 (关键修改点：在最后加上 .json() 转为字符串)
# 这样可以解决 "Expected str" 的报错
basic_chain = prompt | llm | parser | (lambda x: x.model_dump_json(indent=2))

# 7. 包装 RunnableWithMessageHistory
chain_with_memory = RunnableWithMessageHistory(
    basic_chain,
    get_session_history,
    input_messages_key="log_input",
    history_messages_key="history"
)

# 8. 命令行主程序入口
def main():
    print("=== 日志分析助手 CLI 模式 ===")
    print("输入 'exit' 或 'quit' 退出程序。")
    print("请输入日志内容进行分析...\n")

    session_id = "test-session-1" # 你可以根据需要动态生成会话ID

    while True:
        try:
            # 1. 获取用户输入
            user_input = input("\n[用户] 请输入日志片段: ").strip()

            # 2. 退出逻辑
            if user_input.lower() in ['exit', 'quit', 'q', '退出']:
                print("再见！")
                break

            # 3. 调用链 (注意：config 参数必须包含 session_id)
            response = chain_with_memory.invoke(
                {"log_input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # 4. 处理并打印结果
            # 由于我们在 basic_chain 末尾加了 .json()，这里的 response 是字符串
            # 如果你没有加 .json()，response 是 LogAnalysisReport 对象，需要用 .model_dump() 打印
            print(f"\n[助手] 分析结果: {response}")

        except KeyboardInterrupt:
            print("\n\n检测到退出指令，再见！")
            break
        except Exception as e:
            print(f"\n[错误] 发生未知错误: {e}")

if __name__ == "__main__":
    main()