# coding: utf-8
"""
@Time: 2026-03-29
@Author: 怀风・Halcyon
@Description: 日志分析助手，提示词模板
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 提示词模板，并创建包含历史占位符的提示模板
prompt_template = """
你是一个资深的运维日志分析专家。请根据用户的日志片段，结合之前的对话上下文，分析错误原因并给出修复建议。

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
cus_prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    MessagesPlaceholder(variable_name="history"),  # 关键：历史消息占位符
    ("human", "请分析以下日志信息：\n{log_text}")
])