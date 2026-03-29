# coding: utf-8
"""
日志分析器，无记忆功能
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 定义响应结构
class LogAnalyzeInfo(BaseModel):
    error_type: str = Field(description="错误类型")
    solution: str = Field(description="解决方案")
    severity: str = Field(description="风险等级，high/medium/low")

# 2. 初始化解析器
parser = PydanticOutputParser(pydantic_object=LogAnalyzeInfo)
format_instructions = parser.get_format_instructions()

# 3. 构造提示词模板
prompt_template = """
你是一个资深运维专家。请分析以下日志片段，指出错误原因并给出修复命令，同时评估风险等级。

日志片段：
{log_text}

请严格按照以下格式输出：
{format_instructions}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template)
]).partial(format_instructions=format_instructions)

# 4.构造llm 和 完整调用链 Prompt -> LLM -> Parser
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
) # temperature=0 让输出更稳定
chain = prompt | llm | parser

log_text = ""
with open("log_analyzer/sample/6.log", "r", encoding="utf-8") as f:
    log_text += f.read()
# ("数据库连接失败", log_text_1),
# ("JVM内存溢出", log_text_2),
# ("磁盘空间不足", log_text_3),
# ("权限拒绝", log_text_4),
# ("API超时", log_text_5),
# ("Nginx 502错误", log_text_6)

result = chain.invoke({"log_text": log_text})
print(result)
print(type(result))
print(result.error_type)
print(result.solution)
print(result.severity)


