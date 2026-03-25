# coding: utf-8
# 输出解析
# 定义结构 -> 生成提示词 -> 模拟模型输出 -> 解析结果
"""
智能简历分析助手
1. 场景背景: 假设你正在开发一个人力资源系统，需要自动处理大量求职者的简历文本。大语言模型（LLM）擅长阅读简历，但它的输出通常是自然语言段落，程序很难直接存储到数据库中。
2. 目标：使用 ResponseSchema 定义我们想要的数据格式，并使用 StructuredOutputParser 强制 LLM 按此格式输出，最后将其转换为 Python 字典。
"""

import os
from dotenv import load_dotenv

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field # 需要安装 pydantic: pip install pydantic

load_dotenv()

# --- 1. 使用 Pydantic 模型定义响应结构 ---
# 这比之前的 ResponseSchema 列表更清晰、更强大
class ResumeInfo(BaseModel):
    candidate_name: str = Field(description="候选人的全名")
    years_of_experience: int = Field(description="候选人的总工作年限")
    skills: list[str] = Field(description="候选人掌握的关键技术技能列表")
    summary: str = Field(description="对候选人优势的一句话简短总结")

# --- 2. 初始化 Pydantic 输出解析器 ---
parser = PydanticOutputParser(pydantic_object=ResumeInfo)

# --- 3. 创建提示词模板 ---
prompt_template = """
你是一位专业的人力资源专家。请分析以下简历文本，并提取关键信息。

简历内容:
{resume_text}

请严格按照以下格式输出：
{format_instructions}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template)
]).partial(format_instructions=parser.get_format_instructions())


llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
) # temperature=0 让输出更稳定

# 构建完整的链：Prompt -> LLM -> Parser
chain = prompt | llm | parser

resume_input = "李四，毕业于清华大学，拥有8年大数据开发经验。精通Hadoop, Spark, Flink。曾主导过亿级数据平台建设。"
result = chain.invoke({"resume_text": resume_input})
print(result)