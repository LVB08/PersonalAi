# coding: utf-8

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 1. 初始化DeepSeek模型（兼容OpenAI接口）
llm = ChatOpenAI(
    model="deepseek-chat",           # deepseek-chat对应DeepSeek-V3，deepseek-reasoner对应R1
    temperature=1.3,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

# 2. 创建ChatPromptTemplate模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，请用{language}回答问题。"),
    ("user", "{input}")
])

# 3. 使用模板生成消息
messages = prompt_template.format_messages(
    role="专业的编程顾问",
    language="中文",
    input="什么是LangChain的LCEL语法？"
)

# 4. 调用模型
response = llm.invoke(messages)
print(response.content)