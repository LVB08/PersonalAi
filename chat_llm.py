# coding: utf-8
# 请求大模型函数

import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def chat_deepseek_sample(content_text=None):
    """请求deepseek, openai方式"""
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")

    if content_text:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": content_text},
            ],
            temperature=1.3,  # 通用对话设置成1.3
            stream=False
        )
        # return response.choices[0].message.content
        return response
    else:
        return "传入文本不能为空！"


def chat_deepseek_langchain(messages):
    """请求deepseek，langchain方式"""
    llm = ChatOpenAI(
        model="deepseek-chat",  # deepseek-chat对应DeepSeek-V3，deepseek-reasoner对应R1
        temperature=1.3,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )

    response = llm.invoke(messages)
    return response.content


def chat_deepseek_langchain_prompt_text(paramas_list, **kwargs):
    """请求deepseek，langchain方式"""
    llm = ChatOpenAI(
        model="deepseek-chat",  # deepseek-chat对应DeepSeek-V3，deepseek-reasoner对应R1
        temperature=1.3,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )

    # 2. 创建ChatPromptTemplate模板
    prompt_template = ChatPromptTemplate.from_messages(paramas_list)

    # 3. 使用模板生成消息
    messages = prompt_template.format_messages(**kwargs)

    response = llm.invoke(messages)
    return response.content


if __name__ == '__main__':
    paramas_list = [
    ("system", "你是一位{role}，请用{language}回答问题。"),
    ("user", "{input}")]
    res = chat_deepseek_langchain_prompt_text(paramas_list, role="专业的编程顾问", language="中文", input="什么是LangChain的LCEL语法？")
    print(res)

    # text = "今天是几月几号?"
    # res = chat_deepseek_sample(text)
    # print(res)
    # print(res.choices[0].message.content)