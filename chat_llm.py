# coding: utf-8
# 请求大模型函数

import os
from openai import OpenAI


def chat_llm_deepseek(content_text=None, messages_list=[]):
    """请求deepseek"""
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
    elif messages_list:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages_list,
            temperature=1.3,  # 通用对话设置成1.3
            stream=False
        )

    # return response.choices[0].message.content
    return response


if __name__ == '__main__':
    text = "今天是几月几号?"
    res = chat_llm_deepseek(text)
    print(res)
    print(res.choices[0].message.content)
