# coding: utf-8
"""
1. 构建应用程序时，temperature默认的值0.7，为什么要设置成0重做一遍 ？？【为了让输出的随机性降低一点】

"""

from langchain_core.prompts import ChatPromptTemplate

## from_messages
# 定义模板
prompt_template_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业的翻译助手，将用户的话翻译成{target_language}。"),
        ("human", "{text}")
    ]
)
messages1 = prompt_template_1.invoke({"target_language": "法语", "text": "你好，世界!"})
print(messages1)


## from_template
# 定义模板字符串
template_str = "告诉我关于 {topic} 的三个有趣事实。"
prompt_template_2 = ChatPromptTemplate.from_template(template_str)
messages = prompt_template_2.invoke({"topic": "量子力学"})

# 查看结果
# print(messages)
# 输出: [HumanMessage(content='告诉我关于量子力学的三个有趣事实。')]

# print(prompt_template_2.messages[0].prompt)  # 视频用法，可以提取prompt模板的不同变量
# print(prompt_template_2.messages[0].prompt.input_variables)  # 视频用法，可以提取prompt模板的不同变量

from chat_llm import chat_llm_deepseek
cus_mes = prompt_template_2.format_messages(topic="量子力学")
print(cus_mes)
print(cus_mes[0])
res = chat_llm_deepseek(messages_list=messages1)
print(res)