# coding: utf-8
import numpy as np
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer


# # 调用在线大模型方式
# load_dotenv()
# client = OpenAI(api_key=os.getenv("api_key"), base_url=os.getenv("base_url"))


def cos_sim(a, b):
    """余弦相似度 -- 越大越相似"""
    return dot(a, b) / (norm(a) * norm(b))


def l2_distance(a, b):
    """欧式距离 -- 越小越相似"""
    x = np.asarray(a) - np.asarray(b)
    return norm(x)


# model_path = r"E:\LLM\local_model\BAAI\bge-large-zh-v1___5"  # 替换为实际路径
model_path = r"/Users/htf/PycharmProjects/ysEnvPro/llm_model_file/BAAI/bge-large-zh-v1___5"  # 替换为实际路径
local_model = SentenceTransformer(model_path)


def get_embeddings(texts, model_name="text-embedding-v3"):
    """
    生成文本的嵌入表示，结果存储在data中。
    :param texts: 是一个包含要获取嵌入表示的文本的列表，
    :param model_name: 是用来指定要使用的模型的名称
    :return: 返回了一个包含所有嵌入表示的列表
    """
    # # 调用在线大模型方式
    # data = client.embeddings.create(input=texts, model=model_name).data
    # # 返回了一个包含所有嵌入表示的列表
    # return [x.embedding for x in data]

    embeddings = local_model.encode(texts)
    return embeddings


# 且能支持跨语言
# query = "global conflicts"
query = "我国开展舱外辐射生物学暴露实验"
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = get_embeddings([query])[0]

doc_vecs = get_embeddings(documents)

print("余弦相似度:")
print(cos_sim(query_vec, query_vec))
print("=================="*5)
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))

print("=================="*5)
print("\n欧式距离:")
print(l2_distance(query_vec, query_vec))
for vec in doc_vecs:
    print(l2_distance(query_vec, vec))
