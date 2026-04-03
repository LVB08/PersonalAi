# coding: utf-8

import chromadb
from chromadb.config import Settings
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

load_dotenv()
text_model_path = r"/Users/htf/PycharmProjects/ysEnvPro/llm_model_file/BAAI/bge-large-zh-v1___5"
local_text_model = SentenceTransformer(text_model_path)


class MyVectorDBConnector:
    def __init__(self, collection_name):
        # 创建一个客户端
        # chroma_client = chromadb.Client(Settings(allow_reset=True)  # allow_reset: 允许调用参数清空数据库; 生产环境建议设置False;
        chroma_client = chromadb.PersistentClient(
            path=r"/Users/htf/PycharmProjects/ysEnvPro/PersonalAi/rag_trial/bili/chroma_data",
            settings=chromadb.Settings(allow_reset=True))

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)

    # def get_embeddings_qwen(self, texts, model="text-embedding-v3"):
    #     """封装 qwen 的 Embedding 模型接口"""
    #     # print('texts', texts)
    #     qwen_client = OpenAI(api_key=os.getenv("api_key"), base_url=os.getenv("base_url"))
    #     data = qwen_client.embeddings.create(input=texts, model=model).data
    #     return [x.embedding for x in data]

    def get_embeddings_local(self, texts_list: list):
        """加载本地文本向量模型，获取embeddings"""
        embeddings = local_text_model.encode(texts_list)
        return embeddings

    def add_documents(self, instructions, outputs):
        """向 collection 中添加文档与向量"""
        # get_embeddings(instructions)
        # 将数据向量化
        embeddings = self.get_embeddings_local(instructions)

        # 把向量化的数据和原文存入向量数据库
        self.collection.add(
            embeddings=embeddings,  # 每个文档的向量
            documents=outputs,  # 文档的原文
            ids=[f"id{i}" for i in range(len(outputs))]  # 每个文档的 id
        )
        # print(self.collection.count())

    def search(self, query):
        """检索向量数据库"""
        # 把我们查询的问题向量化, 在chroma当中进行查询
        results = self.collection.query(
            query_embeddings=self.get_embeddings_local([query]),
            n_results=2,
        )
        return results


if __name__ == '__main__':
    # 读取文件
    with open('train_zh.json', 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    # print(len(data_list))
    # print(data_list[:3])

    # 获取前10条的问题和输出
    instructions = [entry['question'] for entry in data_list[0:10]]
    outputs = [entry['answer'] for entry in data_list[0:10]]

    # 创建一个向量数据库对象
    vector_db = MyVectorDBConnector("demo")

    # 向向量数据库中添加文档
    vector_db.add_documents(instructions, outputs)
    # print(vector_db.collection.get(include=["documents", "embeddings"]))
    # print(vector_db.collection.get()
    # user_query = "白癜风"
    user_query = "感冒怎么办？"
    results = vector_db.search(user_query)
    print(results)

    print("==============="*5)
    for para in results['documents'][0]:
        print(para + "\n")
