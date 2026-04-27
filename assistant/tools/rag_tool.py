# coding: utf-8
# 文本向量模型下载
# from modelscope import snapshot_download
# # pip install modelscope
# model_dir = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir=r"E:\LLM\local_model")

import os
from dotenv import load_dotenv
from langchain.tools import tool
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import chromadb
from config import CHROMADA_DATA_PATH, TEXT_MODEL_PATH, VECTOR_DB_NAME
from sentence_transformers import SentenceTransformer

import numpy as np
from numpy import dot
from numpy.linalg import norm


load_dotenv()
local_text_model = SentenceTransformer(TEXT_MODEL_PATH)


def sliding_window_chunks(text, chunk_size, stride):
    """按照固定字符切割文档"""
    return [text[i:i + chunk_size] for i in range(0, len(text), stride)]


def extract_text_from_pdf(filename, page_numbers=None):
    """从 PDF 文件中（按指定页码）提取文字"""
    full_text = ''
    for i, page_layout in enumerate(extract_pages(filename)):
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            # 检查element是不是文本
            if isinstance(element, LTTextContainer):
                full_text += element.get_text().replace("\n", "").replace(" ", "")
    text_chunks = sliding_window_chunks(full_text, 100, 50)
    return text_chunks


class MyVectorDB:

    def __init__(self, collection_name, n_results=5):
        client = chromadb.PersistentClient(path=CHROMADA_DATA_PATH)
        self.collection = client.get_or_create_collection(name=collection_name)
        self.n_results = n_results

    def get_embeddings_local(self, texts_list: list):
        """使用本地模型进行向量化"""
        embeddings = local_text_model.encode(texts_list)
        return embeddings

    def add_documents(self, documents):
        """向 collection 中添加文档与向量"""
        self.collection.add(
            embeddings=self.get_embeddings_local(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query):
        """检索向量数据库，并使用欧式距离判断相似度"""
        query_vec = self.get_embeddings_local([query])
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=self.n_results
        )
        l2_distance_list = results.get("distances", [[]])[0]
        doc_str = "本地文档库中没有相关信息！"
        for i in range(len(l2_distance_list)):
            if l2_distance_list[i] < 0.2:
                print(f"欧式距离: {l2_distance_list[i]}; rag文档库，成功命中！")
                doc_str = results["documents"][0][i]
                break
        return doc_str
        # return results
        # return "\n\n".join([doc for doc in results["documents"][0]])



def text_vector(doc_path):
    """文档文本向量化"""
    paragraphs = extract_text_from_pdf(doc_path)
    vector_db = MyVectorDB(VECTOR_DB_NAME)
    vector_db.add_documents(paragraphs)  # 向向量数据库中添加文档


@tool
def search_vector_db(query: str) -> str:
    """用于检索核心财务指标或相关信息的工具"""
    vector_db = MyVectorDB(VECTOR_DB_NAME)
    return vector_db.search(query)


if __name__ == '__main__':
    doc_path = r"/Users/htf/PycharmProjects/ysEnvPro/PersonalAi/assistant/history_doc/new_financial_report.pdf"
    # text_vector(doc_path)
    vector_db = MyVectorDB(VECTOR_DB_NAME)
    # print(vector_db.search("小米核心财务指标？"))
    # print(search_vector_db.invoke("请对比小米公司和华为公司2024年的营收增长情况。"))
    # print(search_vector_db.invoke("小米公司2024年营收情况"))

    print(search_vector_db.invoke("华为的股价是多少"))


