# coding: utf-8

# # 模型下载
# from modelscope import snapshot_download
# # pip install modelscope
# model_dir = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir=r"E:\LLM\local_model")



# pip install sentence-transformers
# 用来加载模型,并使用模型生成向量
from sentence_transformers import SentenceTransformer

model_path = r"E:\LLM\local_model\BAAI\bge-large-zh-v1___5"  # 替换为实际路径
model = SentenceTransformer(model_path)

sentences = [
    "苹果",  # "梨", "汽车"
]

# 生成向量（默认返回numpy数组）
embeddings = model.encode(sentences)
print(embeddings)
print("----------")
print(embeddings.shape)  # 输出: (2, 1024)  （维度取决于模型）
print("----------")
print(len(embeddings))
print("----------")
print(len(embeddings[0]))
print("----------")

for i in embeddings:
    for j in i:
        print(j, end="||")

