# coding: utf-8
import chromadb

# =========创建连接=========
# client = chromadb.Client()
# 创建 client 时，设置数据持久化路径, 默认存储在内存,程序运行完时会丢失
client = chromadb.PersistentClient(path=r"/Users/htf/PycharmProjects/ysEnvPro/PersonalAi/rag_trial/bili/chroma_data")
# 存在这个集合就返回，不存在就创建
collection = client.get_or_create_collection(name="test")


# =========增删改查=========
# # 添加数据
# collection.add(
#     documents=["Article by john", "Article by Jack", "Article by Jill"],  # 文本内容列表，每个元素是一段文本（如文章、句子等）
#     embeddings=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 嵌入向量列表，每个元素是一个与 documents 对应的向量表示
#     ids=["1", "2", "3"]  # 自定义 ID 列表，用于唯一标识每条记录
# )

# 查询数据
# ids、where_document相当于两个查询条件，可以选填，也可全填
aa = collection.get(
    ids=["3"],
    # where_document={"$contains": "john"},  # 表示文本内容中包含 "john" 的文档
    include=["embeddings"]  # 包含嵌入向量, 出于性能考虑，默认不返回嵌入向量
)
print(aa)

# # 删除数据
# collection.delete(
#     ids=["1"]
# )
# print(collection.get(include=["embeddings"]))

# # 修改数据
# collection.update(
#     documents=["Article by john", "Article by Jack", "Article by Jill"],
#     embeddings=[[10, 2, 3], [40, 5, 6], [70, 8, 9]],
#     ids=["1", "2", "3"])
# print(collection.get(include=["embeddings"]))
