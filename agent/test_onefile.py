import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# --- 1. 模拟工具定义 (保持不变) ---

@tool
def get_stock_price(symbol: str) -> str:
    """根据股票代码获取当前股价。"""
    mock_prices = {"600519": "1850.50", "000001": "10.25"}
    price = mock_prices.get(symbol, "123.45")
    return f"股票 {symbol} 的当前价格为 {price} 元。"


@tool
def calculate_growth_rate(old_value: float, new_value: float) -> float:
    """计算增长率。"""
    return round(((new_value - old_value) / old_value) * 100, 2)


# --- 2. 模拟 RAG 组件 (保持不变) ---

def setup_rag_tool():
    """设置并返回一个用于检索研报的 RAG 工具。"""
    mock_reports = [
        Document(page_content="A公司 (600519) 2025年年报显示，其营收同比增长20%，净利润增长25%。"),
        Document(page_content="B公司 (000001) 2025年年报显示，其营收同比增长15%，净利润增长10%。"),
        Document(page_content="行业分析指出，高端白酒市场未来三年将保持稳定增长。"),
    ]

    embeddings = OpenAIEmbeddings()
    # 为简化示例，假设 ChromaDB 数据已持久化，这里直接加载
    # 首次运行时，请确保有 Chroma.from_documents(...) 的代码来创建数据库
    try:
        vector_db = Chroma(collection_name="research_reports", embedding_function=embeddings)
        if vector_db._collection.count() == 0:  # 如果数据库为空，则初始化
            vector_db = Chroma.from_documents(documents=mock_reports, collection_name="research_reports",
                                              embedding=embeddings)
    except Exception:
        # 如果加载失败（例如首次运行），则创建并填充
        vector_db = Chroma.from_documents(documents=mock_reports, collection_name="research_reports",
                                          embedding=embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    @tool
    def search_research_reports(query: str) -> str:
        """在本地研报库中搜索相关信息。"""
        relevant_docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in relevant_docs])

    return search_research_reports


# --- 3. 构建带记忆的 Agent ---

def main():
    # 3.1 初始化 LLM 和工具
    llm = ChatOpenAI(model="deepseek-chat", temperature=0, api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    tools = [get_stock_price, calculate_growth_rate, setup_rag_tool()]

    # 3.2 定义系统提示词
    # 注意：这里必须包含一个 MessagesPlaceholder 来为历史消息预留位置
    system_prompt = """
    你是一个专业的智能投研助手。你的任务是利用提供的工具，准确、高效地回答用户的金融投资相关问题。
    在回答时，请遵循以下步骤：
    1.  **思考 (Thought):** 仔细分析用户的问题，判断需要哪些信息。
    2.  **行动 (Action):** 根据思考结果，选择最合适的工具来获取信息。你可以多次使用不同的工具。
    3.  **观察 (Observation):** 查看工具返回的结果。
    4.  **最终答案 (Final Answer):** 综合所有观察到的信息，生成一份结构清晰、内容详实的分析报告。

    可用工具列表：
    - `get_stock_price`: 查询实时股价。
    - `calculate_growth_rate`: 进行财务数据的计算。
    - `search_research_reports`: 检索本地的研报和文档库。

    请确保你的回答专业、客观，并引用数据来源。
    """

    # 创建带有历史占位符的 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),  # 关键：历史消息将注入此处
        ("human", "{input}")  # 用户当前的输入
    ])

    # 3.3 创建基础 Agent
    # 注意：create_agent 返回的是一个 LangGraph 的 CompiledStateGraph，它本身就是一个 Runnable
    agent = create_agent(
        model=llm,
        tools=tools,
        prompt=prompt  # 将带有历史占位符的 prompt 传入
    )

    # 3.4 设置会话历史存储
    # 这里使用内存字典模拟数据库，key 是 session_id
    # 生产环境中，可替换为 RedisChatMessageHistory 或 PostgresChatMessageHistory
    message_history_store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        """根据 session_id 获取或创建会话历史对象"""
        if session_id not in message_history_store:
            message_history_store[session_id] = ChatMessageHistory()
        return message_history_store[session_id]

    # 3.5 使用 RunnableWithMessageHistory 包装 Agent
    # 这是实现多轮对话的核心步骤
    agent_with_memory = RunnableWithMessageHistory(
        agent,  # 被包装的 Runnable (我们的 Agent)
        get_session_history,  # 获取历史记录的工厂函数
        input_messages_key="input",  # 告诉包装器，用户输入在字典的哪个 key
        history_messages_key="chat_history",  # 告诉包装器，历史记录对应 prompt 中的哪个占位符
    )

    # --- 4. 模拟多轮对话 ---

    # 使用一个固定的 session_id 来代表当前用户的会话
    config = {"configurable": {"session_id": "user_investor_001"}}

    # --- 第一轮对话 ---
    query_1 = "请对比A公司(600519)和B公司(000001)去年的营收增长情况。"
    print(f"👤 用户 (第1轮): {query_1}\n")
    print("🤖 助手正在思考并执行任务...")
    response_1 = agent_with_memory.invoke({"input": query_1}, config=config)
    print(f"🤖 助手回复:\n{response_1['messages'][-1].content}\n" + "-" * 60)

    # --- 第二轮对话 (依赖上下文) ---
    query_2 = "它们分别增长了多少？"
    print(f"👤 用户 (第2轮): {query_2}\n")
    print("🤖 助手正在思考并执行任务...")
    response_2 = agent_with_memory.invoke({"input": query_2}, config=config)
    print(f"🤖 助手回复:\n{response_2['messages'][-1].content}\n" + "-" * 60)

    # --- 第三轮对话 (开启新话题) ---
    query_3 = "给我写一首关于投资的诗。"
    print(f"👤 用户 (第3轮): {query_3}\n")
    print("🤖 助手正在思考并执行任务...")
    response_3 = agent_with_memory.invoke({"input": query_3}, config=config)
    print(f"🤖 助手回复:\n{response_3['messages'][-1].content}")


if __name__ == "__main__":
    main()