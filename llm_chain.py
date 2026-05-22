"""
LLM 鏈模組 — 建構 Hybrid Search (BM25 + Vector) + MultiQueryRetriever + RAG 生成鏈。
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import settings
from vector_store import get_vector_store, get_all_documents


# ── System Prompt ────────────────────────────────────────
SYSTEM_PROMPT = """你是一個專業的食品法規知識庫助理，負責處理食品標示、廣告合規性審查及法規查詢。
你只能根據「參考文件」中的內容來回答問題。

嚴格規則：
1. 你的唯一知識來源是下方的「參考文件」，禁止使用訓練知識、常識或外部資訊。
2. 若答案不在文件中，必須回答：「根據您提供的文件，找不到相關資訊。」
3. 不得使用「一般而言」、「通常」等依賴外部知識的說法。
4. 回答時必須明確標示來源，格式為「依 [來源檔名] 第X條...」。
5. 所有回答必須使用繁體中文。

任務處理流程：
在回答前，請先判斷使用者的問題類型：

類型 A：法規內容查詢（如：「食品安全衛生管理法第20條是什麼？」、「標示應包含哪些事項？」）
→ 請直接引用文件內容回答，並整理成易讀的列表。格式如下：
[條文名稱] 第X條：
• 條文內容：...
• 說明：...

類型 B：廣告/標示合規性審查（如：「宣稱這款薑黃可以護肝合法嗎？」、「外包裝沒寫材質可以嗎？」）
→ 請依照以下格式進行嚴格合規分析：
---
**Q：** [複述使用者提出的宣稱內容]

**A：**
① 違規認定：[分析是否涉及違規，及違反類型]
② 相關法條：
• 依 [來源檔名] 第X條第X項：「引用原文」
③ 判罰依據：[文件中的相關罰則，若無則標示「文件中未提及具體裁罰」]
④ 建議修改：[提供具體、合法的替代說法]
---

參考文件：
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# ── MultiQueryRetriever 用的 Prompt ─────────────────────
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一個 AI 語言模型助手，專門處理法規文件查詢。你的任務是根據使用者的問題生成 3 個不同版本的搜尋查詢，
以便從法規向量資料庫中檢索到相關條文。請從不同角度改寫問題（例如：換用法律術語、拆解子問題、使用同義詞）。
請提供這些替代問題，每行一個。
原始問題：{question}""",
)


def get_llm():
    """根據設定建立 LLM 實例。"""
    if settings.LLM_MODE == "openai":
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    else:
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )


def get_retriever():
    """
    建構 Hybrid Search Retriever：
    BM25（關鍵字精確匹配）+ Vector（語意相似度）→ EnsembleRetriever
    再包一層 MultiQueryRetriever 提升召回率。
    """
    k = settings.RETRIEVER_K
    db = get_vector_store()
    vector_retriever = db.as_retriever(search_kwargs={"k": k})

    # 嘗試從 ChromaDB 取出所有文件，建立 BM25 索引
    all_docs = get_all_documents()

    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[settings.BM25_WEIGHT, 1.0 - settings.BM25_WEIGHT],
        )
    else:
        # 知識庫為空時降級回純向量搜尋
        base_retriever = vector_retriever

    llm = get_llm()
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=MULTI_QUERY_PROMPT,
    )


def format_docs(docs):
    """將檢索到的文件格式化為純文字，保留 header context。"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "未知來源")
        formatted.append(f"[文件 {i} | 來源: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain():
    """建構完整的 RAG 鏈：Hybrid Retriever → Format → Prompt → LLM → Output。"""
    retriever = get_retriever()
    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> str:
    """對知識庫提問並回傳答案。"""
    chain = build_rag_chain()
    return chain.invoke(question)
