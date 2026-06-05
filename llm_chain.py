"""
LLM 鏈模組 — 建構 Hybrid Search (BM25 + Vector) + MultiQueryRetriever + RAG 生成鏈。
"""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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

任務流程

第一步：判斷問題類型。必須先分類，再套用對應格式。

類型 B：廣告或標示合規性審查
若問題符合以下任一條件，使用類型 B 格式：
- 包含具體宣稱、廣告或標示文字。
- 詢問某宣稱、廣告或標示是否合法、是否可以使用。
- 詢問某廣告行為違反哪些規定或是否違規。
- 包含「宣稱」「廣告」「標示」以及「合法」「違法」「違規」「可以嗎」「哪些規定」等語意。

類型 A：法規內容查詢
若問題符合以下任一條件，使用類型 A 格式：
- 直接詢問某條法律條文的內容。
- 詢問法規中某項目的具體要求。
- 不含具體宣稱文字，純粹查詢法條本身。

類型 C：一般諮詢
若問題不屬於類型 A 或類型 B，使用類型 C 格式。

回答規則

1. 優先引用參考文件。參考文件有的內容必須引用並標示來源。
2. 文件不足時可以補充一般法規知識，但必須在該段末尾標示「一般法規知識」。
3. 所有回答使用繁體中文。
4. 必須依問題類型使用指定格式，不得省略必要段落。
5. 法條引用必須明確標示法規名稱、第 X 條、第 X 項。若條文有款或目，也要一併標示。
6. 引用法條時必須逐字引用條文原文，不得只說「依法不可」或「根據相關規定」。
7. 若有多條相關法條，必須逐條列出，不得合併或省略。
8. 表格只放適合快速掃描的短內容。每個儲存格盡量控制在 30 字內，不在表格內放完整法條或長段說明。
9. 若內容過長，表格中使用摘要，並在表格下方用條列補充細節。
10. Markdown 表格欄位必須固定，不要新增或刪除欄位。沒有資料時填「文件未載明」。

類型 A 輸出格式

法規名稱

第 X 條第 X 項：條文標題或簡述
條文原文：
「條文原文逐字引用，不得刪減或改寫」
來源：[來源檔名]

重點說明：
- 拆解條文各項要求。若有子項或款，請逐項列出。
- 說明特別注意事項或例外規定。

常見違規情境：
- 舉例說明哪些行為容易觸法，並對應至具體條號。

類型 B 輸出格式

審查宣稱

「完整複述使用者提出的宣稱內容」

一、法規依據

違規詞標注：
| 詞彙 | 違規類型 | 主要理由 | 風險 |
|---|---|---|---|
| [詞彙] | [療效宣稱等] | [簡短說明] | 高、中或低 |

相關法條：
- 依 [來源檔名] 第 X 條第 X 項：
  「條文原文，逐字引用，不得縮寫」
  說明：說明此宣稱的哪個詞彙具體違反該條文的哪個要求。

- 依 [來源檔名] 第 X 條第 X 項，罰則：
  「罰則原文」
  說明：說明對應的裁罰後果。

若有更多相關法條，繼續逐條列出。文件未載明者，可補充一般法規原則並標示「一般法規知識」。

二、判罰實例

從參考文件中找出 3 到 5 筆最相關的實際處罰案例：
| 編號 | 裁決書發文日期 | 業者或產品 | 宣稱摘要 | 違反法規 | 罰鍰（元） | 來源 |
|---|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... | [來源檔名] |
| 2 | ... | ... | ... | ... | ... | [來源檔名] |
| 3 | ... | ... | ... | ... | ... | [來源檔名] |

罰鍰金額必須只使用參考文件中「抽取罰鍰金額」或原文明確記載的數字。
不得自行推估、換算不確定金額或引用其他列的金額。
若案例有相關性但沒有明確罰鍰金額，該列金額填「文件未載明」。
裁決書發文日期必須只使用參考文件中的「裁決書發文日期」或「發文日期」，沒有資料時填「文件未載明」。
若文件中找不到類似案例，說明「文件中未找到直接相關判罰案例」，並補充一般裁罰金額範圍，且標示「一般法規知識」。

三、建議修改

對比：
| 項目 | 文字 | 說明 |
|---|---|---|
| 原句 | [使用者的宣稱原文] | [主要風險] |
| 建議改為 | [具體的合法替代說法] | [調整原因] |

修改原則：
- 原則一：例如改用功能性原料描述，而非療效宣稱。
- 原則二：例如加入合理限定語，避免治療、改善疾病等醫療效果表述。
- 原則三：若涉及健康食品功效，確認是否取得主管機關許可。

類型 C 輸出格式

問題主題

一句話摘要：[30 字內的核心答案]

詳細說明：
[展開詳細說明，綜合文件與專業知識，結構化列點或分段]

相關法條：
- 依 [來源檔名] 第 X 條第 X 項：
  「條文原文」
  說明：說明此條文與問題的關聯。

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
    """根據設定建立 LLM 實例。支援 ollama / openai / gemini。"""
    if settings.LLM_MODE == "openai":
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    if settings.LLM_MODE == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY or None,
            temperature=0,
        )
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
        keep_alive=300,  # 主鏈 LLM 保留 5 分鐘，避免連續問答重複 load
    )


def get_multiquery_llm():
    """
    MultiQueryRetriever 專用 LLM。
    Ollama 模式下優先使用 MULTIQUERY_MODEL（較輕量），
    避免與主鏈 LLM 同時佔用記憶體導致 OOM。
    其他模式與主鏈共用同一 model（API 無記憶體問題）。
    """
    if settings.LLM_MODE == "ollama":
        mq_model = settings.MULTIQUERY_MODEL or settings.OLLAMA_MODEL
        return ChatOllama(
            model=mq_model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            keep_alive=0,  # 跑完 MultiQuery 立刻 unload，讓主鏈 LLM 有記憶體可用
        )
    # openai / gemini 用 API，不受本機記憶體限制，直接共用主鏈 LLM
    return get_llm()


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

    # 用輕量 LLM 跑 MultiQuery，主鏈 LLM 留給生成用
    mq_llm = get_multiquery_llm()
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=mq_llm,
        prompt=MULTI_QUERY_PROMPT,
    )


def get_penalty_case_retriever():
    """
    專門針對裁罰案件建立 BM25 Retriever。
    優先使用一案一筆的 penalty_case；相容舊資料中的 table chunk。
    """
    all_docs = get_all_documents()
    penalty_docs = [
        d for d in all_docs
        if d.metadata.get("doc_type") in {"penalty_case", "table"}
    ]
    if not penalty_docs:
        return None
    k = min(settings.PENALTY_CASE_K, len(penalty_docs))
    return BM25Retriever.from_documents(penalty_docs, k=k)


def format_docs(docs):
    """將檢索到的文件格式化為純文字，保留 header context。"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "未知來源")
        doc_type = doc.metadata.get("doc_type", "document")
        meta_parts = [f"類型: {doc_type}", f"來源: {source}"]
        if "document_date" in doc.metadata:
            meta_parts.append(f"裁決書發文日期: {doc.metadata['document_date']}")
        if "penalty_amount_text" in doc.metadata:
            meta_parts.append(f"抽取罰鍰金額: {doc.metadata['penalty_amount_text']}")
        if "penalty_amount" in doc.metadata:
            meta_parts.append(f"罰鍰金額元: {doc.metadata['penalty_amount']}")
        formatted.append(f"[文件 {i} | {' | '.join(meta_parts)}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain():
    """
    建構完整的 RAG 鏈：Hybrid Retriever → Format → Prompt → LLM → Output。
    另外強制將 doc_type='table' 的 chunk（裁罰案件統計表）合入 context，
    避免判罰實例因語意不匹配而無法召回。
    """
    retriever = get_retriever()
    penalty_case_retriever = get_penalty_case_retriever()
    llm = get_llm()

    def build_context(question: str) -> str:
        # 主要結果：Hybrid Search + MultiQuery
        main_docs = retriever.invoke(question)
        seen_keys: set[tuple[str, str]] = {
            (d.metadata.get("source_file", ""), d.page_content) for d in main_docs
        }

        # 強制合入裁罰案件（用 question 直接搜尋，拉高候選數）
        penalty_docs: list = []
        if penalty_case_retriever:
            for d in penalty_case_retriever.invoke(question):
                key = (d.metadata.get("source_file", ""), d.page_content)
                if key not in seen_keys:
                    penalty_docs.append(d)
                    seen_keys.add(key)

        all_docs = main_docs + penalty_docs
        return format_docs(all_docs)

    chain = (
        {
            "context": build_context,
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
