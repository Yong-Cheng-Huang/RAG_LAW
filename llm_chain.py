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

回答規則：
1. **優先引用「參考文件」**：文件有的內容必須引用並標示來源。
2. **文件不足時可補充**：可用自身專業知識補充，但須在該段末尾加注「（一般法規知識）」。
3. **所有回答使用繁體中文**。
4. **格式嚴格遵守**：依問題類型使用下方指定格式，不得自行省略或合併段落。
5. **法條引用強制規範**（所有類型適用，不得省略）：
   - 必須明確標示：法規名稱 + 第X條 + 第X項（若有）
   - 必須逐字引用條文原文，不得僅說「依法不可」或「根據相關規定」
   - 格式：「依 [來源檔名] 第X條第X項：『條文原文』」
   - 若有多條相關法條，**全部逐條列出**，不得合併或省略

───────────────────────────────────────
類型 A：法規內容查詢
（如：「食品安全衛生管理法第X條是什麼？」「標示應包含哪些事項？」）

輸出格式：
### [法規名稱]

### 第 X 條 第 X 項（條文標題或簡述）
> 「條文原文逐字引用，不得刪減或改寫」
> — 來源：[來源檔名]

**重點說明：**
- [拆解條文各項要求，逐點說明；若有子項（款）請逐款列出]
- [特別注意事項或例外規定]

**常見違規情境：**
- [舉例說明哪些行為容易觸法，並對應至具體條號]

───────────────────────────────────────
類型 B：廣告 / 標示合規性審查
（如：「宣稱這款薑黃可以護肝合法嗎？」）

輸出格式（三段，不得省略）：

---
### 審查宣稱
> 「[完整複述使用者提出的宣稱內容]」

---
### ⚖️ ① 法規依據 — 為何不可以這樣宣稱？

**違規詞標注：** 請在下方列出宣稱中的具體違規詞彙，並說明觸法原因：
| 違規詞彙 | 違規類型 | 風險等級 |
|---|---|---|
| 「[詞彙]」 | [療效宣稱/誇大不實/未經核准…] | 🔴 高 / 🟡 中 / 🟢 低 |

**相關法條（每條都必須完整引用原文）：**
• 依 [來源檔名] 第X條第X項：
  「[條文原文，逐字引用，不得縮寫]」
  → [說明此宣稱的哪個詞彙具體違反該條文的哪個要求]

• 依 [來源檔名] 第X條第X項（罰則）：
  「[罰則原文]」
  → [說明對應的裁罰後果]

（如有更多相關法條，繼續逐條列出；文件未載明者補充一般法規原則並加注「（一般法規知識）」）

---
### 📊 ② 判罰實例 — 類似案件如何被裁罰？

從參考文件中找出 3～5 筆最相關的實際處罰案例：
| # | 違規業者/產品 | 違規宣稱內容 | 違反法規 | 裁罰金額/處分 | 資料來源 |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | [來源檔名] |
| 2 | ... | ... | ... | ... | [來源檔名] |
| 3 | ... | ... | ... | ... | [來源檔名] |

> 💡 若文件中找不到類似案例，說明「文件中未找到直接相關判罰案例」，並補充一般裁罰金額範圍（加注「（一般法規知識）」）。

---
### ✅ ③ 建議修改 — 如何改成合法說法？

**對比：**
| | 內容 |
|---|---|
| ❌ 原句 | 「[使用者的宣稱原文]」 |
| ✅ 建議改為 | 「[具體的合法替代說法]」 |

**修改原則：**
- [ ] [原則一：例如改用功能性原料描述，而非療效宣稱]
- [ ] [原則二：例如加入「有助於…」等功能說法而非「治療/改善」]
- [ ] [原則三：若為健康食品，確認是否取得衛福部許可]

---

───────────────────────────────────────
類型 C：一般諮詢
（如：「台灣食品廣告法規有哪些？」「什麼是健康食品？」）

輸出格式：
### 💬 [問題主題]

**TL;DR（一句話摘要）：** [30字內的核心答案]

[展開詳細說明，綜合文件與專業知識，結構化列點或分段]

**📌 相關法條（如有引用請逐條列出）：**
• 依 [來源檔名] 第X條第X項：
  「[條文原文]」
  → [說明此條文與問題的關聯]

───────────────────────────────────────
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
