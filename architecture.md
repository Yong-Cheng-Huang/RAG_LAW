# RAG PDF 知識庫問答系統 — 架構文件

## 專案概覽

本系統是一個針對**台灣食品法規 PDF** 設計的 RAG（Retrieval-Augmented Generation）問答系統。
使用者上傳 PDF 後，系統自動建立向量索引；提問時透過 Hybrid Search 找到最相關的法條，再由 LLM 生成答案。

---

## 系統架構總覽

```
┌─────────────────────────────────────────────────────────┐
│                    使用者 (Streamlit UI)                  │
│              上傳 PDF  ◄──────►  輸入問題                 │
└───────────┬─────────────────────────────┬───────────────┘
            │ 攝取流程                      │ 查詢流程
            ▼                             ▼
┌───────────────────┐         ┌─────────────────────────┐
│   vector_store.py  │         │      llm_chain.py        │
└───────────────────┘         └─────────────────────────┘
            │                             │
            ▼                             ▼
┌───────────────────┐         ┌─────────────────────────┐
│  ChromaDB (本地)   │ ◄──────►│  ChromaDB + BM25 索引    │
└───────────────────┘         └─────────────────────────┘
```

---

## 一、PDF 攝取流程（Ingestion Pipeline）

使用者點擊「開始攝取」後，`ingest_pdf()` 依序執行以下步驟：

```
PDF 檔案
    │
    ▼  [1] 載入
UnstructuredPDFLoader
    │  → 1 個 Document（整份 PDF 純文字 + source metadata）
    │
    ▼  [2] 決定切割策略
LEGAL_CHUNK_MODE = true？
    ├── Yes → _legal_split()     ← 目前使用
    └── No  → _default_split()  ← fallback（純字元切割）
    │
    ▼  [3] 法規語意切割 _legal_split()
    │
    │  (3a) 注入 CONTEXT 標記
    │       掃描每一行，遇到「第X章/節/條」時在前面插入：
    │       <<<CONTEXT:第一章總則 | 第1條>>>
    │
    │  (3b) RecursiveCharacterTextSplitter 切割
    │       chunk_size  = min(CHUNK_SIZE, 500)   → 500 字
    │       chunk_overlap = min(CHUNK_OVERLAP, 60) → 60 字
    │
    │       Separator 優先順序：
    │       1. \n(?=<<<CONTEXT:)          ← CONTEXT 標記前（最高優先）
    │       2. \n\n(?=第\s*...\s*條)      ← 條文邊界備援
    │       3. \n(?=第\s*...\s*項)        ← 第X項
    │       4. \n(?=[一二...]+、)          ← 條列 一、二、
    │       5. \n\n → \n → 。→ 空格 → ''  ← 逐級退化
    │
    │  (3c) 提取 CONTEXT → 可讀 Header
    │       <<<CONTEXT:第一章總則 | 第1條>>>
    │                   ↓ 轉換
    │       [第一章總則 | 第1條]（注入 page_content 開頭）
    │       chapter = "第一章總則"  → metadata
    │       article = "第1條"       → metadata
    │
    ▼  [4] 加入 source_file metadata
    │       chunk.metadata["source_file"] = "健康食品管理法.pdf"
    │
    ▼  [5] 向量化 & 存入 ChromaDB
    │       Embedding Model: bge-m3（via Ollama）
    │       持久化目錄: ./chroma_db
    │       Collection: pdf_knowledge_base
    │
    ▼  回傳 chunk 數量
```

### 切割後 Chunk 樣式範例

```
page_content:
  [第一章總則 | 第1條]
  第 1 條

  為加強健康食品之管理與監督，維護國民健康，並保障消費者之
  權益，特制定本法；本法未規定者，適用其他有關法律之規定。

metadata:
  source_file: "健康食品管理法.pdf"
  source: "/var/folders/.../tmp.pdf"
  chapter: "第一章總則"
  article: "第1條"
```

---

## 二、查詢流程（Query Pipeline）

使用者送出問題後，`ask(question)` → `build_rag_chain()` 依序執行：

```
使用者問題："食品添加物的標示要求是什麼？"
    │
    ▼  [1] MultiQueryRetriever — 擴充查詢
    │
    │  LLM 生成 3 個不同角度的子查詢，例如：
    │  → "食品添加物標示規定"
    │  → "食品添加物應標示哪些事項"
    │  → "食品添加物包裝標示法規"
    │
    ▼  [2] EnsembleRetriever — Hybrid Search
    │
    │  每個子查詢同時送入兩個 retriever（各取 K=5 筆）：
    │
    │  ┌─────────────────────────────────────────────┐
    │  │  BM25Retriever (weight=0.4)                  │
    │  │  關鍵字精確匹配，適合法條號碼、專有名詞        │
    │  │                                               │
    │  │  VectorRetriever (weight=0.6)                 │
    │  │  語意相似度，適合語意模糊、同義詞問題           │
    │  └─────────────────────────────────────────────┘
    │          ↓ 取聯集去重
    │
    ▼  [3] format_docs() — 格式化 Context
    │
    │  將檢索到的 chunks 格式化為：
    │  [文件 1 | 來源: 健康食品管理法.pdf]
    │  [第一章總則 | 第2條]
    │  ...條文內容...
    │
    │  ---
    │
    │  [文件 2 | 來源: 食品安全衛生管理法.pdf]
    │  ...
    │
    ▼  [4] RAG Prompt — 組裝最終 Prompt
    │
    │  System Prompt（嚴格規則）：
    │  - 只能使用參考文件內容
    │  - 回答必須標示來源條文
    │  - 依問題類型 A/B 格式回答
    │
    ▼  [5] LLM 生成答案
    │
    │  LLM_MODE = "ollama"  → ChatOllama（本地）
    │  LLM_MODE = "openai"  → ChatOpenAI（雲端）
    │  temperature = 0（確保確定性輸出）
    │
    ▼  [6] StrOutputParser → 純文字答案
    │
    ▼  Streamlit 顯示
```

---

## 三、模組職責

| 檔案 | 職責 |
|------|------|
| [app.py](file:///Users/frank/Desktop/RAG/app.py) | Streamlit UI、PDF 上傳、對話介面 |
| [vector_store.py](file:///Users/frank/Desktop/RAG/vector_store.py) | PDF 載入、法規語意切割、ChromaDB 讀寫 |
| [llm_chain.py](file:///Users/frank/Desktop/RAG/llm_chain.py) | Hybrid Search、MultiQueryRetriever、RAG Chain 組裝 |
| [config.py](file:///Users/frank/Desktop/RAG/config.py) | 所有參數集中管理（從 .env 讀取） |
| [inspect_chunks.py](file:///Users/frank/Desktop/RAG/inspect_chunks.py) | 除錯工具：檢視 ChromaDB 中的 chunk 內容與大小 |
| [inspect_load.py](file:///Users/frank/Desktop/RAG/inspect_load.py) | 除錯工具：檢視 PDF load 後的原始樣式（切割前） |

---

## 四、關鍵設定（config.py / .env）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LLM_MODE` | `ollama` | `ollama`（本地）或 `openai`（雲端） |
| `OLLAMA_MODEL` | `llama3` | 本地 LLM 模型名稱 |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI 模型名稱 |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding 模型（建議 bge-m3 中文效果佳） |
| `CHUNK_SIZE` | `1000` | 上限字元數（法規模式實際取 min(1000,500)=500） |
| `CHUNK_OVERLAP` | `200` | 重疊字元數（法規模式實際取 min(200,60)=60） |
| `LEGAL_CHUNK_MODE` | `true` | 啟用法規語意切割 |
| `RETRIEVER_K` | `5` | 每個 retriever 各取幾份文件 |
| `BM25_WEIGHT` | `0.4` | BM25 在 Hybrid Search 的權重 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB 持久化路徑 |

---

## 五、Regex 邊界識別規則

```python
_CHAPTER_RE = r"第\s*[一二三四五六七八九十百千零\d]+\s*章[^\n]{0,30}"
_SECTION_RE = r"第\s*[一二三四五六七八九十百千零\d]+\s*節[^\n]{0,30}"
_ARTICLE_RE = r"第\s*[一二三四五六七八九十百千零\d]+(?:-\d+)?\s*條"
#                                                    ^^^^^^^
#                     ↑ \s* 允許 PDF 空格格式如「第 一 章」
#                                                 允許複合條號如「第 56-1 條」
```

---

## 六、LLM 回答格式

### 類型 A：法規查詢
```
[食品安全衛生管理法] 第22條：
• 條文內容：食品及食品原料之容器或外包裝，應以中文...
• 說明：...
```

### 類型 B：合規性審查
```
Q：宣稱「這款薑黃可以護肝」是否合法？

A：
① 違規認定：涉及醫療效能宣稱，違反健康食品管理法
② 相關法條：
   • 依 健康食品管理法.pdf 第2條第2項：「...非屬治療、矯正人類疾病之醫療效能...」
③ 判罰依據：...
④ 建議修改：改為「有助於維持正常生理功能」
```
 