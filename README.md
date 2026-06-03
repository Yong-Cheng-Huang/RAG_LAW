# RAG PDF 知識庫問答系統

一個針對台灣食品法規、食品標示與廣告合規審查設計的 PDF RAG 問答系統。使用者可以上傳 PDF，系統會依文件類型自動切割內容、建立 ChromaDB 向量索引，並透過 Hybrid Search 與 LLM 產生具來源依據的繁體中文回答。

## 功能特色

- PDF 上傳與知識庫建立
- 支援法規條文、統計表 / 處罰案件表、一般文件三種切割策略
- 法規文件會依章、節、條建立 contextual header，降低條文混淆
- 表格文件會優先抽取一案一筆的裁罰案例與罰鍰金額
- 結合 BM25 關鍵字搜尋、向量搜尋與 MultiQueryRetriever
- 支援 Ollama、OpenAI、Gemini 作為 LLM
- 支援 Ollama 或 Gemini Embedding
- 使用 Streamlit 提供簡潔的 Web UI
- 使用 ChromaDB 在本地持久化知識庫

## 系統架構

```text
PDF 上傳
  -> UnstructuredPDFLoader
  -> 文件切割 legal / table / default
  -> Embedding
  -> ChromaDB
  -> Hybrid Search: BM25 + Vector + MultiQuery
  -> 裁罰案例補強檢索
  -> LLM 生成回答
  -> Streamlit 顯示
```

詳細設計請參考 [architecture.md](architecture.md)。

## 專案結構

```text
.
├── app.py              # Streamlit 主介面
├── config.py           # 全域設定與 .env 讀取
├── vector_store.py     # PDF 載入、切割、Embedding、ChromaDB 操作
├── llm_chain.py        # Retriever、RAG prompt、LLM chain
├── architecture.md     # 架構文件
├── requirements.txt    # Python 套件需求
└── chroma_db/          # 本地向量資料庫，預設不提交 Git
```

## 快速開始

### 1. 建立虛擬環境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定環境變數

建立 `.env`，內容可參考下方範例。`.env` 會包含 API key 等私密設定，預設不提交 Git。

## `.env` 範例

### Ollama 本地模式

```env
LLM_MODE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
MULTIQUERY_MODEL=

EMBEDDING_MODE=ollama
EMBEDDING_MODEL=bge-m3

CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=pdf_knowledge_base

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LEGAL_CHUNK_MODE=true

RETRIEVER_K=5
PENALTY_CASE_K=25
BM25_WEIGHT=0.4
```

使用 Ollama 時，請先確認本機已啟動 Ollama，並已下載需要的模型：

```bash
ollama pull llama3
ollama pull bge-m3
```

### OpenAI 模式

```env
LLM_MODE=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

EMBEDDING_MODE=ollama
EMBEDDING_MODEL=bge-m3
```

### Gemini 模式

```env
LLM_MODE=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

EMBEDDING_MODE=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

## 啟動

```bash
streamlit run app.py
```

啟動後在瀏覽器開啟 Streamlit 顯示的網址，通常是：

```text
http://localhost:8501
```

## 使用方式

1. 在左側 sidebar 上傳一個或多個 PDF。
2. 選擇文件類型：
   - 法規條文：依條文邊界切割。
   - 統計表 / 處罰案件表：保留表格 header，並嘗試抽取裁罰案例。
   - 一般文件：使用一般字元數切割。
3. 點擊「開始攝取」建立知識庫。
4. 在主畫面輸入問題。
5. 系統會檢索相關文件片段並產生回答。

## 文件切割策略

| 類型 | 適用文件 | 說明 |
|---|---|---|
| `legal` | 法規、辦法、準則 | 偵測章、節、條、附則、附表，建立條文級 chunk |
| `table` | 裁罰案件、統計表 | 優先抽取裁罰案例；否則用 header + 行分組切割 |
| `default` | 一般 PDF | 依 `CHUNK_SIZE` 與 `CHUNK_OVERLAP` 切割 |

## 重要設定

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `LLM_MODE` | `ollama` | `ollama`、`openai` 或 `gemini` |
| `OLLAMA_MODEL` | `llama3` | Ollama 主生成模型 |
| `MULTIQUERY_MODEL` | 空字串 | MultiQuery 專用模型；空值代表沿用主模型 |
| `EMBEDDING_MODE` | `ollama` | `ollama` 或 `gemini` |
| `EMBEDDING_MODEL` | `bge-m3` | Ollama embedding 模型 |
| `CHUNK_SIZE` | `1000` | chunk 大小 |
| `CHUNK_OVERLAP` | `200` | 一般文件與長條文二次切割的重疊字元數 |
| `RETRIEVER_K` | `5` | 一般 retriever 候選文件數 |
| `PENALTY_CASE_K` | `25` | 裁罰案例候選文件數 |
| `BM25_WEIGHT` | `0.4` | Hybrid Search 中 BM25 的權重 |

## 常見問題

### 為什麼查不到剛上傳的 PDF？

請確認攝取流程有成功完成，且 sidebar 的「知識庫內容」有列出檔名。若 ChromaDB 狀態異常，可在 UI 中清空知識庫後重新攝取。

### 掃描版 PDF 可以用嗎？

如果 PDF 是掃描圖片，`UnstructuredPDFLoader` 可能無法取得足夠文字。建議先做 OCR，再上傳可選取文字的 PDF。

### `CHUNK_OVERLAP` 會影響表格嗎？

不會。表格模式使用 `chunk_overlap=0`，避免資料列重複造成裁罰金額或案件重複。`CHUNK_OVERLAP` 主要影響一般文件與法規超長條文的二次切割。

### 可以混用不同 Embedding 模型嗎？

不建議。若已用某個 embedding 模型建立知識庫，切換 embedding 模型後最好清空 `chroma_db/` 並重新攝取文件。

## 注意事項

- `.env`、`chroma_db/`、`.venv/` 預設不提交 Git。
- 本系統產生的回答應作為法規查詢與合規初步審查輔助，不應取代專業法律意見。
- 檢索品質取決於 PDF 文字解析品質、切割策略與知識庫內容完整性。