# 免費部署指南：Streamlit Community Cloud + Gemini

本專案最簡單的免費部署方式是 Streamlit Community Cloud。正式環境使用 Gemini，不需要另外架 Ollama，也不需要使用者登入。

## 重要限制

Streamlit Community Cloud 不保證本地檔案儲存會永久保留，因此 `chroma_db/` 可能在 app 休眠、重啟或重新部署後消失。這代表使用者上傳並攝取的 PDF 知識庫可能需要重新建立。

若之後需要穩定保留知識庫，建議改成有 persistent disk 的付費平台，或把向量庫改接外部服務。

## 事前準備

1. 將專案推到 GitHub repository。
2. 確認 repository 內有：
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
3. 不要提交 `.env`、`.streamlit/secrets.toml`、`chroma_db/`。

## Streamlit Cloud 部署步驟

1. 前往 `https://share.streamlit.io/`。
2. 點選 `Create app`。
3. 選擇 GitHub repository、branch，以及 entrypoint：

   ```text
   app.py
   ```

4. 在 `Advanced settings` 中選擇 Python 版本。建議選擇和本機相近的穩定版本，例如 Python 3.12。
5. 在 `Secrets` 欄位貼上以下設定，並將 Gemini API key 換成正式 key：

   ```toml
   LLM_MODE = "gemini"
   EMBEDDING_MODE = "gemini"

   GEMINI_API_KEY = "your_gemini_api_key"
   GEMINI_MODEL = "gemini-2.0-flash"
   GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

   CHROMA_PERSIST_DIR = "./chroma_db"
   CHROMA_COLLECTION = "pdf_knowledge_base"

   CHUNK_SIZE = "500"
   CHUNK_OVERLAP = "60"
   RETRIEVER_K = "8"
   PENALTY_CASE_K = "25"
   BM25_WEIGHT = "0.7"
   ```

6. 按下 Deploy。

## 部署後測試

1. 開啟 Streamlit Cloud 提供的 `*.streamlit.app` 網址。
2. 上傳一份 PDF。
3. 文件類型選擇對應模式：
   - 法規條文：`法規條文（依條切割）`
   - 裁罰案件表：`統計表 / 處罰案件表（表格切割）`
   - 一般文件：`一般文件（字元數切割）`
4. 點選 `開始攝取`。
5. 問一個 PDF 內容中可驗證的問題。

## 常見部署問題

### 知識庫重啟後不見

這是免費 Streamlit Cloud 的本地儲存限制。重新上傳 PDF 並攝取即可。

### PDF 解析失敗

本專案已加入 `packages.txt` 安裝 PDF 解析常用系統套件：

```text
poppler-utils
tesseract-ocr
tesseract-ocr-chi-tra
libmagic1
libgl1
libglib2.0-0
```

若特定 PDF 仍失敗，建議先確認該 PDF 是否可選取文字。掃描版 PDF 可能需要先 OCR。

### Gemini key 沒有生效

請確認 key 是貼在 Streamlit Cloud 的 Secrets 欄位，而不是提交到 GitHub。Secrets 內容需使用 TOML 格式，且 root-level key 會作為環境變數供程式讀取。
