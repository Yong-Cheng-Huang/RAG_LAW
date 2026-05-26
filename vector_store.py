"""
向量資料庫模組 — 管理 ChromaDB 的初始化、文件攝取與檢索。
增強：法規條文語意切割 + Contextual Header 注入 + BM25 全文件取回。
"""

import re
from pathlib import Path

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import settings


# ── 法規結構 Regex ────────────────────────────────────────
# 用來辨識章節/條文邊界的正規表示式
_CHAPTER_RE = re.compile(r"第\s*[一二三四五六七八九十百千零\d]+\s*章[^\n]{0,30}")
_SECTION_RE = re.compile(r"第\s*[一二三四五六七八九十百千零\d]+\s*節[^\n]{0,30}")
# _ARTICLE_RE = re.compile(r"第\s*[一二三四五六七八九十百千零\d]+\s*條")
_ARTICLE_RE = re.compile(r"第\s*[一二三四五六七八九十百千零\d]+(?:-\d+)?\s*條")
_APPENDIX_RE = re.compile(r"附[則表錄]")


def get_embeddings() -> OllamaEmbeddings | GoogleGenerativeAIEmbeddings:
    """取得 Embedding 函數。
    EMBEDDING_MODE=ollama → Ollama (bge-m3 等)
    EMBEDDING_MODE=gemini → Google Gemini Embedding 2
    """
    if settings.EMBEDDING_MODE == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY or None,
        )
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def get_vector_store() -> Chroma:
    """取得或建立 ChromaDB 向量資料庫實例。"""
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def load_pdf(file_path: str | Path) -> list:
    """使用 UnstructuredPDFLoader 載入單一 PDF。"""
    loader = UnstructuredPDFLoader(str(file_path))
    return loader.load()


# ── Contextual Header 注入 ────────────────────────────────

def _inject_context_markers(text: str) -> str:
    """
    掃描全文，在每個條文開頭前插入 <<<CONTEXT:...>>> 標記。
    標記包含目前所在的章、節、條名稱，供後續切割後提取使用。
    """
    current_chapter: str = ""
    current_section: str = ""
    result_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        ch_match = _CHAPTER_RE.match(stripped)
        sec_match = _SECTION_RE.match(stripped)
        art_match = _ARTICLE_RE.match(stripped)
        app_match = _APPENDIX_RE.match(stripped)

        if ch_match:
            current_chapter = re.sub(r"\s+", "", ch_match.group(0).strip())
            current_section = ""          # 換章時清空節

        if sec_match:
            current_section = re.sub(r"\s+", "", sec_match.group(0).strip())

        # 每遇到條文或附則，插入 CONTEXT 標記
        if art_match or app_match:
            article_label = re.sub(r"\s+", "", stripped[:20])
            ctx_parts = [p for p in [current_chapter, current_section, article_label] if p]
            context_str = " | ".join(ctx_parts)
            result_lines.append(f"<<<CONTEXT:{context_str}>>>")

        result_lines.append(line)

    return "\n".join(result_lines)


def _enrich_chunks_with_headers(chunks: list[Document]) -> list[Document]:
    """
    從每個 chunk 的文字中提取 CONTEXT 標記：
    - 轉成可讀 header 注入 page_content
    - 解析 chapter / article 寫入 metadata（B+C）
    - 移除原始標記
    如果 chunk 被二次切割導致無 CONTEXT，從 metadata 繼承上一個 chunk 的值。
    """
    _CTX_PATTERN = re.compile(r"<<<CONTEXT:([^>]+)>>>")
    enriched: list[Document] = []

    last_chapter: str = ""
    last_article: str = ""

    for chunk in chunks:
        text = chunk.page_content
        contexts = _CTX_PATTERN.findall(text)
        clean_text = _CTX_PATTERN.sub("", text).strip()

        if contexts:
            # 使用第一個 CONTEXT（代表此 chunk 的起始條文脈絡）
            ctx = contexts[0]  # e.g. "第一章 總則 | 第 1 條"
            header = f"[{ctx}]"
            enriched_text = f"{header}\n{clean_text}"

            # 解析 chapter / article 存入 metadata（B+C）
            parts = [p.strip() for p in ctx.split(" | ")]
            for part in parts:
                norm = part.replace(" ", "")
                if _CHAPTER_RE.match(norm) or _CHAPTER_RE.match(part):
                    last_chapter = part
                if _ARTICLE_RE.match(norm) or _ARTICLE_RE.match(part):
                    last_article = part
        else:
            # 二次切割後無 CONTEXT → 繼承前一條的脈絡
            enriched_text = clean_text

        chunk.page_content = enriched_text
        # 寫入 metadata
        if last_chapter:
            chunk.metadata["chapter"] = last_chapter
        if last_article:
            chunk.metadata["article"] = last_article
        enriched.append(chunk)

    return enriched


# ── 切割策略 ──────────────────────────────────────────────

# doc_type 對應的切割函數
# 新增文件類型時，只要在此 dict 加入對應的處理函數即可
_SPLIT_DISPATCH: dict[str, callable] = {}


def split_documents(documents: list, doc_type: str = "legal") -> list:
    """
    依 doc_type 選擇切割策略：
      - 'legal'  → 法規條文語意切割（依「條」邊界）
      - 'table'  → 表格行分組切割（統計表、處罰案件表等）
      - 'default'→ 純字元數切割（fallback）
    """
    if doc_type in _SPLIT_DISPATCH:
        return _SPLIT_DISPATCH[doc_type](documents)
    if doc_type == "table":
        return _table_split(documents)
    if settings.LEGAL_CHUNK_MODE and doc_type == "legal":
        return _legal_split(documents)
    return _default_split(documents)


def _default_split(documents: list) -> list:
    """原始字元數切割（保留作為 fallback）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def _table_split(documents: list) -> list[Document]:
    """
    表格型文件切割（處罰案件統計表等）：
    - 自動偵測第一行為 header（欄位名稱列）
    - 動態計算每批行數，確保 header + 資料行的總字元數不超過 CHUNK_SIZE
    - 若單行本身就超長，退回 RecursiveCharacterTextSplitter 做二次切割
    - metadata 加入 doc_type='table' 與 row_start 方便追蹤來源列

    若 PDF 解析後表格結構不規則（欄位散落），建議搭配
    UnstructuredPDFLoader(mode='elements') 先過濾出 Table element。
    """
    
    TABLE_CHUNK_SIZE = min(settings.CHUNK_SIZE, 800)

    safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE,
        chunk_overlap=0,  # 表格行不做 overlap，避免重複資料列
        separators=["\n", "。", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in documents:
        lines = [ln for ln in doc.page_content.split("\n") if ln.strip()]
        if not lines:
            continue

        # 假設第一行是 header（欄位名稱）
        header = lines[0]
        header_len = len(header) + 1  # +1 for "\n"
        data_lines = lines[1:]

        batch: list[str] = []
        batch_len = header_len
        row_start = 1

        def _flush(batch: list[str], row_start: int) -> list[Document]:
            """將一批行組成 chunk，若超長就用 safety_splitter 二次切割。"""
            chunk_text = f"{header}\n" + "\n".join(batch)
            base_meta = {**doc.metadata, "doc_type": "table", "row_start": row_start}
            if len(chunk_text) <= TABLE_CHUNK_SIZE:
                return [Document(page_content=chunk_text, metadata=base_meta)]
            # 超長：safety splitter 切割，每個子 chunk 都保留 header
            sub_docs = safety_splitter.create_documents([chunk_text], metadatas=[base_meta])
            return sub_docs

        for line in data_lines:
            line_len = len(line) + 1  # +1 for "\n"

            # 若單行本身就超過上限，先 flush 現有 batch，再單獨處理這行
            if line_len > TABLE_CHUNK_SIZE:
                if batch:
                    all_chunks.extend(_flush(batch, row_start))
                    row_start += len(batch)
                    batch = []
                    batch_len = header_len
                all_chunks.extend(_flush([line], row_start))
                row_start += 1
                continue

            # 加入此行後會超限 → 先 flush
            if batch_len + line_len > TABLE_CHUNK_SIZE:
                all_chunks.extend(_flush(batch, row_start))
                row_start += len(batch)
                batch = []
                batch_len = header_len

            batch.append(line)
            batch_len += line_len

        # 剩餘尾巴
        if batch:
            all_chunks.extend(_flush(batch, row_start))

    return all_chunks


def _legal_split(documents: list) -> list:
    """
    法規語意切割：
    1. 注入 CONTEXT 標記
    2. 直接以 CONTEXT 標記為邊界手動切割（每條嚴格獨立成一個 chunk）
    3. 超長條文再以字元數二次切割
    4. 提取 CONTEXT 標記轉為可讀 header
    """
    # 超長條文的二次切割器
    secondary_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"\n(?=第\s*[一二三四五六七八九十百千零\d]+\s*項)",  # 第X項
            r"\n(?=[一二三四五六七八九十]+、)",                    # 一、二、…
            "\n\n",
            "\n",
            "。",
            " ",
            "",
        ],
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        is_separator_regex=True,
        keep_separator=True,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        # 先注入 CONTEXT 標記
        annotated_text = _inject_context_markers(doc.page_content)

        # 直接以 <<<CONTEXT: 為邊界切割，每條嚴格獨立
        parts = re.split(r"\n(?=<<<CONTEXT:)", annotated_text)

        for part in parts:
            if not part.strip():
                continue

            if len(part) > settings.CHUNK_SIZE:
                # 超長條文二次切割
                sub_chunks = secondary_splitter.create_documents(
                    [part], metadatas=[doc.metadata]
                )
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(Document(page_content=part, metadata=doc.metadata.copy()))

    # 提取標記，轉換為可讀 header
    return _enrich_chunks_with_headers(all_chunks)


# ── 攝取流程 ──────────────────────────────────────────────

def ingest_pdf(
    file_path: str | Path,
    display_name: str | None = None,
    doc_type: str = "legal",
) -> int:
    """
    完整的 PDF 攝取流程：載入 → 切割 → 向量化 → 存入 ChromaDB。

    doc_type 控制切割策略：
      - 'legal'   法規條文（依「條」邊界，預設）
      - 'table'   統計表/處罰案件表（保留 header，固定行數分組）
      - 'default' 純字元數切割

    display_name: 覆蓋存入 metadata 的檔名（例如原始上傳檔名），
                  若不傳則退回使用 file_path 的 basename。
    回傳切割後的 chunk 數量。
    """
    docs = load_pdf(file_path)
    chunks = split_documents(docs, doc_type=doc_type)

    if not chunks:
        return 0

    # 加入來源 metadata，優先使用 display_name
    filename = display_name or Path(file_path).name
    for chunk in chunks:
        chunk.metadata["source_file"] = filename
        chunk.metadata.setdefault("doc_type", doc_type)

    db = get_vector_store()
    db.add_documents(chunks)
    return len(chunks)


# ── 查詢工具 ──────────────────────────────────────────────

def get_all_sources() -> list[str]:
    """列出資料庫中所有已攝取的來源檔名。"""
    db = get_vector_store()
    collection = db._collection
    result = collection.get(include=["metadatas"])
    sources = set()
    for meta in result.get("metadatas", []):
        if meta and "source_file" in meta:
            sources.add(meta["source_file"])
    return sorted(sources)


def get_all_documents() -> list[Document]:
    """
    從 ChromaDB 取出所有文件，用於建立 BM25 索引。
    注意：文件量大時略慢，但 BM25 建立後是純記憶體操作。
    """
    db = get_vector_store()
    collection = db._collection
    result = collection.get(include=["documents", "metadatas"])

    docs: list[Document] = []
    for text, meta in zip(
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def delete_collection():
    """清空整個向量資料庫 collection。"""
    db = get_vector_store()
    db.delete_collection()
