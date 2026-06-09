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
_PENALTY_AMOUNT_RE = re.compile(
    r"(?:新臺幣|新台幣|罰鍰|裁罰金額|罰鍰金額|處罰鍰|處以罰鍰|處)?"
    r"\s*([0-9０-９一二三四五六七八九十百千萬億兩壹貳參肆伍陸柒捌玖拾佰仟萬億,.，]+)"
    r"\s*(?:元|圓|萬元|萬|千元)"
)
_PENALTY_AMOUNT_HEADER_RE = re.compile(r"(?:罰\s*鍰|裁\s*罰)\s*金\s*額")
_PLAIN_AMOUNT_RE = re.compile(r"(?<![\d./-])([0-9０-９][0-9０-９,，.]*)")
_VIOLATION_CASE_HEADER_RE = re.compile(
    r"(?:產品\s*名稱|違規\s*情節|處分\s*商號\s*名稱|罰\s*鍰\s*金\s*額|罰則\s*註記)"
)
_CASE_NO_RE = re.compile(r"^\s*(?:\d+|[一二三四五六七八九十]+)[\.、\s]")
_DATE_RE = re.compile(r"^\s*\d{2,4}[-/.年]\d{1,2}[-/.月]\d{1,2}")
_CJK_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "壹": 1,
    "貳": 2,
    "參": 3,
    "肆": 4,
    "伍": 5,
    "陸": 6,
    "柒": 7,
    "捌": 8,
    "玖": 9,
}
_CJK_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_CJK_BIG_UNITS = {"萬": 10000, "亿": 100000000, "億": 100000000}


def get_embeddings() -> OllamaEmbeddings | GoogleGenerativeAIEmbeddings:
    """取得 Embedding 函數。
    EMBEDDING_MODE=ollama → Ollama
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


def load_table_pdf(file_path: str | Path) -> list[Document]:
    """
    載入表格型 PDF，嘗試用 element 座標將「欄位整欄輸出」還原為一列一案。
    還原失敗時退回一般 PDF loader。
    """
    loader = UnstructuredPDFLoader(str(file_path), mode="elements")
    docs = loader.load()
    table_docs = _rebuild_violation_table_documents(docs)
    return table_docs or load_pdf(file_path)


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


def _parse_cjk_number(text: str) -> int | None:
    """解析常見中文金額數字，例如「六萬」「壹拾貳萬」。"""
    total = 0
    section = 0
    number = 0
    seen = False

    for char in text:
        if char in _CJK_DIGITS:
            number = _CJK_DIGITS[char]
            seen = True
        elif char in _CJK_UNITS:
            section += (number or 1) * _CJK_UNITS[char]
            number = 0
            seen = True
        elif char in _CJK_BIG_UNITS:
            section += number
            total += (section or 1) * _CJK_BIG_UNITS[char]
            section = 0
            number = 0
            seen = True

    if not seen:
        return None
    return total + section + number


def _parse_amount_value(amount_text: str) -> int | None:
    """將裁罰金額文字轉為元；無法可靠解析時回傳 None。"""
    normalized = amount_text.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    match = re.search(r"([0-9][0-9,\.]*)", normalized)
    if match:
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            return None
        if re.search(r"萬\s*(?:元|圓)?|萬元", amount_text):
            value *= 10000
        elif re.search(r"千\s*(?:元|圓)?|千元", amount_text):
            value *= 1000
    else:
        cjk_match = re.search(r"([一二三四五六七八九十百千萬億兩壹貳參肆伍陸柒捌玖拾佰仟]+)", amount_text)
        if not cjk_match:
            return None
        parsed = _parse_cjk_number(cjk_match.group(1))
        if parsed is None:
            return None
        value = float(parsed)

    return int(value)


def _extract_penalty_amount(text: str) -> tuple[str, int | None] | None:
    """從一段文字中找出最像裁罰金額的片段。"""
    matches = list(_PENALTY_AMOUNT_RE.finditer(text))
    if not matches:
        return None

    def _score(match: re.Match) -> tuple[int, int]:
        raw = match.group(0)
        keyword_score = 1 if re.search(r"罰鍰|裁罰|處罰|新臺幣|新台幣", raw) else 0
        return keyword_score, len(raw)

    best = max(matches, key=_score)
    amount_text = best.group(0).strip()
    return amount_text, _parse_amount_value(amount_text)


def _extract_table_penalty_amount(header: str, text: str) -> tuple[str, int | None] | None:
    """
    表格 header 已標示「罰鍰金額」時，資料列常只剩純數字。
    此 fallback 僅在 header 有裁罰金額欄位時啟用，避免一般數字被誤判。
    """
    if not _PENALTY_AMOUNT_HEADER_RE.search(header):
        return None

    candidates: list[tuple[int, int, str]] = []
    for match in _PLAIN_AMOUNT_RE.finditer(text):
        raw = match.group(1)
        value = _parse_amount_value(f"{raw}元")
        if value is None or value <= 0:
            continue
        # 常見裁罰金額多為整數元；分數/小數欄位較不像罰鍰。
        if "." in raw and not raw.endswith(".000"):
            continue
        comma_score = 1 if re.search(r"[,，]", raw) else 0
        amount_scale_score = 1 if value >= 1000 else 0
        candidates.append((comma_score + amount_scale_score, match.start(), raw))

    if not candidates:
        return None

    _, _, amount_text = max(candidates)
    amount_value = _parse_amount_value(f"{amount_text}元")
    return f"{amount_text}元", amount_value


def _looks_like_case_start(line: str) -> bool:
    """粗略判斷是否像新案件列的開頭。"""
    return bool(_CASE_NO_RE.match(line) or _DATE_RE.match(line))


def _coord_points(doc: Document) -> tuple[tuple[float, float], ...] | None:
    coordinates = doc.metadata.get("coordinates") or {}
    points = coordinates.get("points")
    if not points:
        return None
    return tuple((float(x), float(y)) for x, y in points)


def _coord_center(doc: Document) -> tuple[float, float] | None:
    points = _coord_points(doc)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _coord_top(doc: Document) -> float | None:
    points = _coord_points(doc)
    if not points:
        return None
    return min(point[1] for point in points)


def _clean_table_text(text: str) -> str:
    """清理 PDF 表格文字中因窄欄換行留下的多餘空白。"""
    return re.sub(r"\s+", " ", text).strip()


def _nearest_row_index(row_ys: list[float], y: float) -> int:
    return min(range(len(row_ys)), key=lambda index: abs(row_ys[index] - y))


def _column_name_from_x(x: float) -> str | None:
    """依這類公告表的固定欄位位置判斷 element 所屬欄位。"""
    if x < 48:
        return "項次"
    if x < 76:
        return "發文日期"
    if x < 153:
        return "產品名稱"
    if x < 174:
        return "來源"
    if x < 432:
        return "違規情節"
    if x < 472:
        return "處分商號名稱"
    if x < 514:
        return "罰鍰金額"
    if x < 545:
        return "罰則註記"
    return "排名"


def _is_table_title(text: str) -> bool:
    return "處理食品" in text and "違規廣告處罰案件統計表" in text


def _rebuild_violation_table_documents(docs: list[Document]) -> list[Document]:
    """將 unstructured 以欄序輸出的違規廣告表，依座標重組成 row-based 文件。"""
    docs_by_page: dict[int, list[Document]] = {}
    for doc in docs:
        page_number = doc.metadata.get("page_number")
        if isinstance(page_number, int):
            docs_by_page.setdefault(page_number, []).append(doc)

    rebuilt_docs: list[Document] = []
    header = "項次\t發文日期\t產品名稱\t來源\t違規情節\t處分商號名稱\t罰鍰金額\t罰則註記\t排名"

    for page_number, page_docs in sorted(docs_by_page.items()):
        item_markers: list[tuple[float, str]] = []
        for doc in page_docs:
            center = _coord_center(doc)
            if not center:
                continue
            x, y = center
            text = _clean_table_text(doc.page_content)
            if x < 48 and re.fullmatch(r"\d+", text):
                item_markers.append((y, text))

        if not item_markers:
            continue

        item_markers.sort()
        row_ys = [item[0] for item in item_markers]
        rows: list[dict[str, str]] = [{"項次": item[1]} for item in item_markers]
        seen_values: set[tuple[int, str, str]] = set()

        for doc in page_docs:
            text = _clean_table_text(doc.page_content)
            if not text or _is_table_title(text):
                continue
            category = doc.metadata.get("category")
            if category in {"Header", "Footer"}:
                continue
            center = _coord_center(doc)
            top = _coord_top(doc)
            if not center or top is None:
                continue
            x, y = center
            column_name = _column_name_from_x(x)
            if not column_name:
                continue
            row_index = _nearest_row_index(row_ys, y)
            # 長篇違規情節的 bounding box 常往上延伸，使用中心點較穩；短欄位則中心/上緣皆可。
            if column_name != "違規情節":
                row_index = _nearest_row_index(row_ys, top)
            if column_name == "項次":
                continue
            key = (row_index, column_name, text)
            if key in seen_values:
                continue
            seen_values.add(key)
            current = rows[row_index].get(column_name, "")
            rows[row_index][column_name] = f"{current} {text}".strip() if current else text

        lines: list[str] = []
        for row in rows:
            if not row.get("產品名稱") and not row.get("違規情節"):
                continue
            lines.append(
                "\t".join(
                    [
                        row.get("項次", ""),
                        row.get("發文日期", ""),
                        row.get("產品名稱", ""),
                        row.get("來源", ""),
                        row.get("違規情節", ""),
                        row.get("處分商號名稱", ""),
                        row.get("罰鍰金額", ""),
                        row.get("罰則註記", ""),
                        row.get("排名", ""),
                    ]
                )
            )

        if lines:
            rebuilt_docs.append(
                Document(
                    page_content=f"{header}\n" + "\n".join(lines),
                    metadata={"page": page_number, "parser": "table_elements"},
                )
            )

    return rebuilt_docs


def _split_table_cells(text: str) -> list[str]:
    """依 PDF 常見欄位間隔拆 cell；拆不開時保留整列。"""
    normalized = text.replace("｜", "|").replace("│", "|")
    if "|" in normalized:
        return [cell.strip() for cell in normalized.split("|") if cell.strip()]
    cells = re.split(r"\t+|\s{2,}", normalized.strip())
    return [cell.strip() for cell in cells if cell.strip()]


def _normalize_header_cell(text: str) -> str:
    """將表格欄名正規化，方便比對「罰鍰金額」等格式。"""
    return re.sub(r"[\s()（）:：]", "", text)


def _find_header_index(header_cells: list[str], aliases: tuple[str, ...]) -> int | None:
    normalized_aliases = tuple(_normalize_header_cell(alias) for alias in aliases)
    for index, cell in enumerate(header_cells):
        normalized = _normalize_header_cell(cell)
        if any(alias in normalized for alias in normalized_aliases):
            return index
    return None


def _parse_violation_case_row(header: str, line: str) -> dict[str, str]:
    """把違規案例表的一列整理為固定欄位；拆欄失敗時保留原始資料列。"""
    header_cells = _split_table_cells(header)
    row_cells = _split_table_cells(line)
    fields: dict[str, str] = {"raw_row": line.strip()}

    field_aliases = {
        "document_date": ("發文日期", "裁決書發文日期", "裁處書發文日期", "處分書發文日期"),
        "product_name": ("產品名稱", "品名", "產品"),
        "media_source": ("來源", "廣告來源", "刊播來源"),
        "violation_details": ("違規情節", "違規內容", "違規詞句", "違規廣告詞句"),
        "disposition_name": ("處分商號名稱", "處分商號", "受處分人", "商號名稱"),
        "penalty_amount": ("罰鍰金額", "裁罰金額"),
        "penalty_note": ("罰則註記", "違反法條", "違反法規", "違反條文"),
        "rank": ("排名",),
    }

    for field_name, aliases in field_aliases.items():
        index = _find_header_index(header_cells, aliases)
        if index is not None and index < len(row_cells):
            fields[field_name] = row_cells[index]

    return fields


def _is_violation_case_table(header: str) -> bool:
    """判斷 header 是否像食品/健康食品違規廣告處罰案件表。"""
    return bool(_VIOLATION_CASE_HEADER_RE.search(header))


def _build_penalty_case_docs(doc: Document, header: str, lines: list[str]) -> list[Document]:
    """將違規案例表整理成一列一個 chunk。"""
    case_docs: list[Document] = []
    is_violation_case_table = _is_violation_case_table(header)

    for index, line in enumerate(lines):
        fields = _parse_violation_case_row(header, line)
        document_date = fields.get("document_date", "").strip()
        product_name = fields.get("product_name", "").strip()
        media_source = fields.get("media_source", "").strip()
        violation_details = fields.get("violation_details", "").strip()
        disposition_name = fields.get("disposition_name", "").strip()
        penalty_note = fields.get("penalty_note", "").strip()
        rank = fields.get("rank", "").strip()
        if fields.get("penalty_amount"):
            amount_text = fields["penalty_amount"]
            amount_value = _parse_amount_value(f"{amount_text}元")
            amount = (amount_text, amount_value)
        else:
            amount = _extract_penalty_amount(line) or _extract_table_penalty_amount(header, line)
            amount_text, amount_value = amount if amount else ("", None)
        if not is_violation_case_table and not amount:
            continue

        context_lines: list[str] = []
        if not is_violation_case_table and index > 0 and not _looks_like_case_start(line):
            context_lines.append(lines[index - 1])
        if fields["raw_row"]:
            context_lines.append(f"原始資料列：{fields['raw_row']}")
        if (
            not is_violation_case_table
            and index + 1 < len(lines)
            and not _looks_like_case_start(lines[index + 1])
        ):
            context_lines.append(lines[index + 1])

        case_title = f"{product_name}違規廣告" if product_name else "違規廣告案例"
        case_lines = [
            f"案例：{case_title}",
            f"裁決書發文日期：{document_date or '未載明'}",
            f"產品名稱：{product_name or '未載明'}",
            f"廣告來源：{media_source or '未載明'}",
            f"違規情節：{violation_details or '未載明'}",
            f"處分商號名稱：{disposition_name or '未載明'}",
            f"罰則註記：{penalty_note or '未載明'}",
        ]
        if amount_text:
            case_lines.append(f"罰鍰金額：{amount_text}")
        if rank:
            case_lines.append(f"排名：{rank}")
        case_lines.extend(context_lines)
        case_text = "\n".join(case_lines)

        metadata = {
            **doc.metadata,
            "doc_type": "penalty_case",
            "row_start": index + 1,
        }
        if document_date:
            metadata["document_date"] = document_date
        if product_name:
            metadata["product_name"] = product_name
        if media_source:
            metadata["media_source"] = media_source
        if violation_details:
            metadata["violation_details"] = violation_details
        if disposition_name:
            metadata["disposition_name"] = disposition_name
        if penalty_note:
            metadata["penalty_note"] = penalty_note
        if rank:
            metadata["rank"] = rank
        if amount_text:
            metadata["penalty_amount_text"] = amount_text
        if amount_value is not None:
            metadata["penalty_amount"] = amount_value

        case_docs.append(Document(page_content=case_text, metadata=metadata))

    return case_docs


def _table_split(documents: list) -> list[Document]:
    """
    表格型文件切割（處罰案件統計表等）：
    - 優先將抓得到罰鍰金額的列整理為 doc_type='penalty_case'，一案一筆
    - 自動偵測第一行為 header（欄位名稱列）
    - 動態計算每批行數，確保 header + 資料行的總字元數不超過 CHUNK_SIZE
    - 若單行本身就超長，退回 RecursiveCharacterTextSplitter 做二次切割
    - metadata 加入 doc_type / row_start / penalty_amount 方便追蹤來源列

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
        penalty_case_docs = _build_penalty_case_docs(doc, header, data_lines)
        if penalty_case_docs:
            all_chunks.extend(penalty_case_docs)
            continue

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
    docs = load_table_pdf(file_path) if doc_type == "table" else load_pdf(file_path)
    chunks = split_documents(docs, doc_type=doc_type)

    if not chunks:
        return 0

    # 加入來源 metadata，優先使用 display_name
    filename = display_name or Path(file_path).name
    for chunk in chunks:
        chunk.metadata["source_file"] = filename
        chunk.metadata.setdefault("doc_type", doc_type)
        if chunk.metadata.get("doc_type") == "penalty_case" and "\n來源：" not in chunk.page_content:
            chunk.page_content = f"{chunk.page_content}\n來源：{Path(filename).stem}"

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
