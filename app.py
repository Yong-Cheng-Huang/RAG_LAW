"""
RAG PDF 知識庫問答系統 — Streamlit 主介面
"""

import sys
from pathlib import Path

# Python 3.14+ 不再自動將腳本目錄加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import tempfile
import os
import time

from config import settings
from vector_store import ingest_pdf, get_all_sources, delete_collection
from llm_chain import ask, get_llm

# ── 頁面設定 ─────────────────────────────────────────────
st.set_page_config(
    page_title="食品法規 AI 問答平台",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── 自訂 CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --ink: #152018;
        --muted: #637268;
        --paper: #f7f8f3;
        --panel: rgba(255, 255, 250, 0.88);
        --panel-solid: #fffefa;
        --line: rgba(21, 32, 24, 0.12);
        --green: #256a4d;
        --green-dark: #123f31;
        --copper: #b45f32;
        --gold: #c79a3a;
        --blue: #1d4f73;
        --shadow: 0 18px 45px rgba(28, 39, 31, 0.10);
    }

    html, body {
        font-family: ui-serif, "Iowan Old Style", "Palatino Linotype", "Noto Serif TC", Georgia, serif;
    }

    .material-icons,
    .material-icons-round,
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: "Material Symbols Rounded", "Material Icons Round", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
    }

    .stApp {
        color: var(--ink);
        background:
            linear-gradient(90deg, rgba(21, 32, 24, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(21, 32, 24, 0.03) 1px, transparent 1px),
            linear-gradient(118deg, rgba(37, 106, 77, 0.12) 0%, transparent 34%),
            linear-gradient(145deg, #f8f9f5 0%, #eef4ef 48%, #f7f4ed 100%);
        background-size: 44px 44px, 44px 44px, auto, auto;
    }

    .main .block-container {
        max-width: 1180px;
        padding: 3.5rem 2.5rem 6rem;
    }

    [data-testid="stSidebar"] {
        background: #fbfbf6;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--ink);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        letter-spacing: 0;
        color: var(--green-dark);
    }

    [data-testid="stSidebar"] .stAlert {
        border-radius: 8px;
        border: 1px solid rgba(37, 106, 77, 0.22);
        background: rgba(37, 106, 77, 0.08);
    }

    [data-testid="stSidebarHeader"] {
        min-height: 56px;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        z-index: 999;
        visibility: visible !important;
    }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"] {
        position: relative;
        width: 34px;
        height: 34px;
        border: 1px solid rgba(18, 63, 49, 0.18);
        border-radius: 8px;
        background: rgba(255, 255, 250, 0.94);
        color: var(--green-dark);
        box-shadow: 0 8px 22px rgba(28, 39, 31, 0.10);
        visibility: visible !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stExpandSidebarButton"]:hover {
        border-color: rgba(37, 106, 77, 0.38);
        background: #fffefa;
        color: var(--green);
    }

    [data-testid="stSidebarCollapseButton"] button::before,
    [data-testid="stSidebarCollapsedControl"] button::before,
    [data-testid="stExpandSidebarButton"]::before {
        content: "‹";
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        color: var(--green-dark);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 1.8rem;
        font-weight: 900;
        line-height: 1;
        transform: translateY(-1px);
    }

    [data-testid="stSidebarCollapsedControl"] button::before,
    [data-testid="stExpandSidebarButton"]::before {
        content: "›";
    }

    [data-testid="stSidebarCollapseButton"] button > *,
    [data-testid="stSidebarCollapsedControl"] button > *,
    [data-testid="stExpandSidebarButton"] > * {
        opacity: 0;
    }

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stExpandSidebarButton"] span {
        color: var(--green-dark) !important;
        fill: currentColor !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(2rem, 5vw, 4.5rem);
        border: 1px solid var(--line);
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(255, 255, 250, 0.96), rgba(239, 246, 238, 0.90)),
            repeating-linear-gradient(120deg, rgba(37, 106, 77, 0.055) 0 1px, transparent 1px 16px);
        box-shadow: var(--shadow);
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: auto 0 0 0;
        height: 7px;
        background: linear-gradient(90deg, var(--green), var(--gold), var(--copper), var(--blue));
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 1.1rem;
        color: var(--green-dark);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .eyebrow::before {
        content: "";
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 0 6px rgba(37, 106, 77, 0.12);
    }

    .hero h1 {
        max-width: 780px;
        margin: 0;
        color: var(--ink);
        font-size: clamp(2.1rem, 4.4vw, 4rem);
        line-height: 1.12;
        letter-spacing: 0;
        font-weight: 780;
        word-break: keep-all;
        overflow-wrap: normal;
    }

    .hero p {
        max-width: 760px;
        margin: 1.35rem 0 0;
        color: #38463d;
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: clamp(1rem, 1.7vw, 1.2rem);
        line-height: 1.78;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.75fr);
        gap: 1.5rem;
        margin-top: 2rem;
        align-items: end;
    }

    .hero-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.75rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        min-height: 36px;
        padding: 0.45rem 0.8rem;
        border: 1px solid rgba(21, 32, 24, 0.14);
        border-radius: 999px;
        background: rgba(255, 255, 250, 0.74);
        color: #27342b;
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 0.88rem;
        font-weight: 700;
    }

    .signal-panel {
        padding: 1.1rem;
        border: 1px solid rgba(21, 32, 24, 0.13);
        border-radius: 8px;
        background: rgba(18, 63, 49, 0.94);
        color: #f7f8f3;
        box-shadow: 0 12px 30px rgba(18, 63, 49, 0.22);
    }

    .signal-panel .label {
        color: rgba(247, 248, 243, 0.68);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .signal-panel .value {
        margin-top: 0.35rem;
        font-size: 2.4rem;
        line-height: 1;
        font-weight: 760;
    }

    .signal-panel .detail {
        margin-top: 0.8rem;
        color: rgba(247, 248, 243, 0.78);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .section-title {
        margin: 2.2rem 0 0.85rem;
        color: var(--green-dark);
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: 0;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 1.2rem 0 0.4rem;
    }

    .info-card {
        min-height: 154px;
        padding: 1.1rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        box-shadow: 0 8px 24px rgba(28, 39, 31, 0.06);
    }

    .info-card strong {
        display: block;
        margin-bottom: 0.45rem;
        color: var(--ink);
        font-size: 1.05rem;
    }

    .info-card span {
        display: block;
        color: var(--muted);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 0.92rem;
        line-height: 1.62;
    }

    .chat-shell {
        margin-top: 1.5rem;
        padding: 1px;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(37, 106, 77, 0.30), rgba(199, 154, 58, 0.24), rgba(29, 79, 115, 0.18));
    }

    .chat-inner {
        padding: 1.1rem 1.2rem;
        border-radius: 7px;
        background: rgba(255, 255, 250, 0.82);
        color: var(--muted);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 0.95rem;
    }

    .stChatMessage {
        border: 1px solid rgba(21, 32, 24, 0.10);
        border-radius: 8px;
        background: rgba(255, 255, 250, 0.82);
        box-shadow: 0 10px 24px rgba(28, 39, 31, 0.055);
    }

    .stChatMessage [data-testid="stMarkdownContainer"] {
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        line-height: 1.75;
    }

    .ask-dock-title {
        margin: 0 0 0.55rem;
        color: var(--green-dark);
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: 0;
    }

    [class*="st-key-ask-dock"],
    [class*="st-key-ask_dock"] {
        margin-top: 1.5rem;
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
        box-sizing: border-box;
        padding: 0.78rem;
        border: 1px solid rgba(21, 32, 24, 0.12);
        border-radius: 8px;
        background: rgba(255, 255, 250, 0.76);
        box-shadow: 0 10px 30px rgba(28, 39, 31, 0.06);
    }

    [class*="st-key-ask-dock"] [data-testid="stMarkdownContainer"],
    [class*="st-key-ask_dock"] [data-testid="stMarkdownContainer"] {
        padding-right: 0;
    }

    [class*="st-key-ask-dock"] [data-testid="stForm"],
    [class*="st-key-ask_dock"] [data-testid="stForm"] {
        margin: 0;
        padding: 0;
        border: 0;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
        backdrop-filter: none;
        width: 100% !important;
    }

    [class*="st-key-ask-dock"] [data-testid="stForm"] > div,
    [class*="st-key-ask_dock"] [data-testid="stForm"] > div {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 112px;
        gap: 0.7rem;
        align-items: center;
    }

    [class*="st-key-ask-dock"] [data-testid="stTextInput"],
    [class*="st-key-ask_dock"] [data-testid="stTextInput"] {
        margin: 0;
    }

    [class*="st-key-ask-dock"] [data-testid="stFormSubmitButton"],
    [class*="st-key-ask_dock"] [data-testid="stFormSubmitButton"] {
        margin: 0;
    }

    [class*="st-key-ask-dock"] [data-testid="stFormSubmitButton"] button,
    [class*="st-key-ask_dock"] [data-testid="stFormSubmitButton"] button {
        height: 52px;
        min-height: 52px;
        border-color: var(--green-dark);
        background: var(--green-dark);
        color: #fffefa;
        box-shadow: 0 8px 20px rgba(18, 63, 49, 0.18);
    }

    [class*="st-key-ask-dock"] [data-testid="stFormSubmitButton"] button:hover,
    [class*="st-key-ask_dock"] [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--green);
        background: var(--green);
        color: #fffefa;
    }

    [class*="st-key-ask-dock"] [data-testid="stForm"] [data-testid="stTextInput"] input,
    [class*="st-key-ask_dock"] [data-testid="stForm"] [data-testid="stTextInput"] input {
        height: 52px;
        min-height: 52px;
        padding-top: 0;
        padding-bottom: 0;
        border-color: rgba(18, 63, 49, 0.20);
        background: #fffefa;
        color: var(--ink);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 1rem;
        line-height: 52px;
    }

    [class*="st-key-ask-dock"] [data-testid="stForm"] [data-baseweb="input"],
    [class*="st-key-ask_dock"] [data-testid="stForm"] [data-baseweb="input"] {
        min-height: 52px;
        align-items: center;
        border-radius: 8px;
        background: #fffefa;
    }

    .stButton > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {
        min-height: 42px;
        border-radius: 8px;
        border: 1px solid rgba(18, 63, 49, 0.22);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-weight: 800;
        letter-spacing: 0;
    }

    .stButton > button:hover {
        border-color: var(--green);
        color: var(--green-dark);
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(37, 106, 77, 0.32);
        border-radius: 8px;
        background: rgba(37, 106, 77, 0.045);
        padding: 0.4rem;
    }

    div[data-baseweb="select"] > div,
    textarea,
    input {
        border-radius: 8px !important;
    }

    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        visibility: hidden;
    }

    @media (max-width: 1100px) {
        .hero-grid {
            grid-template-columns: 1fr;
            align-items: stretch;
        }
    }

    @media (max-width: 820px) {
        .main .block-container {
            padding: 2rem 1rem 5.5rem;
        }

        .hero {
            padding: 1.35rem;
        }

        .info-grid {
            grid-template-columns: 1fr;
        }

        [class*="st-key-ask-dock"] [data-testid="stForm"] > div,
        [class*="st-key-ask_dock"] [data-testid="stForm"] > div {
            grid-template-columns: 1fr;
            gap: 0.55rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Session State 初始化 ─────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def get_runtime_labels() -> tuple[str, str, str]:
    """回傳目前 LLM / Embedding 顯示文字。"""
    if settings.LLM_MODE == "ollama":
        mode_label = "本地模式 · Ollama"
        llm_name = settings.OLLAMA_MODEL
    elif settings.LLM_MODE == "gemini":
        mode_label = "雲端模式 · Gemini"
        llm_name = settings.GEMINI_MODEL
    else:
        mode_label = "雲端模式 · OpenAI"
        llm_name = settings.OPENAI_MODEL

    emb_display = (
        f"Gemini · {settings.GEMINI_EMBEDDING_MODEL}"
        if settings.EMBEDDING_MODE == "gemini"
        else f"Ollama · {settings.EMBEDDING_MODEL}"
    )
    return mode_label, llm_name, emb_display


def safe_sources() -> list[str]:
    try:
        return get_all_sources()
    except Exception:
        return []


mode_label, llm_name, emb_display = get_runtime_labels()

# ── 側邊欄 ───────────────────────────────────────────────
with st.sidebar:
    st.title("Control Room")
    st.caption("文件攝取、模型狀態與知識庫維護")

    st.markdown("---")

    # 模式顯示
    st.info(f"**目前模式：** {mode_label}")
    st.caption(f"**LLM：** `{llm_name}`")
    st.caption(f"**Embedding：** `{emb_display}`")

    st.markdown("---")

    # PDF 上傳
    st.subheader("上傳 PDF")
    uploaded_files = st.file_uploader(
        "選擇 PDF 檔案",
        type=["pdf"],
        accept_multiple_files=True,
        help="支援多檔上傳，系統將自動解析並建立索引。",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        doc_type = st.selectbox(
            "📋 文件類型",
            options=[
                ("法規條文（依條切割）", "legal"),
                ("統計表 / 處罰案件表（表格切割）", "table"),
                ("一般文件（字元數切割）", "default"),
            ],
            format_func=lambda x: x[0],
            help="不同文件類型使用不同切割策略，影響 RAG 召回品質。",
        )[1]

        if st.button("🚀 開始攝取", use_container_width=True):
            all_success = True
            for uploaded_file in uploaded_files:
                with st.spinner(f"正在處理: {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        chunk_count = ingest_pdf(
                            tmp_path,
                            display_name=uploaded_file.name,
                            doc_type=doc_type,
                        )
                        st.success(
                            f"✅ **{uploaded_file.name}** — 切割為 {chunk_count} 個片段"
                        )
                        st.session_state.ingested_files.append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name} 處理失敗: {e}")
                        all_success = False
                    finally:
                        os.unlink(tmp_path)

            # 讓使用者看到成功/失敗訊息後再清空 uploader
            time.sleep(1.5)
            st.session_state.uploader_key += 1
            st.rerun()

    st.markdown("---")

    # 已攝取的文件
    st.subheader("知識庫內容")
    sources = safe_sources()
    if sources:
        for src in sources:
            st.markdown(f"- `{src}`")
    else:
        st.caption("知識庫目前為空，請上傳 PDF 文件。")

    st.markdown("---")

    # 清空知識庫
    if st.button("🗑️ 清空知識庫", use_container_width=True, type="secondary"):
        try:
            delete_collection()
            st.session_state.chat_history = []
            st.session_state.ingested_files = []
            st.success("已清空知識庫與對話記錄。")
            st.rerun()
        except Exception as e:
            st.error(f"清空失敗: {e}")

    # 清空對話
    if st.button("🔄 清空對話記錄", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── 主畫面 ───────────────────────────────────────────────
sources = safe_sources()
source_count = len(sources)
history_count = len(st.session_state.chat_history)
source_label = f"{source_count} 份來源文件" if source_count else "尚未建立來源"

st.markdown(
    f"""
    <section class="hero">
        <div class="eyebrow">AI Knowledge Base Demo</div>
        <div class="hero-grid">
            <div>
                <h1>食品法規 AI 問答平台</h1>
                <p>
                    將 PDF 轉成可追溯的知識庫，結合法規條文切割、案例表格重組、
                    Hybrid Search 與 MultiQuery 檢索，讓每一次回答都回到可驗證的文件脈絡。
                </p>
                <div class="hero-actions">
                    <span class="pill">{mode_label}</span>
                    <span class="pill">LLM · {llm_name}</span>
                    <span class="pill">Embedding · {emb_display}</span>
                </div>
            </div>
            <aside class="signal-panel">
                <div class="label">Knowledge Base</div>
                <div class="value">{source_count}</div>
                <div class="detail">{source_label} · {history_count} 則對話紀錄 · ChromaDB local index</div>
            </aside>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-grid">
        <div class="info-card">
            <strong>Evidence-first retrieval</strong>
            <span>BM25 關鍵字、向量相似度與 MultiQuery 共同召回，降低只靠單一路徑檢索的盲點。</span>
        </div>
        <div class="info-card">
            <strong>Legal-aware chunking</strong>
            <span>法規文件依章、節、條注入上下文，讓條文回答保留原始層級與語意位置。</span>
        </div>
        <div class="info-card">
            <strong>Case table intelligence</strong>
            <span>裁罰案件表會重組產品、來源、違規情節、商號與金額，方便直接追問。</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not sources:
    st.markdown(
        """
        <div class="chat-shell">
            <div class="chat-inner">
                上傳 PDF 後即可開始提問。若要展示既有 demo 知識庫，請確認部署環境包含
                <code>chroma_db</code> 並使用相同 Embedding 設定。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 顯示對話歷史
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 使用者輸入
with st.container(key="ask_dock"):
    st.markdown(
        '<div class="ask-dock-title">Ask The Knowledge Base</div>',
        unsafe_allow_html=True,
    )
    with st.form("ask_knowledge_base", clear_on_submit=True):
        prompt = st.text_input(
            "請輸入你的問題",
            placeholder="例如：健康食品廣告宣稱療效會違反哪一條？",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", use_container_width=True)

if submitted and prompt.strip():
    prompt = prompt.strip()
    # 檢查知識庫是否有內容
    try:
        sources = get_all_sources()
        has_data = len(sources) > 0
    except Exception:
        has_data = False

    if not has_data:
        st.warning("⚠️ 知識庫為空，請先在左側上傳 PDF 文件。")
    else:
        # 顯示使用者訊息
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 取得 AI 回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    response = ask(prompt)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                except Exception as e:
                    error_msg = f"❌ 生成回答失敗: {e}"
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
                    st.rerun()
