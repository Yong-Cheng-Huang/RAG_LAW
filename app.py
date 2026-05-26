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
    page_title="📚 RAG PDF 問答系統",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自訂 CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* 主內容區域寬度控制（sidebar 維持靠左）*/
    .main .block-container {
        max-width: 900px;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* 聊天訊息容器優化 */
    .chat-message {
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid #1a5276; /* 加入邊框標識 */
        background-color: #f9f9f9;
    }

    /* 來源標籤優化 */
    .source-tag {
        display: inline-block;
        background: #e8f4f8;
        color: #1a5276;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        border: 1px solid #d1e7ed;
        transition: all 0.3s ease;
    }
    
    .source-tag:hover {
        background: #1a5276;
        color: white;
    }

    /* 隱藏預設 Streamlit 頁尾 */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session State 初始化 ─────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ── 側邊欄 ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 系統設定")

    st.markdown("---")

    # 模式顯示
    if settings.LLM_MODE == "ollama":
        mode_label = "🟢 本地模式 (Ollama)"
        llm_name = settings.OLLAMA_MODEL
    elif settings.LLM_MODE == "gemini":
        mode_label = "✨ 雲端模式 (Gemini)"
        llm_name = settings.GEMINI_MODEL
    else:
        mode_label = "☁️ 雲端模式 (OpenAI)"
        llm_name = settings.OPENAI_MODEL
    st.info(f"**目前模式：** {mode_label}")
    st.caption(f"**LLM：** `{llm_name}`")
    emb_display = (
        f"gemini / {settings.GEMINI_EMBEDDING_MODEL}"
        if settings.EMBEDDING_MODE == "gemini"
        else f"ollama / {settings.EMBEDDING_MODEL}"
    )
    st.caption(f"**Embedding：** `{emb_display}`")

    st.markdown("---")

    # PDF 上傳
    st.subheader("📄 上傳 PDF")
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
    st.subheader("📂 知識庫內容")
    try:
        sources = get_all_sources()
        if sources:
            for src in sources:
                st.markdown(f"- 📄 `{src}`")
        else:
            st.caption("知識庫目前為空，請上傳 PDF 文件。")
    except Exception:
        st.caption("知識庫尚未初始化。")

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
st.title("📚 RAG PDF 知識庫問答系統")
st.caption("上傳 PDF → 建立向量索引 → 開始提問！使用 MultiQueryRetriever 提升召回率。")

# 顯示對話歷史
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 使用者輸入
if prompt := st.chat_input("請輸入你的問題..."):
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
                    st.markdown(response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"❌ 生成回答失敗: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
