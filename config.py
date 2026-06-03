"""
全域設定模組 — 讀取 .env 並提供統一的設定存取介面。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── LLM ──────────────────────────────────────────────
    LLM_MODE: str = os.getenv("LLM_MODE", "ollama")  # "ollama" | "openai" | "gemini"

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    # MultiQueryRetriever 專用 model（留空則與 OLLAMA_MODEL 相同）
    MULTIQUERY_MODEL: str = os.getenv("MULTIQUERY_MODEL", "")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Gemini LLM
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ── Embedding ────────────────────────────────────────
    # EMBEDDING_MODE: "ollama" | "gemini"
    EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "ollama")

    # Ollama embedding（預設 bge-m3）
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")

    # Gemini Embedding
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # ── ChromaDB ─────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "pdf_knowledge_base")

    # ── Splitter ─────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    # 啟用法規語意切割（條文邊界 + Contextual Header）
    LEGAL_CHUNK_MODE: bool = os.getenv("LEGAL_CHUNK_MODE", "true").lower() == "true"

    # ── Retriever ────────────────────────────────────────
    # 每個 retriever 各取幾份文件（最終取聯集去重）
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))
    # BM25 在 Hybrid Search 中的權重（0~1），剩餘給 Vector
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))

    # ── Paths ────────────────────────────────────────────
    PDF_DIR: Path = Path("./pdfs")


settings = Settings()
