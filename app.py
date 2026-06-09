"""
RAG PDF 知識庫問答系統 — Streamlit 主介面
"""

import sys
from pathlib import Path

# Python 3.14+ 不再自動將腳本目錄加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import streamlit as st
import tempfile
import os
import time
import uuid

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
        --chat-input-bg: #dde3df;
        --chat-input-border: #89958e;
        --chat-input-text: #1b2a24;
        --chat-input-placeholder: #55615b;
        --chat-input-button: #0f4a3a;
        --chat-input-focus: #2f7d62;
        --chat-input-focus-shadow: rgba(85, 97, 91, 0.28);
        --shadow: 0 18px 45px rgba(28, 39, 31, 0.10);
    }

    html, body {
        font-family: ui-serif, "Iowan Old Style", "Palatino Linotype", "Noto Serif TC", Georgia, serif;
        background:
            linear-gradient(90deg, rgba(21, 32, 24, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(21, 32, 24, 0.03) 1px, transparent 1px),
            linear-gradient(118deg, rgba(37, 106, 77, 0.12) 0%, transparent 34%),
            linear-gradient(145deg, #f8f9f5 0%, #eef4ef 48%, #f7f4ed 100%);
        background-size: 44px 44px, 44px 44px, auto, auto;
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
        padding: 2.35rem 2.5rem 12rem;
    }

    [data-testid="stSidebar"] {
        background: #fbfbf6;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.6rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--ink);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin: 0.18rem 0;
        line-height: 1.45;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        letter-spacing: 0;
        color: var(--green-dark);
    }

    [data-testid="stSidebar"] h1 {
        margin: 0 0 0.18rem;
        font-size: 1.45rem;
        line-height: 1.2;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        margin: 0.55rem 0 0.25rem;
        font-size: 1rem;
        line-height: 1.25;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.55rem 0;
    }

    [data-testid="stSidebar"] .stAlert {
        min-height: 42px;
        border-radius: 8px;
        border: 1px solid rgba(37, 106, 77, 0.22);
        background: rgba(37, 106, 77, 0.08) !important;
        background-color: rgba(37, 106, 77, 0.08) !important;
        padding: 0.45rem 0.6rem;
    }

    [data-testid="stSidebar"] .stAlert > div,
    [data-testid="stSidebar"] .stAlert [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] .stAlert [data-testid="stMarkdownContainer"] p {
        display: block;
        min-height: 0;
        margin: 0;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        margin-top: 0.12rem;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        margin: 0.15rem 0 0.45rem;
        background: rgba(37, 106, 77, 0.08) !important;
        background-color: rgba(37, 106, 77, 0.08) !important;
        background-image: none !important;
        color: var(--green-dark) !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        color: var(--green-dark) !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        color: var(--green-dark) !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        min-height: 118px;
        padding: 0.65rem;
        border-color: rgba(37, 106, 77, 0.22) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] ul {
        margin: 0.25rem 0 0.45rem;
        padding-left: 1.1rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
        margin: 0.08rem 0;
        line-height: 1.35;
    }

    [data-testid="stSidebar"] .stButton {
        margin: 0.2rem 0;
    }

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
        background-image: none !important;
        color: var(--green-dark) !important;
    }

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: rgba(37, 106, 77, 0.08) !important;
        background-color: rgba(37, 106, 77, 0.08) !important;
    }

    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
        background: transparent !important;
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        border-color: rgba(37, 106, 77, 0.22) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover * {
        background-image: none !important;
        color: var(--green) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background: rgba(37, 106, 77, 0.12) !important;
        background-color: rgba(37, 106, 77, 0.12) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover * {
        background: transparent !important;
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        border-color: var(--green) !important;
    }

    [data-testid="stSidebarHeader"] {
        min-height: 42px;
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
        min-height: 250px;
        padding: clamp(1.55rem, 3vw, 2.8rem);
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
        margin-bottom: 0.75rem;
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
        font-size: clamp(1.9rem, 3.6vw, 3.35rem);
        line-height: 1.12;
        letter-spacing: 0;
        font-weight: 780;
        word-break: keep-all;
        overflow-wrap: normal;
    }

    .hero p {
        max-width: 760px;
        margin: 0.9rem 0 0;
        color: #38463d;
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: clamp(1rem, 1.7vw, 1.2rem);
        line-height: 1.68;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.75fr);
        gap: 1.1rem;
        margin-top: 0.75rem;
        align-items: center;
    }

    .signal-panel {
        padding: 0.9rem;
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
        font-size: 2.05rem;
        line-height: 1;
        font-weight: 760;
    }

    .signal-panel .detail {
        margin-top: 0.55rem;
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

    .chat-shell.empty-state {
        margin: 1.85rem 0 1.35rem;
    }

    .chat-inner {
        padding: 1.1rem 1.2rem;
        border-radius: 7px;
        background: rgba(255, 255, 250, 0.82);
        color: var(--muted);
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-size: 0.95rem;
    }

    .chat-shell.empty-state .chat-inner,
    .chat-shell.empty-state .chat-inner code {
        color: #34453c !important;
    }

    .chat-shell.empty-state .chat-inner code {
        background: rgba(37, 106, 77, 0.08) !important;
        background-color: rgba(37, 106, 77, 0.08) !important;
        border: 1px solid rgba(37, 106, 77, 0.18);
        border-radius: 4px;
        padding: 0.08em 0.35em;
    }

    .main [data-testid="stAlert"],
    [data-testid="stMainBlockContainer"] [data-testid="stAlert"] {
        margin: 1.15rem 0 1.35rem;
        color: #3f3418 !important;
    }

    .main [data-testid="stAlert"] [data-testid="stAlertContainer"],
    [data-testid="stMainBlockContainer"] [data-testid="stAlertContainer"] {
        border: 1px solid rgba(176, 129, 30, 0.30);
        border-radius: 8px;
        background: rgba(255, 246, 188, 0.62) !important;
        background-color: rgba(255, 246, 188, 0.62) !important;
        color: #3f3418 !important;
    }

    .main [data-testid="stAlert"] *,
    .main [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    .main [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMainBlockContainer"] [data-testid="stAlert"] *,
    [data-testid="stMainBlockContainer"] [data-testid="stAlertContainer"] *,
    [data-testid="stMainBlockContainer"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stMainBlockContainer"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #3f3418 !important;
    }

    [class*="st-key-chat-history"],
    [class*="st-key-chat_history"],
    [class*="st-key-live_chat_turn"] {
        margin-top: 1.6rem;
    }

    .stChatMessage {
        border: 1px solid rgba(21, 32, 24, 0.10);
        border-radius: 8px;
        background: rgba(255, 255, 250, 0.82);
        background-color: rgba(255, 255, 250, 0.82) !important;
        background-image: none !important;
        box-shadow: 0 10px 24px rgba(28, 39, 31, 0.055);
        padding-right: 1.25rem;
    }

    .stChatMessage [data-testid="stChatMessageContent"],
    .stChatMessage [data-testid="stMarkdownContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] {
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        line-height: 1.75;
    }

    .stChatMessage [data-testid="stChatMessageAvatar"] {
        background: rgba(37, 106, 77, 0.08) !important;
        border: 1px solid rgba(37, 106, 77, 0.14);
    }

    .stChatMessage [data-testid="stSpinner"] {
        color: var(--green-dark) !important;
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
        font-weight: 800;
    }

    .stSpinner [data-testid="stSpinnerIcon"],
    [data-testid="stSpinner"] [data-testid="stSpinnerIcon"] {
        color: var(--green-dark) !important;
        border-color: rgba(18, 63, 49, 0.18) !important;
        border-top-color: var(--green-dark) !important;
        border-right-color: var(--green-dark) !important;
        opacity: 1 !important;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] hr {
        height: 1px;
        margin: 1.25rem 0;
        border: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(18, 63, 49, 0.34),
            rgba(199, 154, 58, 0.30),
            transparent
        ) !important;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] blockquote {
        margin: 1rem 0;
        padding: 0.55rem 0.75rem 0.55rem 1rem;
        border-left: 3px solid #a9b0ac !important;
        background: rgba(127, 135, 130, 0.055);
        color: var(--ink);
    }

    .stChatMessage [data-testid="stMarkdownContainer"] table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.92rem;
        border: 1px solid rgba(18, 63, 49, 0.34) !important;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] th,
    .stChatMessage [data-testid="stMarkdownContainer"] td {
        border: 1px solid rgba(18, 63, 49, 0.34) !important;
        padding: 0.45em 0.75em;
        text-align: left;
        color: var(--ink) !important;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] th {
        background: rgba(18, 63, 49, 0.13) !important;
        font-weight: 700;
        color: var(--green-dark) !important;
    }

    .stChatMessage [data-testid="stMarkdownContainer"] tr:nth-child(even) td {
        background: rgba(37, 106, 77, 0.055) !important;
    }

    [data-testid="stMarkdownContainer"] ul > li > code,
    [data-testid="stCaptionContainer"] > p > code {
        background:
            linear-gradient(90deg, rgba(21, 32, 24, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(21, 32, 24, 0.03) 1px, transparent 1px),
            rgba(247, 248, 243, 0.92) !important;
        background-size: 44px 44px, 44px 44px, auto !important;
        border: 1px solid rgba(21, 32, 24, 0.10);
        border-radius: 4px;
        padding: 0.1em 0.4em;
        font-size: 0.88em;
        color: var(--ink) !important;
    }

    .stBottom,
    .stBottom > div,
    .stBottom > div > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        pointer-events: none;
    }

    .stBottom::before,
    .stBottom::after,
    .stBottom > div::before,
    .stBottom > div::after {
        display: none !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        content: none !important;
    }

    .stBottom * {
        box-shadow: none !important;
    }

    [data-testid="stBottomBlockContainer"] > div,
    [data-testid="stBottomBlockContainer"] [data-testid="stVerticalBlock"],
    [data-testid="stBottomBlockContainer"] [data-testid="stVerticalBlock"] > div {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
        pointer-events: none;
    }

    .stBottom [data-testid="stChatInput"],
    [data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        pointer-events: auto;
    }

    [data-testid="stChatInput"] > div {
        background: var(--chat-input-bg) !important;
        background-color: var(--chat-input-bg) !important;
        background-image: none !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] textarea {
        border-radius: 8px !important;
        border-color: var(--chat-input-border) !important;
        background: var(--chat-input-bg) !important;
        background-color: var(--chat-input-bg) !important;
        color: var(--chat-input-text) !important;
        caret-color: var(--chat-input-text) !important;
        font-family: ui-sans-serif, "Noto Sans TC", "Helvetica Neue", sans-serif;
    }

    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"] > div > div {
        background: var(--chat-input-bg) !important;
        background-color: var(--chat-input-bg) !important;
        background-image: none !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--chat-input-focus) !important;
        background: var(--chat-input-bg) !important;
        background-color: var(--chat-input-bg) !important;
        box-shadow: 0 0 0 2px var(--chat-input-focus-shadow) !important;
        outline: none !important;
    }

    [data-testid="stChatInput"]:focus-within [data-baseweb="textarea"],
    [data-testid="stChatInput"]:focus-within [data-baseweb="textarea"] > div,
    [data-testid="stChatInput"]:focus-within [data-baseweb="textarea"] > div > div {
        border-color: var(--chat-input-focus) !important;
        background: var(--chat-input-bg) !important;
        background-color: var(--chat-input-bg) !important;
        background-image: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--chat-input-placeholder) !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] > button,
    [data-testid="stChatInput"] button {
        border-radius: 8px !important;
        background: var(--chat-input-button) !important;
        background-color: var(--chat-input-button) !important;
        color: #fffefa !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: var(--chat-input-focus) !important;
        background-color: var(--chat-input-focus) !important;
        color: #fffefa !important;
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

    [data-testid="stBaseButton-secondary"] {
        background:
            linear-gradient(90deg, rgba(21, 32, 24, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(21, 32, 24, 0.03) 1px, transparent 1px),
            rgba(247, 248, 243, 0.92) !important;
        background-size: 44px 44px, 44px 44px, auto !important;
    }

    [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] {
        background:
            linear-gradient(90deg, rgba(21, 32, 24, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(21, 32, 24, 0.03) 1px, transparent 1px),
            rgba(247, 248, 243, 0.92) !important;
        background-size: 44px 44px, 44px 44px, auto !important;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(37, 106, 77, 0.32);
        border-radius: 8px;
        background: rgba(37, 106, 77, 0.045);
        padding: 0.4rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background:
            linear-gradient(90deg, rgba(21, 32, 24, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(21, 32, 24, 0.03) 1px, transparent 1px),
            rgba(247, 248, 243, 0.92) !important;
        background-size: 44px 44px, 44px 44px, auto !important;
        border-radius: 6px;
    }

    div[data-baseweb="select"] > div,
    textarea,
    input {
        border-radius: 8px !important;
        color: var(--ink) !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #fffefa !important;
        border-color: rgba(18, 63, 49, 0.22) !important;
        color: var(--ink) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        color: var(--ink) !important;
        fill: var(--ink) !important;
    }

    div[data-baseweb="popover"] div[data-baseweb="menu"],
    div[data-baseweb="popover"] ul {
        background: #fffefa !important;
        border: 1px solid rgba(18, 63, 49, 0.18) !important;
        border-radius: 8px !important;
        color: var(--ink) !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"] {
        background: #fffefa !important;
        background-color: #fffefa !important;
        color: var(--ink) !important;
    }

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"],
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [role="option"]:hover > div,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"],
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] > div,
    div[data-baseweb="popover"] [role="option"][data-highlighted="true"],
    div[data-baseweb="popover"] [role="option"][data-highlighted="true"] > div {
        background: rgba(37, 106, 77, 0.10) !important;
        background-color: rgba(37, 106, 77, 0.10) !important;
        color: var(--green-dark) !important;
    }

    div[data-baseweb="popover"] [role="option"] *,
    div[data-baseweb="popover"] [role="option"]:hover *,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] *,
    div[data-baseweb="popover"] [role="option"][data-highlighted="true"] * {
        color: var(--green-dark) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1 !important;
    }

    [data-testid="stChatMessageContent"] {
        color: var(--ink) !important;
    }

    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        visibility: hidden;
    }

    [data-testid="stHeader"] {
        background:
            linear-gradient(90deg, rgba(21, 32, 24, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(21, 32, 24, 0.03) 1px, transparent 1px),
            rgba(247, 248, 243, 0.92) !important;
        background-size: 44px 44px, 44px 44px, auto !important;
        border-bottom: 1px solid rgba(21, 32, 24, 0.08);
        box-shadow: none !important;
        backdrop-filter: blur(12px);
    }

    @media (max-width: 1100px) {
        .hero-grid {
            grid-template-columns: 1fr;
            align-items: stretch;
        }
    }

    @media (max-width: 820px) {
        .main .block-container {
            padding: 1.35rem 1rem 12rem;
        }

        .hero {
            min-height: 230px;
            padding: 1.2rem;
        }

        .info-grid {
            grid-template-columns: 1fr;
        }

    }
</style>
""", unsafe_allow_html=True)

CHAT_HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", ".chat_sessions"))


def get_chat_session_id() -> str:
    """取得目前瀏覽器頁籤的對話 session id，刷新頁面後仍可沿用。"""
    if "chat_session_id" in st.session_state:
        return st.session_state.chat_session_id

    raw_session_id = st.query_params.get("chat_session", "")
    if isinstance(raw_session_id, list):
        raw_session_id = raw_session_id[0] if raw_session_id else ""

    session_id = str(raw_session_id)
    if len(session_id) != 32 or not all(c in "0123456789abcdef" for c in session_id):
        session_id = uuid.uuid4().hex
        st.query_params["chat_session"] = session_id

    st.session_state.chat_session_id = session_id
    return session_id


def get_chat_history_path(session_id: str) -> Path:
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return CHAT_HISTORY_DIR / f"{session_id}.json"


def load_chat_history(session_id: str) -> list[dict[str, str]]:
    path = get_chat_history_path(session_id)
    if not path.exists():
        return []

    try:
        raw_history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    history: list[dict[str, str]] = []
    for msg in raw_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def save_chat_history() -> None:
    session_id = get_chat_session_id()
    path = get_chat_history_path(session_id)
    path.write_text(
        json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_chat_history() -> None:
    st.session_state.chat_history = []
    path = get_chat_history_path(get_chat_session_id())
    if path.exists():
        path.unlink()


# ── Session State 初始化 ─────────────────────────────────
chat_session_id = get_chat_session_id()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(chat_session_id)
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
            st.markdown(f"- {src}")
    else:
        st.caption("知識庫目前為空，請上傳 PDF 文件。")

    st.markdown("---")

    # 清空知識庫
    if st.button("⌫ 清空知識庫", use_container_width=True, type="secondary"):
        try:
            delete_collection()
            clear_chat_history()
            st.session_state.ingested_files = []
            st.success("已清空知識庫與對話記錄。")
            st.rerun()
        except Exception as e:
            st.error(f"清空失敗: {e}")

    # 清空對話
    if st.button("↺ 清空對話記錄", use_container_width=True):
        clear_chat_history()
        st.rerun()


# ── 主畫面 ───────────────────────────────────────────────
sources = safe_sources()
source_count = len(sources)
history_count = sum(
    1 for msg in st.session_state.chat_history if msg.get("role") == "user"
)
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
        <div class="chat-shell empty-state">
            <div class="chat-inner">
                上傳 PDF 後即可開始提問。若要展示既有 demo 知識庫，請確認部署環境包含
                <code>chroma_db</code> 並使用相同 Embedding 設定。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 顯示對話歷史
if st.session_state.chat_history:
    with st.container(key="chat_history"):
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# 使用者輸入
prompt = st.chat_input(
    "例如：健康食品廣告宣稱療效會違反哪一條？",
    key="ask_knowledge_base",
)

if prompt and prompt.strip():
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
        with st.container(key="live_chat_turn"):
            # 顯示使用者訊息
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            save_chat_history()
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
                        save_chat_history()
                        st.rerun()
                    except Exception as e:
                        error_msg = f"❌ 生成回答失敗: {e}"
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": error_msg}
                        )
                        save_chat_history()
                        st.rerun()
