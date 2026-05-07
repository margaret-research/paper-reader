import streamlit as st
import fitz  # PyMuPDF
import anthropic
from anthropic.types import TextBlock
import requests
import tempfile
import os
import json
import re
from datetime import datetime
from pathlib import Path

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Reader",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* サイドバーのボタンを左揃えに */
.stButton > button { text-align: left !important; }

/* チャットエリアの最大高さ */
section[data-testid="stMain"] { padding-top: 1rem; }

/* セクションヘッダー */
.sec-badge {
    background: #1565C0;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* フッター非表示 */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Session state の初期化 ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "pdf_path": None,
        "pages_text": {},
        "total_pages": 0,
        "paper_title": "",
        "sections": [],
        "current_idx": 0,
        "section_explained": False,
        "messages": [],       # 画面表示用 [{role, content}]
        "api_history": [],    # Claude API用 [{role, content}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Claude クライアント ────────────────────────────────────────────────────────
def get_client():
    key = st.session_state.get("api_key", "")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)

# ─── PDF ユーティリティ ────────────────────────────────────────────────────────
def load_pdf_url(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(r.content)
    tmp.close()
    return tmp.name

def extract_pages(pdf_path: str) -> tuple[dict, int]:
    doc = fitz.open(pdf_path)
    n = len(doc)
    pages = {i + 1: doc[i].get_text() for i in range(n)}
    return pages, n

def detect_sections(pages_text: dict, client: anthropic.Anthropic) -> dict:
    sample = "\n\n".join(
        f"[Page {p}]\n{t[:600]}"
        for p, t in list(pages_text.items())[:10]
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": (
                "以下の論文テキストからセクション構成を検出し、必ずJSON形式のみで返してください。"
                "他の文字は一切含めないでください。\n\n"
                '{"title":"論文タイトル","sections":[{"name":"Abstract","start_page":1,"end_page":1},...]}\n\n'
                f"論文テキスト:\n{sample}"
            ),
        }],
    )
    raw = next((b.text for b in resp.content if isinstance(b, TextBlock)), "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"title": "論文", "sections": [
        {"name": f"Page {p}", "start_page": p, "end_page": p}
        for p in pages_text
    ]}

def build_sections(raw: dict, pages_text: dict) -> list:
    sections = []
    for s in raw.get("sections", []):
        content = "".join(
            pages_text.get(p, "")
            for p in range(s["start_page"], s["end_page"] + 1)
        )
        sections.append({**s, "content": content})
    return sections

# ─── Claude 先生 ───────────────────────────────────────────────────────────────
SYSTEM_TEACHER = """あなたは論文解説の先生です。以下のルールで解説してください：
- 専門用語は必ず日本語で丁寧に説明する
- 数式はテキストアートや具体例でイメージを伝える
- マークダウン（見出し・表・コードブロック）で整理する
- 各セクション説明の最後は「質問があればどうぞ！理解できたら「次へ」ボタンで進んでください。」で締める
- 難しい概念はアナロジーや日常の例えで説明する"""

def stream_explain(section_content: str, section_name: str, paper_title: str,
                   api_history: list, client: anthropic.Anthropic):
    messages = api_history + [{
        "role": "user",
        "content": f'論文「{paper_title}」のセクション「{section_name}」を解説してください。\n\n内容:\n{section_content[:3500]}',
    }]
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=SYSTEM_TEACHER,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

def stream_answer(question: str, api_history: list, client: anthropic.Anthropic):
    messages = api_history + [{"role": "user", "content": question}]
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_TEACHER,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

# ─── サイドバー ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Paper Reader")
    st.markdown("---")

    # ── APIキー入力 ──
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-ant-...",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.rerun()

    if not st.session_state.api_key:
        st.warning("APIキーを入力してください")
        st.stop()

    st.markdown("---")

    if not st.session_state.pdf_path:
        # ── 読み込みフォーム ──
        st.markdown("### 論文を読み込む")
        tab_url, tab_file = st.tabs(["URL", "ファイル"])

        with tab_url:
            url = st.text_input(
                "PDF URL",
                placeholder="https://arxiv.org/pdf/1706.03762.pdf",
                label_visibility="collapsed",
            )
            if st.button("読み込む", type="primary", use_container_width=True) and url:
                with st.spinner("ダウンロード中..."):
                    try:
                        path = load_pdf_url(url)
                        pages, total = extract_pages(path)
                        st.session_state.pdf_path = path
                        st.session_state.pages_text = pages
                        st.session_state.total_pages = total
                        st.rerun()
                    except Exception as e:
                        st.error(f"エラー: {e}")

        with tab_file:
            uploaded = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
            if uploaded:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(uploaded.read())
                tmp.close()
                pages, total = extract_pages(tmp.name)
                st.session_state.pdf_path = tmp.name
                st.session_state.pages_text = pages
                st.session_state.total_pages = total
                st.rerun()

    else:
        # ── セクション解析 ──
        if not st.session_state.sections:
            _client = get_client()
            if _client is None:
                st.error("APIキーが無効です")
                st.stop()
            with st.spinner("セクションを解析中..."):
                raw = detect_sections(st.session_state.pages_text, _client)
                st.session_state.paper_title = raw.get("title", "論文")
                st.session_state.sections = build_sections(raw, st.session_state.pages_text)
            st.rerun()

        # ── 論文タイトル ──
        st.markdown(f"**{st.session_state.paper_title}**")
        st.caption(f"全 {st.session_state.total_pages} ページ / {len(st.session_state.sections)} セクション")
        st.markdown("---")

        # ── 目次 ──
        st.markdown("### 目次")
        for i, sec in enumerate(st.session_state.sections):
            is_cur = i == st.session_state.current_idx
            label = ("▶ " if is_cur else "　") + sec["name"]
            btn_type = "primary" if is_cur else "secondary"
            if st.button(label, key=f"toc_{i}", use_container_width=True, type=btn_type):
                if i != st.session_state.current_idx:
                    st.session_state.current_idx = i
                    st.session_state.section_explained = False
                    st.session_state.messages = []
                    # api_history はセッション全体で引き継ぐ
                    st.rerun()

        st.markdown("---")

        # ── 会話を保存 ──
        if st.session_state.api_history:
            if st.button("💾 会話を保存", use_container_width=True, type="primary"):
                save_dir = Path(__file__).parent / "sessions"
                save_dir.mkdir(exist_ok=True)
                date_str = datetime.now().strftime("%Y-%m-%d")
                safe_title = re.sub(r'[\\/:*?"<>|]', "-", st.session_state.paper_title)[:40]
                save_path = save_dir / f"{date_str}-{safe_title}.md"

                lines = [
                    f"# {st.session_state.paper_title}",
                    f"",
                    f"**保存日**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    f"**全ページ数**: {st.session_state.total_pages}",
                    f"",
                    "---",
                    "",
                ]
                role_map = {"user": "🧑‍💻 質問", "assistant": "👩‍🏫 先生"}
                for msg in st.session_state.api_history:
                    role_label = role_map.get(msg["role"], msg["role"])
                    lines.append(f"## {role_label}")
                    lines.append("")
                    lines.append(msg["content"])
                    lines.append("")
                    lines.append("---")
                    lines.append("")

                save_path.write_text("\n".join(lines), encoding="utf-8")
                st.success(f"保存しました！\n`sessions/{save_path.name}`")

        st.markdown("---")
        if st.button("最初からやり直す", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ─── メインエリア ──────────────────────────────────────────────────────────────
if not st.session_state.pdf_path:
    st.markdown("## Paper Reader へようこそ")
    st.info("サイドバーから論文を読み込んでください。\nURL（arXiv等）またはPDFファイルに対応しています。")
    st.markdown("""
**使い方:**
1. サイドバーにarXivのPDF URLを貼る（例: `https://arxiv.org/pdf/1706.03762.pdf`）
2. 先生がセクションごとに解説
3. わからなければ質問 → 理解できたら「次へ」ボタン
""")

elif not st.session_state.sections:
    st.info("セクションを解析中です...")

else:
    sections = st.session_state.sections
    idx = st.session_state.current_idx
    current = sections[idx]

    # ── ヘッダー ──
    col_title, col_nav = st.columns([3, 1])
    with col_title:
        st.markdown(f"### {current['name']}")
        st.caption(f"p.{current['start_page']}–{current['end_page']}　｜　{idx+1} / {len(sections)} セクション")
    with col_nav:
        st.markdown("<br>", unsafe_allow_html=True)
        can_next = idx < len(sections) - 1
        if st.button("次のセクション ▶", type="primary", disabled=not can_next, use_container_width=True):
            st.session_state.current_idx += 1
            st.session_state.section_explained = False
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # ── チャット表示 ──
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="👩‍🏫"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg["content"])

    # ── セクション自動解説 ──
    _client = get_client()
    if _client is None:
        st.error("APIキーが無効です。サイドバーで入力してください。")
        st.stop()

    if not st.session_state.section_explained:
        with st.chat_message("assistant", avatar="👩‍🏫"):
            placeholder = st.empty()
            full = ""
            for chunk in stream_explain(
                current["content"],
                current["name"],
                st.session_state.paper_title,
                st.session_state.api_history,
                _client,
            ):
                full += chunk
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)

        st.session_state.messages.append({"role": "assistant", "content": full})
        st.session_state.api_history.append({"role": "assistant", "content": full})
        st.session_state.section_explained = True
        st.rerun()

    # ── 質問入力 ──
    question = st.chat_input("質問を入力してください（「次へ」と入力すると次のセクションへ）")
    if question:
        # 「次へ」で進める
        if question.strip() in ("次へ", "次", "next", "Next"):
            if idx < len(sections) - 1:
                st.session_state.current_idx += 1
                st.session_state.section_explained = False
                st.session_state.messages = []
                st.rerun()
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.api_history.append({"role": "user", "content": question})

            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="👩‍🏫"):
                placeholder = st.empty()
                full = ""
                for chunk in stream_answer(
                    question,
                    st.session_state.api_history,
                    _client,
                ):
                    full += chunk
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)

            st.session_state.messages.append({"role": "assistant", "content": full})
            st.session_state.api_history.append({"role": "assistant", "content": full})
            st.rerun()
