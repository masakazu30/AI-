"""The Council of Legends — 伝説的投資家によるAI議論アプリ（Streamlit UI）。

ユーザーは米国株のティッカーを指定するだけ。
必要な財務・株価・マクロ情報は自動取得され、AI投資家たちが議論し、
投資すべきか否かを評価する。
"""

from __future__ import annotations

import os

import streamlit as st

# Streamlit Community Cloud の「Secrets」に入れたAPIキーを環境変数として使えるようにする。
# （ローカル実行時は secrets が無いので、この処理は静かにスキップされる）
try:
    for _key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # noqa: BLE001 — secrets未設定時のエラーは無視
    pass

from council.data import build_dossier, fetch_company, format_large_number
from council.debate import run_debate
from council.investors import INVESTORS
from council.llm import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    LLMClient,
    LLMError,
    resolve_api_key,
)
from council.scoring import council_score, grades_to_text, score_all

st.set_page_config(page_title="The Council of Legends", layout="wide", page_icon="🏛️")


# --------------------------------------------------------------------------- #
# サイドバー
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🏛️ 評議会の設定")

    provider_label = st.radio(
        "AIエンジン（投資家の頭脳）",
        ["Claude (Anthropic)", "Gemini (Google)"],
        help="どちらのLLMでAI投資家を動かすか選択します。",
    )
    provider = "claude" if provider_label.startswith("Claude") else "gemini"

    default_model = (
        DEFAULT_CLAUDE_MODEL if provider == "claude" else DEFAULT_GEMINI_MODEL
    )
    key_env_name = "ANTHROPIC_API_KEY" if provider == "claude" else "GEMINI_API_KEY"

    env_key_present = bool(resolve_api_key(provider))
    api_key_input = st.text_input(
        f"{provider_label} APIキー",
        type="password",
        help=f"未入力の場合は環境変数 {key_env_name} を使用します。",
        placeholder="環境変数を使用" if env_key_present else "APIキーを入力",
    )
    model_name = st.text_input("モデル名（任意）", value=default_model)

    st.divider()
    ticker_symbol = st.text_input(
        "ティッカーシンボル", value="NVDA", help="例: NVDA, AAPL, KO, BRK-B"
    ).upper()

    do_rebuttal = st.checkbox(
        "円卓討論（反論ラウンド）を行う",
        value=True,
        help="OFFにすると初期見解→最終評価のみ。API消費を抑えられます。",
    )

    analyze = st.button("⚖️ 評議会を招集する", type="primary", use_container_width=True)

    st.divider()
    st.caption("評議会メンバー")
    for inv in INVESTORS:
        st.markdown(f"{inv.emoji} **{inv.name}** — {inv.tagline}")


# --------------------------------------------------------------------------- #
# セッション状態
# --------------------------------------------------------------------------- #
if "company" not in st.session_state:
    st.session_state.company = None
    st.session_state.grades = None
    st.session_state.debate = None
    st.session_state.dossier = ""


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
st.title("The Council of Legends")
st.caption("伝説の米国株投資家たちが、あなたの選んだ銘柄を議論する。")

if analyze and ticker_symbol:
    # 1) データ自動取得
    with st.spinner(f"{ticker_symbol} のデータと市場環境を自動取得中…"):
        try:
            company = fetch_company(ticker_symbol)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    # 2) 定量スコアリング
    grades = score_all(company.metrics, company.macro)
    dossier = build_dossier(company)

    st.session_state.company = company
    st.session_state.grades = grades
    st.session_state.dossier = dossier
    st.session_state.debate = None  # 新規分析でリセット

    for w in company.warnings:
        st.warning(w)

    # 3) AI議論
    try:
        client = LLMClient(provider=provider, api_key=api_key_input, model=model_name)
    except LLMError as exc:
        st.error(str(exc))
        st.info("定量スコアのみ表示します。AI議論にはAPIキーが必要です。")
        client = None

    if client is not None:
        status = st.status("評議会を招集しています…", expanded=True)

        def _progress(msg: str) -> None:
            status.update(label=msg)
            status.write(msg)

        try:
            debate = run_debate(
                client,
                dossier,
                grades,
                grades_to_text(grades),
                do_rebuttal=do_rebuttal,
                progress=_progress,
            )
            st.session_state.debate = debate
            status.update(label="議論が完了しました", state="complete")
        except LLMError as exc:
            status.update(label="エラー", state="error")
            st.error(str(exc))


# --------------------------------------------------------------------------- #
# 結果表示
# --------------------------------------------------------------------------- #
company = st.session_state.company
grades = st.session_state.grades

if company is not None and grades is not None:
    m = company.metrics
    st.header(f"{company.name} ({company.ticker})")

    total, norm = council_score(grades)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("評議会スコア", f"{norm:.0f} / 100")
    c2.metric("現在株価", f"${m['price']:.2f}" if m["price"] else "N/A")
    c3.metric("時価総額", format_large_number(m["market_cap"]))
    c4.metric("PER(実績)", f"{m['trailing_pe']:.1f}" if m["trailing_pe"] else "N/A")

    debate = st.session_state.debate

    tab_verdict, tab_debate, tab_quant, tab_data = st.tabs(
        ["🏛️ 最終評価", "💬 議論の記録", "📊 定量スコア", "📄 取得データ"]
    )

    # --- 最終評価 ---
    with tab_verdict:
        if debate and debate.verdict:
            st.markdown(debate.verdict)
        elif debate and debate.error:
            st.error(debate.error)
        else:
            st.info("AI議論を実行すると、ここに評議会の最終評価が表示されます。")
        st.caption(
            "※ 本アプリの出力はAIによる分析であり、投資助言ではありません。"
            "最終的な投資判断はご自身の責任で行ってください。"
        )

    # --- 議論の記録 ---
    with tab_debate:
        if debate:
            st.subheader("初期見解")
            for s in debate.openings:
                with st.expander(f"{s.emoji} {s.investor_name}", expanded=False):
                    st.markdown(s.text)
            if debate.rebuttals:
                st.subheader("円卓討論（反論・補強）")
                for s in debate.rebuttals:
                    with st.expander(f"{s.emoji} {s.investor_name}", expanded=False):
                        st.markdown(s.text)
        else:
            st.info("AI議論を実行すると、各投資家の発言がここに表示されます。")

    # --- 定量スコア ---
    with tab_quant:
        st.caption("各投資家のロジックによる客観的な採点（AIに依存しない決定論的スコア）")
        for inv in INVESTORS:
            g = grades[inv.key]
            col_a, col_b = st.columns([1, 4])
            col_a.markdown(f"### {inv.emoji}")
            col_b.markdown(
                f"**{inv.name}** — `{g.grade}`  \n{g.reason}"
            )
        st.divider()
        st.metric("合計スコア", f"{total} / {5.0 * len(grades):.0f}")

    # --- 取得データ ---
    with tab_data:
        st.caption("AI投資家に渡された銘柄ドシエ（自動収集された情報）")
        st.code(st.session_state.dossier, language="markdown")

else:
    st.info(
        "サイドバーでティッカーを入力し、「評議会を招集する」を押してください。\n\n"
        "AI投資家が必要な情報を自動で集め、その銘柄について議論します。"
    )
