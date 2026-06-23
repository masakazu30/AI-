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
    for _key in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "SEC_EDGAR_USER_AGENT",
    ):
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
    include_filings = st.checkbox(
        "SEC 10-K（リスク要因/MD&A）も取得",
        value=True,
        help="SEC EDGARから最新10-Kの定性情報を取得します。取得に少し時間がかかります。",
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
    spin_msg = f"{ticker_symbol} のデータ・市場環境"
    spin_msg += "・SEC 10-K を自動取得中…" if include_filings else " を自動取得中…"
    with st.spinner(spin_msg):
        try:
            company = fetch_company(ticker_symbol, include_filings=include_filings)
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

    d1, d2, d3, d4 = st.columns(4)
    if m.get("target_mean"):
        up = m.get("analyst_upside", 0.0)
        d1.metric(
            "アナリスト目標株価(平均)",
            f"${m['target_mean']:.2f}",
            f"{up * 100:+.1f}%",
        )
    else:
        d1.metric("アナリスト目標株価(平均)", "N/A")
    d2.metric(
        "目標株価レンジ",
        f"${m.get('target_low', 0):.0f}〜${m.get('target_high', 0):.0f}"
        if m.get("target_high")
        else "N/A",
    )
    d3.metric("予想PER", f"{m['forward_pe']:.1f}" if m.get("forward_pe") else "N/A")
    d4.metric(
        "配当利回り",
        f"{m['dividend_yield'] * 100:.2f}%" if m.get("dividend_yield") else "N/A",
    )
    st.caption("🎯 評議会としての目標株価は「最終評価」タブに表示されます。")

    debate = st.session_state.debate

    tab_verdict, tab_debate, tab_quant, tab_filings, tab_data, tab_claude = st.tabs(
        ["🏛️ 最終評価", "💬 議論の記録", "📊 定量スコア", "📑 SEC 10-K",
         "📄 取得データ", "🤖 Claudeへ貼付"]
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

    # --- SEC 10-K ---
    with tab_filings:
        f = company.filings or {}
        if not f:
            st.info("SEC 10-K は取得していません（サイドバーの取得設定をご確認ください）。")
        elif f.get("error"):
            st.warning(f"取得できませんでした: {f['error']}")
        else:
            st.caption(
                f"最新10-K 提出日: {f.get('filing_date', 'N/A')}"
                + (f" ／ [原文を見る]({f['url']})" if f.get("url") else "")
            )
            if f.get("business"):
                with st.expander("事業の詳細 (Item 1. Business)", expanded=False):
                    st.write(f["business"])
            if f.get("risk_factors"):
                with st.expander("⚠️ リスク要因 (Item 1A. Risk Factors)", expanded=True):
                    st.write(f["risk_factors"])
            if f.get("mdna"):
                with st.expander("経営者による分析 (Item 7. MD&A)", expanded=False):
                    st.write(f["mdna"])

    # --- 取得データ ---
    with tab_data:
        st.caption("AI投資家に渡された銘柄ドシエ（自動収集された情報）")
        st.code(st.session_state.dossier, language="markdown")

    # --- Claudeへ貼付（APIキー不要で正確なデータを使う運用） ---
    with tab_claude:
        st.caption(
            "ここの全文をコピーして、Claudeの『伝説の評議会』プロジェクトに貼り付けてください。"
            "正確な財務・10-Kデータ（無料取得）をもとに、Claude側（定額）で議論できます。"
        )
        claude_text = (
            f"以下は {company.name}（{company.ticker}）の最新データです。"
            "あなたのプロジェクト指示に従い『伝説の評議会』のディベートと"
            "最終評価（12ヶ月目標株価を含む）を実行してください。"
            "数値はこのデータを正として用い、不足分のみ必要に応じて補ってください。\n\n"
            f"{st.session_state.dossier}\n\n"
            f"{grades_to_text(grades)}"
        )
        st.code(claude_text, language="markdown")
        st.caption(
            "コピーはコードブロック右上のアイコンから。"
            "プロジェクト未設定の場合は docs/claude_project_prompt.md の指示文を先に設定してください。"
        )

else:
    st.info(
        "サイドバーでティッカーを入力し、「評議会を招集する」を押してください。\n\n"
        "AI投資家が必要な情報を自動で集め、その銘柄について議論します。"
    )
