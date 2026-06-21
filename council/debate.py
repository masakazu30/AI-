"""議論（ディベート）のオーケストレーション。

ハイブリッド方式:
  1. 定量フェーズ … scoring.py で決定論的に採点（debate外で実施）
  2. 初期見解     … 全AI投資家ぶんを1回の呼び出しでまとめて生成（無料枠対応）
  3. 円卓討論     … 互いの見解を読み、反論・補強（任意・1回の呼び出し）
  4. 最終評価     … モデレーターが全議論を統合（1回の呼び出し）

無料枠（例: Gemini は 1分5回まで）でも動くよう、LLM呼び出しは合計2〜3回に抑えている。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .investors import INVESTORS, INVESTORS_BY_KEY
from .llm import LLMClient
from .scoring import Grade


@dataclass
class Statement:
    investor_key: str
    investor_name: str
    emoji: str
    text: str
    is_error: bool = False


@dataclass
class DebateResult:
    openings: list[Statement] = field(default_factory=list)
    rebuttals: list[Statement] = field(default_factory=list)
    verdict: str = ""
    error: str | None = None


ProgressCallback = Callable[[str], None]

_MARKER = re.compile(r"^===\s*([a-z_]+)\s*===\s*$", re.MULTILINE)

_DEBATE_SYSTEM = (
    "あなたは『The Council of Legends（伝説の評議会）』の進行を担うAIです。"
    "指定された複数の伝説的投資家それぞれに、本人の投資哲学・口調で完全になりきって発言を生成します。"
    "各投資家の人物像（実在の著名投資家）に忠実に、与えられた数値を必ず根拠として引用してください。"
    "日本語で、指定されたマーカー形式を厳密に守って出力します。"
)


def _roster_block(grades: dict[str, Grade]) -> str:
    """参加者一覧（名前・系統・定量評価）をテキスト化。"""
    lines = []
    for inv in INVESTORS:
        g = grades[inv.key]
        lines.append(
            f"- {inv.key}（{inv.name} / {inv.school}・{inv.tagline}）: "
            f"定量評価[{g.grade}] {g.reason}"
        )
    return "\n".join(lines)


def _marker_spec() -> str:
    keys = "\n".join(f"==={inv.key}===\n（{inv.name}の発言）" for inv in INVESTORS)
    return (
        "出力は必ず次の形式を厳守してください。各投資家の発言の直前に、"
        "その投資家のマーカー行（=== で囲んだ英小文字キー）を**単独行で**置きます。"
        "前置きや締めの文章は不要です。\n\n" + keys
    )


def _opening_prompt(dossier: str, grades: dict[str, Grade]) -> str:
    return (
        "以下の銘柄について、各投資家の初期見解を生成してください。\n\n"
        f"{dossier}\n\n"
        f"## 参加者と、それぞれのロジックによる定量評価\n{_roster_block(grades)}\n\n"
        "各投資家について、その人物の哲学に忠実に150〜250字で初期見解を書いてください。"
        "必ず①結論（強気/中立/弱気）②最重要ポイント（数値を引用）③最大の懸念点 を含めること。\n\n"
        f"{_marker_spec()}"
    )


def _rebuttal_prompt(dossier: str, grades: dict[str, Grade], openings: str) -> str:
    return (
        "評議会の円卓討論です。各投資家が、他の投資家の初期見解に対して反論または補強を行います。\n\n"
        f"{dossier}\n\n"
        f"## 各投資家の定量評価\n{_roster_block(grades)}\n\n"
        f"## 初期見解の記録\n{openings}\n\n"
        "各投資家について、誰のどの主張に応じるのかを明示しつつ、自分の哲学から見て"
        "見落とされている論点を指摘する反論・補強を150〜250字で書いてください。\n\n"
        f"{_marker_spec()}"
    )


def _verdict_prompt(dossier: str, scores_text: str, transcript: str) -> str:
    return (
        "あなたは『The Council of Legends』の司会者です。"
        "以下の銘柄資料・定量スコア・投資家たちの議論をすべて統合し、"
        "中立かつ実務的な最終評価をまとめてください。\n\n"
        f"{dossier}\n\n{scores_text}\n\n"
        f"## 議論の記録\n{transcript}\n\n"
        "## 出力フォーマット（Markdown）\n"
        "### 🏛️ 評議会の最終評価\n"
        "**総合判断:** （強気 / やや強気 / 中立 / やや弱気 / 弱気 のいずれか）\n"
        "**確信度:** （0〜100%）\n\n"
        "**🎯 12ヶ月目標株価（評議会）:** "
        "強気シナリオ $◯ / 基本シナリオ $◯ / 弱気シナリオ $◯ "
        "（基本シナリオの現在株価比 約±◯%）\n"
        "**目標株価の根拠:** "
        "（予想EPS×妥当PER、EV/EBITDA、グレアム数、アナリスト目標株価などの"
        "バリュエーション指標を2〜3個明示的に用いて、どう算出したかを2〜3文で説明する）\n\n"
        "**強気派の論拠:** （箇条書き2〜3点、誰の主張かを含む）\n"
        "**弱気派の論拠:** （箇条書き2〜3点、誰の主張かを含む）\n"
        "**意見が割れた論点:** （1〜2点）\n\n"
        "**この銘柄に投資すると儲かるか:** "
        "（投資妙味・時間軸・主なリスクを3〜4文で。"
        "ただし最終判断は読者自身が行うべきと明記する）"
    )


def _parse_marked(text: str) -> dict[str, str]:
    """マーカー区切りのテキストを {投資家キー: 発言} に分解する。"""
    sections: dict[str, str] = {}
    matches = list(_MARKER.finditer(text))
    for i, mt in enumerate(matches):
        key = mt.group(1)
        start = mt.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if key in INVESTORS_BY_KEY:
            sections[key] = text[start:end].strip()
    return sections


def _to_statements(parsed: dict[str, str]) -> list[Statement]:
    out: list[Statement] = []
    for inv in INVESTORS:
        body = parsed.get(inv.key)
        if body:
            out.append(Statement(inv.key, inv.name, inv.emoji, body))
        else:
            out.append(
                Statement(
                    inv.key, inv.name, inv.emoji,
                    "（この投資家の発言を取得できませんでした）", is_error=True,
                )
            )
    return out


def _format_statements(statements: list[Statement]) -> str:
    return "\n\n".join(
        f"{s.emoji} {s.investor_name}:\n{s.text}"
        for s in statements
        if not s.is_error
    )


# --------------------------------------------------------------------------- #
def run_debate(
    client: LLMClient,
    dossier: str,
    grades: dict[str, Grade],
    scores_text: str,
    *,
    do_rebuttal: bool = True,
    progress: ProgressCallback | None = None,
) -> DebateResult:
    """評議会の議論を実行する（LLM呼び出しは合計2〜3回）。"""

    def notify(msg: str) -> None:
        if progress:
            progress(msg)

    result = DebateResult()

    # --- ラウンド1: 初期見解（1回の呼び出しで全員ぶん） ---
    notify("各投資家が初期見解を作成中…")
    opening_text = client.complete(
        _DEBATE_SYSTEM, _opening_prompt(dossier, grades), max_tokens=8000
    )
    result.openings = _to_statements(_parse_marked(opening_text))
    transcript_parts = ["### 初期見解\n" + _format_statements(result.openings)]

    # --- ラウンド2: 円卓討論（任意・1回の呼び出し） ---
    if do_rebuttal:
        notify("円卓討論（反論・補強）を実施中…")
        rebuttal_text = client.complete(
            _DEBATE_SYSTEM,
            _rebuttal_prompt(dossier, grades, _format_statements(result.openings)),
            max_tokens=8000,
        )
        result.rebuttals = _to_statements(_parse_marked(rebuttal_text))
        transcript_parts.append("### 円卓討論\n" + _format_statements(result.rebuttals))

    # --- ラウンド3: 最終評価（1回の呼び出し） ---
    notify("司会が最終評価を統合中…")
    transcript = "\n\n".join(transcript_parts)
    try:
        result.verdict = client.complete(
            "あなたは中立で実務的な投資評議会の司会者です。日本語で回答します。",
            _verdict_prompt(dossier, scores_text, transcript),
            max_tokens=8000,
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"最終評価の生成に失敗しました: {exc}"

    notify("完了")
    return result
