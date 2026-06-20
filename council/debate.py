"""議論（ディベート）のオーケストレーション。

ハイブリッド方式:
  1. 定量フェーズ … scoring.py で決定論的に採点（debate外で実施）
  2. 初期見解     … 各AI投資家が並列に意見を述べる
  3. 円卓討論     … 互いの見解を読み、反論・補強する（任意で複数ラウンド）
  4. 最終評価     … モデレーターが全議論を統合し、強気/中立/弱気と確信度を出す
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable

from .investors import INVESTORS, Investor
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


def _opening_prompt(
    investor: Investor, dossier: str, grade: Grade
) -> str:
    return (
        f"以下の銘柄について、あなた（{investor.name}）の投資哲学に基づき初期見解を述べてください。\n\n"
        f"{dossier}\n\n"
        f"## あなたのロジックによる定量評価\n"
        f"評価: [{grade.grade}]\n根拠: {grade.reason}\n\n"
        "出力フォーマット（200〜300字程度、簡潔に）:\n"
        "1. 結論（強気 / 中立 / 弱気）と一言理由\n"
        "2. あなたの哲学から見た最重要ポイント1〜2点（必ず数字を引用）\n"
        "3. 最大の懸念点 1点"
    )


def _rebuttal_prompt(
    investor: Investor, dossier: str, grade: Grade, others: str
) -> str:
    return (
        f"あなた（{investor.name}）は評議会の円卓討論に参加しています。\n"
        f"対象銘柄の資料と、他の伝説的投資家たちの初期見解は以下の通りです。\n\n"
        f"{dossier}\n\n"
        f"## あなたの定量評価: [{grade.grade}] {grade.reason}\n\n"
        f"## 他の投資家の初期見解\n{others}\n\n"
        "他の投資家の主張に対して、あなたの立場から反論または補強を行ってください。"
        "誰のどの主張に応じるのかを明示し、あなたの哲学から見て"
        "見落とされている論点を指摘してください。"
        "150〜250字程度で、自分の口調を保ち、鋭く。"
    )


def _verdict_prompt(dossier: str, scores_text: str, transcript: str) -> str:
    return (
        "あなたは『The Council of Legends』の司会者（モデレーター）です。"
        "以下の銘柄資料・定量スコア・投資家たちの議論をすべて統合し、"
        "中立かつ実務的な最終評価をまとめてください。\n\n"
        f"{dossier}\n\n"
        f"{scores_text}\n\n"
        f"## 議論の記録\n{transcript}\n\n"
        "## 出力フォーマット（Markdown）\n"
        "### 🏛️ 評議会の最終評価\n"
        "**総合判断:** （強気 / やや強気 / 中立 / やや弱気 / 弱気 のいずれか）\n"
        "**確信度:** （0〜100%）\n\n"
        "**強気派の論拠:** （箇条書き2〜3点、誰の主張かを含む）\n"
        "**弱気派の論拠:** （箇条書き2〜3点、誰の主張かを含む）\n"
        "**意見が割れた論点:** （1〜2点）\n\n"
        "**この銘柄に投資すると儲かるか:** "
        "（投資妙味・時間軸・主なリスクを3〜4文で。"
        "ただし最終判断は読者自身が行うべきと明記する）"
    )


# --------------------------------------------------------------------------- #
def _run_one(
    client: LLMClient, investor: Investor, prompt: str, max_tokens: int
) -> Statement:
    try:
        text = client.complete(investor.system_prompt, prompt, max_tokens=max_tokens)
        if not text:
            return Statement(
                investor.key, investor.name, investor.emoji,
                "（応答が空でした）", is_error=True,
            )
        return Statement(investor.key, investor.name, investor.emoji, text)
    except Exception as exc:  # noqa: BLE001
        return Statement(
            investor.key, investor.name, investor.emoji,
            f"（発言の生成に失敗しました: {exc}）", is_error=True,
        )


def _run_round_parallel(
    client: LLMClient,
    prompt_fn: Callable[[Investor], str],
    max_workers: int,
    max_tokens: int,
) -> list[Statement]:
    """各投資家の発言を並列生成する（入力順を保持して返す）。"""
    results: dict[str, Statement] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, client, inv, prompt_fn(inv), max_tokens): inv.key
            for inv in INVESTORS
        }
        for fut in concurrent.futures.as_completed(futures):
            stmt = fut.result()
            results[stmt.investor_key] = stmt
    return [results[inv.key] for inv in INVESTORS if inv.key in results]


def _format_statements(statements: list[Statement]) -> str:
    return "\n\n".join(
        f"{s.emoji} {s.investor_name}:\n{s.text}" for s in statements
    )


# --------------------------------------------------------------------------- #
def run_debate(
    client: LLMClient,
    dossier: str,
    grades: dict[str, Grade],
    scores_text: str,
    *,
    do_rebuttal: bool = True,
    max_workers: int = 5,
    progress: ProgressCallback | None = None,
) -> DebateResult:
    """評議会の議論を実行する。"""

    def notify(msg: str) -> None:
        if progress:
            progress(msg)

    result = DebateResult()

    # --- ラウンド1: 初期見解（並列） ---
    notify("各投資家が初期見解を作成中…")
    result.openings = _run_round_parallel(
        client,
        lambda inv: _opening_prompt(inv, dossier, grades[inv.key]),
        max_workers=max_workers,
        max_tokens=6000,
    )

    transcript_parts = ["### 初期見解\n" + _format_statements(result.openings)]

    # --- ラウンド2: 円卓討論（任意・並列） ---
    if do_rebuttal:
        notify("円卓討論（反論・補強）を実施中…")
        openings_text = _format_statements(result.openings)

        def reb_prompt(inv: Investor) -> str:
            # 自分以外の見解を渡す
            others = "\n\n".join(
                f"{s.emoji} {s.investor_name}:\n{s.text}"
                for s in result.openings
                if s.investor_key != inv.key
            )
            return _rebuttal_prompt(inv, dossier, grades[inv.key], others or openings_text)

        result.rebuttals = _run_round_parallel(
            client, reb_prompt, max_workers=max_workers, max_tokens=5000
        )
        transcript_parts.append("### 円卓討論\n" + _format_statements(result.rebuttals))

    # --- ラウンド3: 最終評価 ---
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
