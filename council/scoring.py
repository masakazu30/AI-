"""定量スコアリング（決定論的）。

各AI投資家のロジックを、yfinance 由来の指標から客観的に採点する。
AIの揺らぎに依存しない議論の土台を作るのが目的。

各スコア関数は (grade, score, reason) を返す。
- grade: 表示用の評価ラベル（'S'/'A'/'B'/'C'/'BUY'/'WAIT' など）
- score: 0〜5 の数値（合計スコア算出に使用）
- reason: 判断根拠の一文
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .investors import INVESTORS


@dataclass
class Grade:
    grade: str
    score: float
    reason: str


def _safe(m: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = m.get(key, default)
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 投資家別スコア関数
# --------------------------------------------------------------------------- #
def _graham(m: dict[str, Any]) -> Grade:
    price = _safe(m, "price")
    ncav = _safe(m, "ncav_per_share")
    gnum = _safe(m, "graham_number")
    if ncav > 0 and price > 0 and price < ncav * 0.66:
        return Grade("S", 5, f"株価(${price:.2f})がNCAV(${ncav:.2f})の2/3以下。ネットネット株。")
    if gnum > 0 and price > 0 and price < gnum:
        return Grade("A", 4, f"株価(${price:.2f})がグレアム数(${gnum:.2f})を下回る割安。")
    if gnum > 0 and price < gnum * 1.3:
        return Grade("B", 3, "グレアム数にやや近いが安全域は薄い。")
    return Grade("C", 1, "安全域(Margin of Safety)が不足している。")


def _buffett(m: dict[str, Any]) -> Grade:
    roe = _safe(m, "roe")
    oe = _safe(m, "owner_earnings")
    margin_std = _safe(m, "gross_margin_stability", 999)
    if roe > 0.15 and oe > 0 and margin_std < 0.05:
        return Grade("A", 4, "高ROE(15%超)・プラスのオーナー利益・安定した粗利率。堀がある。")
    if roe > 0.10 and oe > 0:
        return Grade("B", 3, "一定のROEとオーナー利益はあるが圧倒的ではない。")
    return Grade("C", 1, "ROEが低い、オーナー利益がマイナス、または粗利率が不安定。")


def _munger(m: dict[str, Any]) -> Grade:
    roe = _safe(m, "roe")
    margin = _safe(m, "profit_margins")
    margin_std = _safe(m, "gross_margin_stability", 999)
    if roe > 0.18 and margin > 0.15 and margin_std < 0.05:
        return Grade("A", 4, "卓越した資本効率と高利益率。質の高いビジネス。")
    if roe > 0.12 and margin > 0.08:
        return Grade("B", 3, "そこそこ質は高いが、妥当な価格かは要検討。")
    return Grade("C", 1, "資本効率・利益率が凡庸。わざわざ買う理由が乏しい。")


def _lynch(m: dict[str, Any]) -> Grade:
    peg = _safe(m, "peg", 999)
    if 0 < peg < 0.5:
        return Grade("S", 5, f"PEG({peg:.2f})が0.5未満。驚異的な成長割安株。")
    if 0 < peg <= 1.0:
        return Grade("A", 4, f"PEG({peg:.2f})が1.0以下。適正価格で成長を買える。")
    if 1.0 < peg <= 2.0:
        return Grade("B", 3, f"PEG({peg:.2f})は平均的。")
    if peg > 2.0 and peg < 900:
        return Grade("C", 1, f"PEG({peg:.2f})が2.0超。成長に対して割高。")
    return Grade("C", 1, "成長率データが不十分でPEGを評価できない。")


def _fisher(m: dict[str, Any]) -> Grade:
    rev_g = _safe(m, "revenue_growth")
    margin = _safe(m, "profit_margins")
    if rev_g > 0.15 and margin > 0.10:
        return Grade("A", 4, "高い売上成長と良好な利益率。優れた成長企業の特徴。")
    if rev_g > 0.05:
        return Grade("B", 3, "一定の成長はあるが卓越とまでは言えない（定性確認が必要）。")
    return Grade("C", 1, "成長の勢いが乏しい。定性面での将来性確認が必須。")


def _templeton(m: dict[str, Any]) -> Grade:
    price = _safe(m, "price")
    low = _safe(m, "low_52")
    if low > 0 and price > 0 and price <= low * 1.10:
        return Grade("S", 5, f"株価が52週安値(${low:.2f})圏。悲観の極みの可能性。")
    if low > 0 and price <= low * 1.25:
        return Grade("A", 4, "52週安値圏に近い。逆張りの好機を探る価値あり。")
    return Grade("C", 1, "悲観の極み(安値圏)ではない。逆張りの妙味は薄い。")


def _livermore(m: dict[str, Any]) -> Grade:
    if m.get("is_breakout"):
        return Grade("BUY", 5, "52週高値圏でのブレイクアウト。トレンドは上向き。")
    price = _safe(m, "price")
    high = _safe(m, "high_52")
    if high > 0 and price >= high * 0.85:
        return Grade("WATCH", 3, f"高値(${high:.2f})に接近中。ブレイク待ち。")
    return Grade("WAIT", 1, f"高値(${high:.2f})まで距離がある。トレンド未確立。")


def _icahn(m: dict[str, Any]) -> Grade:
    price = _safe(m, "price")
    bps = _safe(m, "bps")
    pbr = price / bps if bps > 0 else _safe(m, "pbr", 999)
    cash_ratio = _safe(m, "cash_ratio")
    if 0 < pbr < 1.0 and cash_ratio > 0.30:
        return Grade("S", 5, f"PBR1倍割れ({pbr:.2f})かつキャッシュリッチ({cash_ratio:.0%})。標的。")
    if 0 < pbr < 1.0:
        return Grade("A", 4, f"PBR1倍割れ({pbr:.2f})の資産割安株。価値が見過ごされている。")
    if cash_ratio > 0.30:
        return Grade("B", 3, f"現金比率が高い({cash_ratio:.0%})。株主還元の余地あり。")
    return Grade("C", 1, "明確な資産バリュー(PBR割れ)でもキャッシュリッチでもない。")


def _dalio(m: dict[str, Any], macro: dict[str, Any]) -> Grade:
    season = macro.get("season", "判定不能")
    beta = _safe(m, "beta", 1.0)
    if season == "適温相場 (Goldilocks)":
        return Grade("A", 4, f"局面は{season}。リスク資産に追い風。")
    if season == "リフレーション (Reflation)":
        return Grade("B", 3, f"局面は{season}。景気敏感には妙味、金利上昇は逆風。")
    if season in ("スタグフレーション (Stagflation)", "デフレ/景気後退 (Deflation/Recession)"):
        g = "C"
        s = 1.0
        note = "リスク資産には逆風。守りを固める局面。"
        if beta > 1.2:
            note += f" 高ベータ({beta:.2f})はさらに不利。"
        return Grade(g, s, f"局面は{season}。{note}")
    return Grade("B", 3, "マクロ局面が不明瞭。中立。")


def _marks(m: dict[str, Any], macro: dict[str, Any]) -> Grade:
    vix = _safe(macro, "vix") if macro else 0.0
    near_low = bool(m.get("near_52w_low"))
    is_breakout = bool(m.get("is_breakout"))
    # 二次的思考: 恐怖が高い(VIX高)局面や売られ過ぎは、リスクが報われやすい
    if vix > 25 or near_low:
        return Grade("A", 4, "市場心理が恐怖寄り/売られ過ぎ。リスク対リターンが非対称に良い可能性。")
    if is_breakout and vix < 15:
        return Grade("C", 1, "強欲・楽観が支配的。良い話は織り込み済みでリスクが高い。")
    return Grade("B", 3, "市場心理は中立。振り子の位置を見極める段階。")


# --------------------------------------------------------------------------- #
# ディスパッチ
# --------------------------------------------------------------------------- #
_SCORERS: dict[str, Callable[..., Grade]] = {
    "graham": _graham,
    "buffett": _buffett,
    "munger": _munger,
    "lynch": _lynch,
    "fisher": _fisher,
    "templeton": _templeton,
    "livermore": _livermore,
    "icahn": _icahn,
    "dalio": _dalio,
    "marks": _marks,
}

_MACRO_AWARE = {"dalio", "marks"}


def score_all(metrics: dict[str, Any], macro: dict[str, Any]) -> dict[str, Grade]:
    """全AI投資家の定量グレードを算出する。"""
    grades: dict[str, Grade] = {}
    for inv in INVESTORS:
        scorer = _SCORERS.get(inv.key)
        if scorer is None:
            grades[inv.key] = Grade("B", 3, "定量評価は中立。")
            continue
        if inv.key in _MACRO_AWARE:
            grades[inv.key] = scorer(metrics, macro)
        else:
            grades[inv.key] = scorer(metrics)
    return grades


def council_score(grades: dict[str, Grade]) -> tuple[float, float]:
    """評議会の合計スコアと満点(100点換算)を返す。"""
    if not grades:
        return 0.0, 0.0
    total = sum(g.score for g in grades.values())
    max_total = 5.0 * len(grades)
    normalized = (total / max_total) * 100 if max_total else 0.0
    return round(total, 1), round(normalized, 1)


def grades_to_text(grades: dict[str, Grade]) -> str:
    """LLM へ渡すための、定量スコア一覧テキスト。"""
    from .investors import INVESTORS_BY_KEY

    lines = ["## 定量スコアリング（各投資家ロジックによる客観採点）"]
    for key, g in grades.items():
        inv = INVESTORS_BY_KEY[key]
        lines.append(f"- {inv.emoji} {inv.name}: [{g.grade}] {g.reason}")
    total, norm = council_score(grades)
    lines.append(f"\n合計: {total} / {5.0 * len(grades):.0f} （{norm:.0f}点換算）")
    return "\n".join(lines)
