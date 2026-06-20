"""銘柄・マクロデータの自動取得と、投資判断に必要な指標の算出。

ユーザーはティッカーを指定するだけでよい。AI投資家が議論するために必要な
財務・株価・マクロ指標はこのモジュールがすべて自動で収集・計算する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------------------------------- #
# 低レベルヘルパー
# --------------------------------------------------------------------------- #
def _fin_value(df: pd.DataFrame | None, keys: list[str]) -> float:
    """財務データフレームから、複数の表記ゆれを許容して最新値を安全に取得する。"""
    if df is None or df.empty:
        return 0.0
    for key in keys:
        if key in df.index:
            try:
                val = df.loc[key].iloc[0]
                if not pd.isna(val):
                    return float(val)
            except Exception:
                continue
    return 0.0


def _num(value: Any, default: float = 0.0) -> float:
    """None / NaN を安全に数値へ変換する。"""
    if value is None:
        return default
    try:
        f = float(value)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def format_large_number(num: float | None) -> str:
    """大きな数値を T/B/M 表記に整形する。"""
    if num is None or num == 0:
        return "N/A"
    n = float(num)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12:
        return f"{sign}{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{sign}{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{sign}{n / 1e6:.2f}M"
    return f"{sign}{n:,.0f}"


# --------------------------------------------------------------------------- #
# データ構造
# --------------------------------------------------------------------------- #
@dataclass
class CompanyData:
    ticker: str
    name: str
    metrics: dict[str, Any]
    macro: dict[str, Any]
    summary: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# マクロ環境
# --------------------------------------------------------------------------- #
def fetch_macro() -> dict[str, Any]:
    """金利・VIX・S&P500・金から、ダリオ的な相場局面とドラッケンミラー的な流動性を推定。"""
    macro: dict[str, Any] = {}
    try:
        tickers = ["^TNX", "^VIX", "^GSPC", "GC=F"]
        data = yf.download(
            tickers, period="1y", interval="1d", progress=False, auto_adjust=True
        )["Close"]
        if data is None or data.empty:
            return {"error": "マクロデータの取得に失敗しました。"}

        def last(col: str) -> float:
            try:
                return _num(data[col].dropna().iloc[-1])
            except Exception:
                return 0.0

        macro["tnx"] = last("^TNX")
        macro["vix"] = last("^VIX")
        macro["spx"] = last("^GSPC")
        macro["gold"] = last("GC=F")

        def trend_up(col: str) -> bool | None:
            try:
                series = data[col].dropna()
                ma50 = series.rolling(50).mean().iloc[-1]
                ma200 = series.rolling(200).mean().iloc[-1]
                if pd.isna(ma50) or pd.isna(ma200):
                    return None
                return bool(ma50 > ma200)
            except Exception:
                return None

        growth_up = trend_up("^GSPC")
        inflation_up = trend_up("^TNX")  # 金利を期待インフレの代理変数として使用

        if growth_up is None or inflation_up is None:
            macro["season"] = "判定不能"
            macro["liquidity"] = "判定不能"
        else:
            if growth_up and not inflation_up:
                macro["season"] = "適温相場 (Goldilocks)"
            elif growth_up and inflation_up:
                macro["season"] = "リフレーション (Reflation)"
            elif not growth_up and inflation_up:
                macro["season"] = "スタグフレーション (Stagflation)"
            else:
                macro["season"] = "デフレ/景気後退 (Deflation/Recession)"
            macro["liquidity"] = "縮小 (Tightening)" if inflation_up else "拡大/逃避 (Easing)"
    except Exception as exc:  # noqa: BLE001
        macro["error"] = str(exc)
    return macro


# --------------------------------------------------------------------------- #
# 銘柄指標
# --------------------------------------------------------------------------- #
def _compute_metrics(ticker: yf.Ticker, info: dict[str, Any]) -> dict[str, Any]:
    balance = ticker.balance_sheet
    income = ticker.income_stmt
    cashflow = ticker.cashflow

    shares = _num(info.get("sharesOutstanding"), 0.0) or 1.0

    m: dict[str, Any] = {}

    # --- 価格 ---
    m["price"] = _num(info.get("currentPrice") or info.get("regularMarketPrice"))
    m["high_52"] = _num(info.get("fiftyTwoWeekHigh"))
    m["low_52"] = _num(info.get("fiftyTwoWeekLow"))
    m["market_cap"] = _num(info.get("marketCap"))
    m["shares_outstanding"] = shares

    # --- バリュエーション ---
    m["trailing_pe"] = _num(info.get("trailingPE"))
    m["forward_pe"] = _num(info.get("forwardPE"))
    m["pbr"] = _num(info.get("priceToBook"))
    m["eps"] = _num(info.get("trailingEps"))
    m["bps"] = _num(info.get("bookValue"))
    m["profit_margins"] = _num(info.get("profitMargins"))
    m["dividend_yield"] = _num(info.get("dividendYield"))
    m["beta"] = _num(info.get("beta"))
    m["roe"] = _num(info.get("returnOnEquity"))
    m["revenue_growth"] = _num(info.get("revenueGrowth"))
    m["earnings_growth"] = _num(info.get("earningsGrowth"))

    # --- PEG（公式値が無ければ PER / 利益成長率 で補完）---
    peg = info.get("pegRatio") or info.get("trailingPegRatio")
    if peg is None:
        pe = info.get("trailingPE")
        growth = info.get("earningsGrowth")
        if pe and growth and growth > 0:
            peg = pe / (growth * 100)
    m["peg"] = _num(peg, 999.0) or 999.0

    # --- グレアム指標 ---
    current_assets = _fin_value(balance, ["Current Assets", "Total Current Assets"])
    total_liabilities = _fin_value(
        balance, ["Total Liabilities Net Minority Interest", "Total Liabilities"]
    )
    m["ncav_per_share"] = (current_assets - total_liabilities) / shares
    if m["eps"] > 0 and m["bps"] > 0:
        m["graham_number"] = math.sqrt(22.5 * m["eps"] * m["bps"])
    else:
        m["graham_number"] = 0.0

    # --- バフェット指標 ---
    net_income = _fin_value(income, ["Net Income", "Net Income Common Stockholders"])
    depreciation = _fin_value(
        cashflow, ["Depreciation And Amortization", "Depreciation & Amortization"]
    )
    capex = abs(_fin_value(cashflow, ["Capital Expenditure", "Capital Expenditures"]))
    m["owner_earnings"] = net_income + depreciation - capex

    if (
        income is not None
        and not income.empty
        and "Gross Profit" in income.index
        and "Total Revenue" in income.index
    ):
        try:
            gp = income.loc["Gross Profit"]
            rev = income.loc["Total Revenue"].replace(0, np.nan)
            m["gross_margin_stability"] = float((gp / rev).std())
        except Exception:
            m["gross_margin_stability"] = 999.0
    else:
        m["gross_margin_stability"] = 999.0

    # --- 現金 / アイカーン指標 ---
    total_cash = _fin_value(
        balance,
        ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    )
    total_debt = _fin_value(balance, ["Total Debt"])
    m["total_cash"] = total_cash
    m["total_debt"] = total_debt
    m["net_cash_per_share"] = (total_cash - total_debt) / shares
    m["cash_ratio"] = total_cash / m["market_cap"] if m["market_cap"] else 0.0

    # --- リバモア / テンプルトン指標 ---
    m["is_breakout"] = m["high_52"] > 0 and m["price"] >= m["high_52"] * 0.98
    m["near_52w_low"] = m["low_52"] > 0 and m["price"] <= m["low_52"] * 1.10

    return m


def fetch_company(ticker_symbol: str, include_macro: bool = True) -> CompanyData:
    """ティッカーから企業データ一式を取得する。失敗時は ValueError を送出。"""
    ticker_symbol = ticker_symbol.strip().upper()
    if not ticker_symbol:
        raise ValueError("ティッカーシンボルが空です。")

    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info or {}
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"データ取得に失敗しました: {exc}") from exc

    if not info or not (
        info.get("regularMarketPrice")
        or info.get("currentPrice")
        or info.get("shortName")
    ):
        raise ValueError(
            f"'{ticker_symbol}' のデータが見つかりません。ティッカーをご確認ください。"
        )

    metrics = _compute_metrics(ticker, info)
    warnings: list[str] = []
    if metrics["price"] <= 0:
        warnings.append("現在株価が取得できませんでした。")
    if metrics["market_cap"] <= 0:
        warnings.append("時価総額が取得できませんでした。")

    macro = fetch_macro() if include_macro else {}

    return CompanyData(
        ticker=ticker_symbol,
        name=info.get("shortName") or info.get("longName") or ticker_symbol,
        metrics=metrics,
        macro=macro,
        summary=(info.get("longBusinessSummary") or "")[:800],
        sector=info.get("sector") or "Unknown",
        industry=info.get("industry") or "Unknown",
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# 議論用ドシエ（LLM へ渡す情報のテキスト化）
# --------------------------------------------------------------------------- #
def build_dossier(company: CompanyData) -> str:
    """AI投資家が読むための、銘柄情報サマリー（資料）を組み立てる。"""
    m = company.metrics
    macro = company.macro

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%" if x else "N/A"

    def usd(x: float) -> str:
        return f"${x:,.2f}" if x else "N/A"

    peg_str = f"{m['peg']:.2f}" if m["peg"] < 900 else "N/A"
    gms = m["gross_margin_stability"]
    gms_str = f"{gms:.3f}" if gms < 900 else "N/A"

    lines = [
        f"# 分析対象: {company.name} ({company.ticker})",
        f"- セクター: {company.sector} / 業種: {company.industry}",
        f"- 事業概要: {company.summary or 'N/A'}",
        "",
        "## 株価・規模",
        f"- 現在株価: {usd(m['price'])}",
        f"- 52週高値: {usd(m['high_52'])} / 52週安値: {usd(m['low_52'])}",
        f"- 高値ブレイクアウト: {'YES' if m['is_breakout'] else 'NO'}"
        f" / 52週安値圏: {'YES' if m['near_52w_low'] else 'NO'}",
        f"- 時価総額: {format_large_number(m['market_cap'])}",
        "",
        "## バリュエーション",
        f"- PER(実績): {m['trailing_pe']:.2f} / PER(予想): {m['forward_pe']:.2f}",
        f"- PBR: {m['pbr']:.2f} / PEGレシオ: {peg_str}",
        f"- EPS: {usd(m['eps'])} / BPS: {usd(m['bps'])}",
        f"- 配当利回り: {pct(m['dividend_yield'])} / ベータ: {m['beta']:.2f}",
        "",
        "## 収益性・財務（投資家別の核心指標）",
        f"- ROE: {pct(m['roe'])} / 純利益率: {pct(m['profit_margins'])}",
        f"- 売上成長率: {pct(m['revenue_growth'])} / 利益成長率: {pct(m['earnings_growth'])}",
        f"- オーナー利益(Buffett): {format_large_number(m['owner_earnings'])}",
        f"- 粗利率の標準偏差(安定性, 低いほど良): {gms_str}",
        f"- NCAV/株(Graham): {usd(m['ncav_per_share'])}"
        f" / グレアム数: {usd(m['graham_number'])}",
        f"- 1株純現金(Lynch/Icahn): {usd(m['net_cash_per_share'])}",
        f"- 現金: {format_large_number(m['total_cash'])}"
        f" / 有利子負債: {format_large_number(m['total_debt'])}"
        f" / 現金比率(対時価総額): {pct(m['cash_ratio'])}",
        "",
        "## マクロ環境",
    ]
    if macro.get("error"):
        lines.append(f"- 取得エラー: {macro['error']}")
    elif macro:
        lines += [
            f"- 相場局面(Dalio): {macro.get('season', 'N/A')}",
            f"- 流動性シグナル(Druckenmiller): {macro.get('liquidity', 'N/A')}",
            f"- 米10年債利回り: {macro.get('tnx', 0):.2f}%"
            f" / VIX: {macro.get('vix', 0):.2f}",
            f"- S&P500: {macro.get('spx', 0):,.0f}"
            f" / Gold: ${macro.get('gold', 0):,.0f}",
        ]
    else:
        lines.append("- （マクロ情報なし）")

    return "\n".join(lines)
