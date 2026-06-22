"""SEC EDGAR から最新の Form 10-K を取得し、定性セクションを抽出する。

取得するもの:
  - Item 1.  Business（事業の詳細）
  - Item 1A. Risk Factors（リスク要因）
  - Item 7.  Management's Discussion and Analysis / MD&A（経営者による分析）

EDGAR は無料・APIキー不要だが、SECは識別可能な User-Agent を要求する。
SEC_EDGAR_USER_AGENT 環境変数で上書き可能。ETF等10-Kが無い銘柄は静かにスキップする。
ネットワーク失敗時も例外は投げず、{"error": ...} を返して呼び出し側で継続できるようにする。
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# SECは識別可能な「名前＋連絡先」形式のUser-Agentを要求する。
# 確実性を上げたい場合は SEC_EDGAR_USER_AGENT に「あなたの名前 you@example.com」を設定する。
_DEFAULT_UA = "Council of Legends investing-research-app contact@example.com"

# セクションごとの抽出上限文字数（トークン/コスト配慮）
_LIMITS = {"business": 2500, "risk_factors": 4500, "mdna": 4500}

# プロセス内キャッシュ
_TICKER_MAP: dict[str, int] | None = None
_SECTION_CACHE: dict[str, dict] = {}


def _user_agent() -> str:
    return os.environ.get("SEC_EDGAR_USER_AGENT") or _DEFAULT_UA


def _http_get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Host": urllib.parse.urlparse(url).netloc,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip

            data = gzip.decompress(data)
    return data


def _load_ticker_map() -> dict[str, int]:
    global _TICKER_MAP
    if _TICKER_MAP is not None:
        return _TICKER_MAP
    raw = _http_get(_TICKERS_URL)
    obj = json.loads(raw)
    mapping: dict[str, int] = {}
    for row in obj.values():
        ticker = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = int(cik)
    _TICKER_MAP = mapping
    return mapping


def get_cik(ticker: str) -> int | None:
    try:
        return _load_ticker_map().get(ticker.strip().upper())
    except Exception:  # noqa: BLE001
        return None


def _latest_10k(cik: int) -> dict | None:
    """最新の 10-K のアクセッション番号・主要ドキュメント名を返す。"""
    raw = _http_get(_SUBMISSIONS_URL.format(cik=cik))
    obj = json.loads(raw)
    recent = obj.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    for i, form in enumerate(forms):
        if form == "10-K":
            return {
                "accession": accessions[i],
                "primary_doc": primary_docs[i],
                "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                "report_date": report_dates[i] if i < len(report_dates) else "",
            }
    return None


def _clean_html(raw: bytes) -> str:
    """HTML/iXBRL を素のテキストに変換する（軽量・依存ライブラリ不要）。"""
    text = raw.decode("utf-8", errors="ignore")
    # script/style とXBRLの隠し要素を除去
    text = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<ix:header.*?</ix:header>", " ", text)
    # 改行になりうるブロック要素
    text = re.sub(r"(?i)<(br|/p|/div|/tr|/li|/h\d)[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)  # 残りのタグ除去
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _extract_section(text: str, start_pat: str, end_pats: list[str], max_chars: int) -> str:
    """start_pat から end_pats のいずれかまでを抽出する。

    目次(TOC)は本文より前にあるため、まず「実質的な長さを持つ最後の開始位置」を採用し、
    見つからなければ最長候補にフォールバックする。
    """
    starts = [m.start() for m in re.finditer(start_pat, text, re.IGNORECASE)]
    if not starts:
        return ""

    def section_from(s: int) -> str:
        end = len(text)
        for ep in end_pats:
            mm = re.search(ep, text[s + 60:], re.IGNORECASE)
            if mm:
                end = min(end, s + 60 + mm.start())
        return text[s:end].strip()

    chosen = ""
    for s in reversed(starts):  # TOC(前方)を避け、後方の本文セクションを優先
        sec = section_from(s)
        if len(sec) >= 300:
            chosen = sec
            break
    if not chosen:
        chosen = max((section_from(s) for s in starts), key=len, default="")
    if len(chosen) > max_chars:
        chosen = chosen[:max_chars].rsplit(" ", 1)[0] + " …(以下省略)"
    return chosen


def fetch_10k_sections(ticker: str) -> dict:
    """ティッカーから最新10-Kの定性3セクションを取得する。失敗時は {'error': ...}。"""
    ticker = ticker.strip().upper()
    if ticker in _SECTION_CACHE:
        return _SECTION_CACHE[ticker]

    result: dict = {}
    try:
        try:
            ticker_map = _load_ticker_map()
        except Exception as exc:  # noqa: BLE001 — SECへの接続失敗
            return {"error": f"SEC EDGARへ接続できませんでした: {exc}"}

        cik = ticker_map.get(ticker)
        if cik is None:
            result = {"error": f"'{ticker}' のCIKが見つかりません（米国上場でない可能性）。"}
            _SECTION_CACHE[ticker] = result
            return result

        filing = _latest_10k(cik)
        if not filing:
            result = {"error": "この銘柄には10-Kがありません（ETF/ADR等の可能性）。"}
            _SECTION_CACHE[ticker] = result
            return result

        acc_nodash = filing["accession"].replace("-", "")
        url = _ARCHIVE_BASE.format(cik=cik, acc=acc_nodash, doc=filing["primary_doc"])
        text = _clean_html(_http_get(url, timeout=25.0))

        business = _extract_section(
            text, r"item\s*1\.?\s*business",
            [r"item\s*1a[\.\s]", r"item\s*2[\.\s]"], _LIMITS["business"],
        )
        risk = _extract_section(
            text, r"item\s*1a\.?\s*risk\s*factors",
            [r"item\s*1b[\.\s]", r"item\s*2[\.\s]"], _LIMITS["risk_factors"],
        )
        mdna = _extract_section(
            text, r"item\s*7\.?\s*management.s\s+discussion",
            [r"item\s*7a[\.\s]", r"item\s*8[\.\s]"], _LIMITS["mdna"],
        )

        result = {
            "filing_date": filing.get("filing_date", ""),
            "report_date": filing.get("report_date", ""),
            "url": url,
            "business": business,
            "risk_factors": risk,
            "mdna": mdna,
        }
        if not (business or risk or mdna):
            result["error"] = "10-Kは取得できましたが、本文セクションを抽出できませんでした。"
    except Exception as exc:  # noqa: BLE001
        result = {"error": f"EDGAR取得エラー: {exc}"}

    _SECTION_CACHE[ticker] = result
    return result
