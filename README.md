# 🏛️ The Council of Legends — 伝説的投資家によるAI議論アプリ

過去から現在までの**伝説の米国株投資家をAIで再現**し、ユーザーが指定した銘柄について
AI投資家同士が議論を重ね、「投資すべきか否か」を評価するアプリです。

ユーザーはティッカーを入力するだけ。投資判断に必要な財務・株価・マクロ指標は**自動で取得**され、
その情報をもとにAI投資家たちが議論を開始します。

---

## 仕組み（処理の流れ）

```
ユーザーがティッカー入力 (例: NVDA)
        │
        ▼
1. データ自動取得
   ├─ yfinance … 財務(複数年)・株価・各種指標・アナリスト目標株価・マクロ(金利/VIX/S&P500/金)
   └─ SEC EDGAR … 最新10-Kの定性情報(事業詳細 Item1 / リスク要因 Item1A / MD&A Item7)
        │
        ▼
2. 定量スコアリング (決定論的)       … 各投資家ロジックで客観採点。議論の土台
        │
        ▼
3. AI議論 (ハイブリッド方式)
   ├─ 初期見解  … 各AI投資家が自分の哲学で意見表明
   ├─ 円卓討論  … 互いの見解に反論・補強
   └─ 最終評価  … 司会AIが統合し「強気/中立/弱気＋確信度＋12ヶ月目標株価」を提示
```

### 収集データ
- **yfinance**: 複数年の売上/純利益/FCF推移とCAGR、収益性(各利益率/ROE/ROA)、財務健全性(流動比率/D-E/FCF利回り)、バリュエーション(PER/PBR/PSR/EV-EBITDA/PEG)、配当、アナリストのコンセンサス目標株価・レーティング、事業概要
- **SEC EDGAR (Form 10-K)**: 事業の詳細(Item 1)、リスク要因(Item 1A)、経営者による分析 MD&A(Item 7)。米国上場の10-K提出企業のみ（ETF/ADR等は自動スキップ）

伝説の投資家たちの投資ロジックの調査・分析は
[`docs/legendary_investors.md`](docs/legendary_investors.md) にまとめています。
各AI投資家の人格はこの分析に基づいて設計されています。

## 評議会メンバー（10名）

| | 投資家 | 系統 |
| --- | --- | --- |
| 🛡️ | ベンジャミン・グレアム | バリュー / 安全域 |
| 🏰 | ウォーレン・バフェット | クオリティ / 堀 |
| 🧠 | チャーリー・マンガー | クオリティ / 反転思考 |
| 🚀 | ピーター・リンチ | 成長 / GARP |
| 🔬 | フィリップ・フィッシャー | クオリティ / 定性 |
| 📉 | ジョン・テンプルトン | バリュー / 逆張り |
| 📈 | ジェシー・リバモア | モメンタム |
| 🦁 | カール・アイカーン | アクティビスト |
| 🌍 | レイ・ダリオ | マクロ |
| 🎯 | ハワード・マークス | 市場心理 / リスク |

## セットアップ

```bash
pip install -r requirements.txt
```

### APIキー

Claude（既定）か Gemini のどちらかが使えます。環境変数で設定するか、アプリのサイドバーで入力します。

```bash
# Claude を使う場合（既定モデル: claude-opus-4-8）
export ANTHROPIC_API_KEY="sk-ant-..."

# Gemini を使う場合（既定モデル: gemini-2.5-flash）
export GEMINI_API_KEY="..."
```

### （任意）SEC EDGAR の User-Agent

SECは識別可能な User-Agent を要求します。既定値でも動きますが、環境によって 403 で
10-K取得に失敗する場合は、あなたの連絡先を設定すると確実です（任意）。

```bash
export SEC_EDGAR_USER_AGENT="Your Name your-email@example.com"
```

Streamlit Cloud では「Secrets」に `SEC_EDGAR_USER_AGENT = "Your Name your-email@example.com"` を追加します。

## 起動

```bash
streamlit run app.py
```

ブラウザが開いたら、サイドバーで AIエンジン・ティッカーを選び「⚖️ 評議会を招集する」を押します。

## 使い方は2通り

### A. アプリ内でAIに議論させる（API課金あり）
サイドバーでAPIキーを入れて実行すると、データ取得→議論→最終評価まで全自動。

### B. 正確なデータ × Claude定額（APIキー不要・おすすめ）
データ取得（yfinance/EDGAR）はLLMを使わないため**APIキー無しで動きます**。
1. **APIキーを入れずに**アプリを実行（データ・定量スコア・10-Kまで取得できる）
2. 「🤖 Claudeへ貼付」タブの全文をコピー
3. Claude.ai の「伝説の評議会」プロジェクト（`docs/claude_project_prompt.md` 参照）に貼り付け
→ **正確な数値（無料）× Claude本体の議論（定額）** の“いいとこ取り”。

API不要でClaudeチャット内だけで完結させる方法は `docs/claude_project_prompt.md` を参照。

## プロジェクト構成

```
app.py                       Streamlit UI
council/
  data.py                    銘柄・マクロデータの自動取得と指標算出
  scoring.py                 各投資家ロジックによる定量スコアリング（決定論的）
  investors.py               AI投資家の人格・哲学（system prompt）
  llm.py                     Claude / Gemini 共通クライアント
  debate.py                  議論オーケストレーション（初期見解→円卓討論→最終評価）
docs/
  legendary_investors.md     伝説の投資家の投資ロジック調査・分析まとめ
```

## 注意

- 本アプリの出力は **AIによる分析であり、投資助言ではありません**。
  最終的な投資判断はご自身の責任で行ってください。
- 株価・財務データは yfinance（Yahoo Finance 非公式API）に依存します。
  銘柄や時間帯により一部指標が取得できないことがあります。
- AI議論には選択したプロバイダのAPI利用料が発生します。
  円卓討論をOFFにするとAPI消費を抑えられます。
