"""LLMクライアント（Claude / Gemini 切替対応）。

両プロバイダを共通インターフェース `LLMClient.complete()` で扱う。
- Claude: Anthropic 公式SDK。既定モデルは claude-opus-4-8（最新・最高性能）。
- Gemini: google-generativeai。既定モデルは gemini-2.5-flash。
"""

from __future__ import annotations

import os

# 既定モデル
DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

PROVIDERS = ("claude", "gemini")


class LLMError(RuntimeError):
    """LLM呼び出しに関する例外。"""


def resolve_api_key(provider: str, explicit: str | None = None) -> str | None:
    """UI入力 → 環境変数 の順でAPIキーを解決する。"""
    if explicit:
        return explicit.strip()
    if provider == "claude":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return None


class LLMClient:
    def __init__(
        self,
        provider: str = "claude",
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise LLMError(f"未対応のプロバイダです: {provider}")
        self.provider = provider
        self.api_key = resolve_api_key(provider, api_key)
        if not self.api_key:
            raise LLMError(
                f"{provider} のAPIキーが設定されていません。"
                "サイドバーで入力するか環境変数を設定してください。"
            )
        self.model = model or (
            DEFAULT_CLAUDE_MODEL if provider == "claude" else DEFAULT_GEMINI_MODEL
        )
        self._client = None  # 遅延初期化

    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "claude":
            try:
                import anthropic
            except ImportError as exc:  # noqa: BLE001
                raise LLMError("anthropic パッケージが見つかりません。") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            try:
                import google.generativeai as genai
            except ImportError as exc:  # noqa: BLE001
                raise LLMError("google-generativeai パッケージが見つかりません。") from exc
            genai.configure(api_key=self.api_key)
            self._client = genai

    # ------------------------------------------------------------------ #
    def complete(self, system: str, prompt: str, max_tokens: int = 8000) -> str:
        """system指示とユーザープロンプトから応答テキストを得る。"""
        self._ensure_client()
        if self.provider == "claude":
            return self._complete_claude(system, prompt, max_tokens)
        return self._complete_gemini(system, prompt, max_tokens)

    def _complete_claude(self, system: str, prompt: str, max_tokens: int) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                # 議論は複雑な推論を要するため適応的思考を有効化
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Claude呼び出しエラー: {exc}") from exc
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts).strip()

    def _complete_gemini(self, system: str, prompt: str, max_tokens: int) -> str:
        try:
            model = self._client.GenerativeModel(
                model_name=self.model,
                system_instruction=system,
            )
            resp = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens},
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini呼び出しエラー: {exc}") from exc
        try:
            return (resp.text or "").strip()
        except Exception:  # noqa: BLE001 — 安全フィルタ等で text が無い場合
            return "（Geminiから有効な応答が得られませんでした。）"
