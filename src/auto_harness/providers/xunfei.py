import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from auto_harness.providers.base import LLMResult, Message


class XunfeiSparkProvider:
    """Xunfei provider with an Anthropic-compatible default transport.

    Secrets must be injected through environment variables. Do not write real
    keys into repository files, prompts, reports, or progress logs.
    """

    def __init__(self, urlopen=None) -> None:
        self.app_id = os.environ.get("XUNFEI_APP_ID", "")
        self.api_key = os.environ.get("XUNFEI_API_KEY", "")
        self.api_secret = os.environ.get("XUNFEI_API_SECRET", "")
        self.model = os.environ.get("XUNFEI_MODEL", "")
        self.api_base = os.environ.get("XUNFEI_API_BASE", "").rstrip("/")
        self.api_url = os.environ.get("XUNFEI_API_URL", "")
        self.timeout_seconds = int(os.environ.get("XUNFEI_TIMEOUT_SECONDS", "60"))
        self.max_tokens = int(os.environ.get("XUNFEI_MAX_TOKENS", "2048"))
        self.context_window_tokens = int(
            os.environ.get("XUNFEI_CONTEXT_WINDOW_TOKENS", "0")
        )
        self.anthropic_version = os.environ.get("XUNFEI_ANTHROPIC_VERSION", "2023-06-01")
        self.urlopen = urlopen or urllib.request.urlopen

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_output_tokens: int = None,
    ) -> LLMResult:
        url = self._resolve_url()
        started = time.time()
        payload = self._build_anthropic_payload(
            messages, temperature, max_output_tokens=max_output_tokens
        )
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        for name, value in self._headers().items():
            req.add_header(name, value)
        try:
            with self.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw_text = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("Xunfei HTTP error %s: %s" % (exc.code, detail))
        latency_ms = int((time.time() - started) * 1000)
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            return LLMResult(text=raw_text, raw={"raw_text": raw_text}, latency_ms=latency_ms)
        text = self._extract_text(raw)
        return LLMResult(text=text, raw=raw, usage=raw.get("usage") if isinstance(raw, dict) else None, latency_ms=latency_ms, protocol="json_action")

    def _resolve_url(self) -> str:
        if self.api_url:
            return self.api_url
        if self.api_base:
            return self.api_base.rstrip("/") + "/v1/messages"
        raise RuntimeError("XUNFEI_API_URL or XUNFEI_API_BASE is not configured")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = "Bearer %s" % self.api_key
        if self.app_id:
            headers["X-App-Id"] = self.app_id
        return headers

    def _build_anthropic_payload(
        self,
        messages: List[Message],
        temperature: float,
        max_output_tokens: int = None,
    ) -> Dict:
        system_parts: List[str] = []
        user_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                role = msg.role if msg.role in ("user", "assistant") else "user"
                user_messages.append({"role": role, "content": msg.content})
        payload: Dict = {
            "model": self.model,
            "max_tokens": min(self.max_tokens, int(max_output_tokens))
            if max_output_tokens
            else self.max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        return payload

    def _extract_text(self, raw):
        if not isinstance(raw, dict):
            return str(raw)
        content = raw.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item["text"]))
                    elif "text" in item:
                        parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join(parts)
        if isinstance(content, str):
            return content
        if "choices" in raw and raw["choices"]:
            choice = raw["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and "content" in message:
                    return message["content"]
                if "text" in choice:
                    return choice["text"]
        for key in ("content", "text", "answer"):
            if key in raw:
                return str(raw[key])
        return json.dumps(raw, ensure_ascii=False)
