"""Deterministic model source clients (Document A Phase A3).

These clients resolve mutable revisions to immutable commit SHAs, fetch model
config, and list files — strictly against the official host for each source.

Security invariants:
  - only the official host per source is contacted
  - redirects are re-checked against the allowed host
  - an injected transport is used; tests never touch the public network
  - response size, file count, JSON depth, and timeouts are bounded
  - 429 honours a bounded Retry-After; 401/403 never leak the token
  - tokens are only read from env vars and sent in request headers, never
    persisted into any Artifact or log
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from auto_harness.model_runtime.schemas import hash_payload

# Limits to keep hostile/inflated responses from exhausting the resolver.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_FILE_COUNT = 20000
MAX_JSON_DEPTH = 64
DEFAULT_TIMEOUT_SECONDS = 60
MAX_RETRY_AFTER_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 3

COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

ALLOWED_HOSTS = {
    "huggingface": frozenset({"huggingface.co"}),
    "modelscope": frozenset({"www.modelscope.cn", "modelscope.cn"}),
}

TOKEN_ENV = {
    "huggingface": "HF_TOKEN",
    "modelscope": "MODELSCOPE_TOKEN",
}

API_BASES = {
    "huggingface": "https://huggingface.co",
    "modelscope": "https://www.modelscope.cn",
}


class SourceClientError(Exception):
    """A deterministic, redacted source failure with a machine-readable status."""

    def __init__(self, status: str, message: str = "") -> None:
        self.status = status
        super().__init__(message or status)


class TransportResponse:
    """Minimal HTTP response abstraction returned by an injected transport."""

    def __init__(self, status: int, body: bytes, headers: Optional[Dict[str, str]] = None, url: str = "") -> None:
        self.status = int(status)
        self._body = body if isinstance(body, bytes) else bytes(body)
        self.headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        self.url = url

    def read(self) -> bytes:
        return self._body

    def read_text(self, encoding: str = "utf-8") -> str:
        return self._body.decode(encoding, errors="replace")


class UrlopenTransport:
    """Default transport backed by urllib (used only outside of unit tests)."""

    def request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> TransportResponse:
        req = urllib.request.Request(url, method=method)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return TransportResponse(
                status=getattr(resp, "status", 200) or 200,
                body=body,
                headers=dict(resp.headers.items()) if hasattr(resp, "headers") else {},
                url=getattr(resp, "geturl", lambda: url)(),
            )


class ModelSourceClient:
    """Base source client with host enforcement, retries, and auth."""

    source: str = ""
    token_env: str = ""

    def __init__(
        self,
        transport: Optional[Callable[..., TransportResponse]] = None,
        token: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        api_base: Optional[str] = None,
    ) -> None:
        self._transport = transport or UrlopenTransport()
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.api_base = (api_base or API_BASES.get(self.source, "")).rstrip("/")
        self._token = token if token is not None else os.environ.get(self.token_env)

    # ---- public interface ----

    def resolve_revision(self, repo_id: str, requested_revision: str) -> Dict[str, Any]:
        raise NotImplementedError

    def fetch_model_config(self, repo_id: str, resolved_revision: str) -> Dict[str, Any]:
        raise NotImplementedError

    def fetch_file(self, repo_id: str, resolved_revision: str, path: str) -> bytes:
        raise NotImplementedError

    def list_files(self, repo_id: str, resolved_revision: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def auth_summary(self) -> Dict[str, Any]:
        return {
            "auth_env_name": self.token_env,
            "auth_present": bool(self._token),
        }

    # ---- shared plumbing ----

    def _get(self, url: str) -> TransportResponse:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer %s" % self._token
        attempts = self.max_attempts
        last_error: Optional[SourceClientError] = None
        for attempt in range(attempts):
            try:
                response = self._transport.request(
                    url, method="GET", headers=headers, timeout=self.timeout
                )
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
                last_error = SourceClientError("network_failed", self._redact(str(exc)))
                if attempt == attempts - 1:
                    break
                time.sleep(min(1.0 * (attempt + 1), 5.0))
                continue
            self._check_host(url, response.url)
            if response.status == 429:
                retry_after = self._bounded_retry_after(response.headers.get("retry-after", ""))
                if attempt == attempts - 1 or retry_after is None:
                    last_error = SourceClientError("retry_exhausted", "429 too many requests")
                    break
                time.sleep(retry_after)
                continue
            if response.status in (401, 403):
                status = self._auth_failure_status(response)
                raise SourceClientError(status, "authentication or authorization failed")
            if response.status == 404:
                raise SourceClientError("not_found", "model not found")
            if response.status >= 500:
                last_error = SourceClientError("network_failed", "upstream 5xx")
                if attempt == attempts - 1:
                    break
                time.sleep(min(1.0 * (attempt + 1), 5.0))
                continue
            if response.status not in (200, 201):
                raise SourceClientError("network_failed", "unexpected HTTP %s" % response.status)
            body = response.read()
            if len(body) > MAX_RESPONSE_BYTES:
                raise SourceClientError("metadata_invalid", "response exceeds size limit")
            return TransportResponse(response.status, body, response.headers, response.url)
        raise last_error or SourceClientError("network_failed", "request failed")

    def _check_host(self, requested_url: str, final_url: str) -> None:
        if not final_url:
            return
        host = (urlparse(final_url).hostname or "").lower()
        if host not in ALLOWED_HOSTS[self.source]:
            raise SourceClientError(
                "network_failed",
                "redirect or request resolved to disallowed host: %s" % host,
            )

    def _bounded_retry_after(self, value: str) -> Optional[float]:
        if not value:
            return None
        try:
            seconds = float(value)
        except ValueError:
            return None
        if seconds < 0:
            return None
        return min(seconds, MAX_RETRY_AFTER_SECONDS)

    def _auth_failure_status(self, response: TransportResponse) -> str:
        body = response.read_text().lower()
        if "gated" in body or "license" in body or "terms" in body:
            return "license_acceptance_required"
        return "access_required"

    def _parse_json(self, body: bytes, context: str) -> Any:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceClientError("metadata_invalid", "%s: non-utf8 body" % context) from exc
        try:
            value = json.loads(text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise SourceClientError("metadata_invalid", "%s: invalid JSON" % context) from exc
        self._check_depth(value, 0, MAX_JSON_DEPTH, context)
        return value

    def _check_depth(self, value: Any, depth: int, limit: int, context: str) -> None:
        if depth > limit:
            raise SourceClientError("metadata_invalid", "%s: JSON depth exceeds limit" % context)
        if isinstance(value, dict):
            for child in value.values():
                self._check_depth(child, depth + 1, limit, context)
        elif isinstance(value, list):
            for child in value:
                self._check_depth(child, depth + 1, limit, context)

    @staticmethod
    def _redact(message: str) -> str:
        return re.sub(r"(hf_|ms_|Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1***", message, flags=re.IGNORECASE)

    @staticmethod
    def _quote(repo_id: str) -> str:
        return urllib.parse.quote(repo_id, safe="/")


class HuggingFaceSourceClient(ModelSourceClient):
    source = "huggingface"
    token_env = "HF_TOKEN"

    def resolve_revision(self, repo_id: str, requested_revision: str) -> Dict[str, Any]:
        repo = self._quote(repo_id)
        rev = urllib.parse.quote(requested_revision or "main", safe="")
        url = "%s/api/models/%s/revision/%s" % (self.api_base, repo, rev)
        response = self._get(url)
        payload = self._parse_json(response.read(), "revision resolution")
        sha = payload.get("sha") or payload.get("revision") or ""
        if not COMMIT_SHA.match(str(sha)):
            raise SourceClientError("metadata_invalid", "resolved revision is not an immutable commit SHA")
        return {
            "resolved_revision": str(sha),
            "gated": bool(payload.get("gated", False)),
            "license": str(payload.get("cardData", {}).get("license", "") if isinstance(payload.get("cardData"), dict) else ""),
            "private": bool(payload.get("private", False)),
        }

    def fetch_model_config(self, repo_id: str, resolved_revision: str) -> Dict[str, Any]:
        repo = self._quote(repo_id)
        rev = urllib.parse.quote(resolved_revision, safe="")
        url = "%s/%s/resolve/%s/config.json" % (self.api_base, repo, rev)
        response = self._get(url)
        payload = self._parse_json(response.read(), "model config")
        if not isinstance(payload, dict):
            raise SourceClientError("metadata_invalid", "model config must be an object")
        return payload

    def fetch_file(self, repo_id: str, resolved_revision: str, path: str) -> bytes:
        repo = self._quote(repo_id)
        rev = urllib.parse.quote(resolved_revision, safe="")
        rel = urllib.parse.quote(path, safe="/")
        url = "%s/%s/resolve/%s/%s" % (self.api_base, repo, rev, rel)
        response = self._get(url)
        return response.read()

    def list_files(self, repo_id: str, resolved_revision: str) -> List[Dict[str, Any]]:
        repo = self._quote(repo_id)
        rev = urllib.parse.quote(resolved_revision, safe="")
        url = "%s/api/models/%s/tree/%s?recursive=true" % (self.api_base, repo, rev)
        response = self._get(url)
        payload = self._parse_json(response.read(), "file tree")
        if not isinstance(payload, list):
            raise SourceClientError("metadata_invalid", "file tree must be a list")
        if len(payload) > MAX_FILE_COUNT:
            raise SourceClientError("metadata_invalid", "file tree exceeds file count limit")
        files = []
        for item in payload:
            if not isinstance(item, dict) or item.get("type") not in (None, "file"):
                continue
            path = item.get("path") or item.get("rfilename") or ""
            if not path:
                continue
            files.append({
                "path": path,
                "size_bytes": item.get("size"),
                "sha256": item.get("sha256"),
                "etag": item.get("oid") or item.get("blob_id") or (item.get("lfs") or {}).get("oid"),
            })
        return files


class ModelScopeSourceClient(ModelSourceClient):
    source = "modelscope"
    token_env = "MODELSCOPE_TOKEN"

    def __init__(self, *args, download_base: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.download_base = (
            download_base
            or os.environ.get("MODELSCOPE_DOWNLOAD_BASE")
            or "https://www.modelscope.cn/models"
        ).rstrip("/")

    def resolve_revision(self, repo_id: str, requested_revision: str) -> Dict[str, Any]:
        repo = self._quote(repo_id)
        rev = urllib.parse.quote(requested_revision or "master", safe="")
        url = "%s/%s/repo/files?Revision=%s" % (self.api_base, repo, rev)
        response = self._get(url)
        payload = self._parse_json(response.read(), "revision resolution")
        data = payload.get("Data") if isinstance(payload, dict) else payload
        if not isinstance(data, dict):
            raise SourceClientError("metadata_invalid", "modelscope revision payload invalid")
        sha = data.get("Revision") or data.get("CommitID") or data.get("CommitId") or ""
        if not COMMIT_SHA.match(str(sha)):
            raise SourceClientError("metadata_invalid", "resolved revision is not an immutable commit SHA")
        return {
            "resolved_revision": str(sha),
            "gated": False,
            "license": str(data.get("License") or data.get("license") or ""),
            "private": bool(data.get("Private") or False),
        }

    def fetch_model_config(self, repo_id: str, resolved_revision: str) -> Dict[str, Any]:
        repo = self._quote(repo_id)
        rev = urllib.parse.quote(resolved_revision, safe="")
        url = "%s/%s/resolve/%s/config.json" % (self.download_base, repo, rev)
        response = self._get(url)
        payload = self._parse_json(response.read(), "model config")
        if not isinstance(payload, dict):
            raise SourceClientError("metadata_invalid", "model config must be an object")
        return payload

    def list_files(self, repo_id: str, resolved_revision: str) -> List[Dict[str, Any]]:
        repo = self._quote(repo_id)
        rev = urllib.parse.quote(resolved_revision, safe="")
        url = "%s/%s/repo/files?Revision=%s&Recursive=true" % (self.api_base, repo, rev)
        response = self._get(url)
        payload = self._parse_json(response.read(), "file tree")
        data = payload.get("Data") if isinstance(payload, dict) else payload
        items = data.get("Files") if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = payload.get("files") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            raise SourceClientError("metadata_invalid", "file tree must be a list")
        if len(items) > MAX_FILE_COUNT:
            raise SourceClientError("metadata_invalid", "file tree exceeds file count limit")
        files = []
        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("Path") or item.get("path") or item.get("Name") or item.get("name") or ""
            if not path:
                continue
            files.append({
                "path": path,
                "size_bytes": item.get("Size") or item.get("size"),
                "sha256": item.get("Sha256") or item.get("sha256"),
                "etag": item.get("Sha256") or item.get("sha256") or item.get("Revision"),
            })
        return files


def source_client_for(source: str, transport=None, token: Optional[str] = None) -> ModelSourceClient:
    """Build a source client for a validated source string."""
    if source == "huggingface":
        return HuggingFaceSourceClient(transport=transport, token=token)
    if source == "modelscope":
        return ModelScopeSourceClient(transport=transport, token=token)
    raise ValueError("unsupported model source: %s" % source)


def source_metadata_hash(config: Dict[str, Any], files: List[Dict[str, Any]], resolved_revision: str) -> str:
    """Bind the source metadata (config + file closure + commit) to a hash."""
    payload = {
        "resolved_revision": resolved_revision,
        "config": config,
        "files": sorted(
            (
                {
                    "path": f.get("path"),
                    "size_bytes": f.get("size_bytes"),
                    "sha256": f.get("sha256"),
                    "etag": f.get("etag"),
                }
                for f in files
            ),
            key=lambda item: item.get("path") or "",
        ),
    }
    return hash_payload(payload)
