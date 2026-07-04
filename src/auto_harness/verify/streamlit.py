import urllib.parse
import urllib.request
from typing import Dict


class StreamlitVerifier:
    ERROR_MARKERS = (
        "Traceback (most recent call last)",
        "ModuleNotFoundError",
        "ImportError",
        "Exception",
        "ConnectionError",
    )

    def __init__(self, urlopen=None) -> None:
        self.urlopen = urlopen or urllib.request.urlopen

    def probe(self, endpoint: str, trace_id: str) -> Dict:
        url = self._append_trace(endpoint, trace_id)
        try:
            req = urllib.request.Request(url, method="GET")
            with self.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                status_code = getattr(resp, "status", None) or getattr(resp, "code", None)
        except Exception as exc:  # noqa: BLE001 - stored as evidence
            return {
                "name": "streamlit_dom_probe",
                "status": "uncertain",
                "evidence": {"url": url, "error": str(exc)},
                "reason": "Streamlit page fetch failed",
            }

        markers = self._markers(body)
        if markers["error_markers"]:
            status = "fail"
            reason = "Streamlit page contains error markers"
        elif markers["streamlit_markers"]:
            status = "pass" if trace_id in body else "uncertain"
            reason = "Streamlit page loaded; trace observed" if status == "pass" else "Streamlit page loaded but trace was not observed"
        else:
            status = "uncertain"
            reason = "Streamlit markers not found"
        return {
            "name": "streamlit_dom_probe",
            "status": status,
            "evidence": {
                "url": url,
                "status_code": status_code,
                "body_tail": body[-2000:],
                "markers": markers,
            },
            "reason": reason,
        }

    def _append_trace(self, endpoint: str, trace_id: str) -> str:
        parsed = urllib.parse.urlparse(endpoint)
        query = urllib.parse.parse_qs(parsed.query)
        query["_auto_harness_trace"] = [trace_id]
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))

    def _markers(self, body: str) -> Dict:
        lower = body.lower()
        streamlit = []
        for marker in ("streamlit", "stapp", "data-testid"):
            if marker in lower:
                streamlit.append(marker)
        errors = [marker for marker in self.ERROR_MARKERS if marker in body]
        return {
            "streamlit_markers": streamlit,
            "error_markers": errors,
        }
