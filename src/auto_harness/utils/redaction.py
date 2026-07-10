"""Redaction checker: verify that evidence files don't contain sensitive patterns.

Scans text for:
- Tokens (HF, ModelScope, Bearer, Authorization)
- Home paths (/home/<user>, /Users/<user>)
- Public IP addresses
- Cloud key patterns (AWS, GCP, Azure)
- SSH references

Used to validate evidence files before committing to version control.
"""
import re
from pathlib import Path
from typing import Dict, List


# Patterns that indicate unredacted sensitive content
_REDACTION_PATTERNS = [
    # Tokens
    (re.compile(r'hf_[A-Za-z0-9]{20,}', re.IGNORECASE), "Hugging Face token"),
    (re.compile(r'ms_[A-Za-z0-9]{20,}', re.IGNORECASE), "ModelScope token"),
    (re.compile(r'Bearer\s+[A-Za-z0-9\-_\.]{10,}', re.IGNORECASE), "Bearer token"),
    (re.compile(r'Authorization[:\s]+[A-Za-z0-9\-_\.]{10,}', re.IGNORECASE), "Authorization header"),

    # Home paths (not redacted)
    (re.compile(r'/home/[a-z][a-z0-9_\-]{2,}(?![<])', re.IGNORECASE), "Unredacted /home/<user> path"),
    (re.compile(r'/Users/[a-z][a-z0-9_\-]{2,}(?![<])', re.IGNORECASE), "Unredacted /Users/<user> path"),

    # Public IP addresses (exclude private ranges: 10.x, 172.16-31.x, 192.168.x, 127.x)
    # This is a simplified check — matches any IP-like pattern then filters private ranges
    (re.compile(r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)(?!\d)(?!\.\d)'), "Public IP address"),

    # Cloud keys
    (re.compile(r'AKIA[A-Z0-9]{16}', re.IGNORECASE), "AWS access key"),
    (re.compile(r'AIza[A-Za-z0-9\-_]{35}', re.IGNORECASE), "GCP API key"),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.onmicrosoft\.com', re.IGNORECASE), "Azure tenant"),

    # SSH
    (re.compile(r'ssh://[^\s<]+', re.IGNORECASE), "SSH URL"),
    (re.compile(r'\.ssh/[^\s<]+', re.IGNORECASE), "SSH file path"),
]


def check_redaction(text: str) -> List[Dict]:
    """Check text for unredacted sensitive patterns.

    Args:
        text: The text to check.

    Returns:
        List of findings, each with pattern_name, match, and position.
        Empty list means no violations found.
    """
    findings = []
    for pattern, name in _REDACTION_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group()

            # Filter out private IP addresses
            if name == "Public IP address" and _is_private_ip(matched_text):
                continue

            findings.append({
                "pattern_name": name,
                "match": matched_text[:50],
                "position": match.start(),
            })
    return findings


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/reserved range."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_reserved
    except ValueError:
        return False


def check_file(path: Path) -> List[Dict]:
    """Check a file for unredacted sensitive patterns.

    Args:
        path: Path to the file to check.

    Returns:
        List of findings.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return check_redaction(text)


def check_directory(dir_path: Path) -> Dict[str, List[Dict]]:
    """Check all files in a directory for unredacted sensitive patterns.

    Args:
        dir_path: Path to the directory to check.

    Returns:
        Dict mapping filename to list of findings.
    """
    results = {}
    for path in dir_path.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            findings = check_file(path)
            if findings:
                results[str(path)] = findings
    return results
