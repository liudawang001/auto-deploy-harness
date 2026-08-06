"""Tests for redaction checker (Phase 6).

Verifies:
- Redaction checker detects unredacted tokens
- Redaction checker detects unredacted home paths
- Redaction checker detects public IP addresses
- Redaction checker detects cloud key patterns
- Redaction checker passes clean (redacted) text
- Redaction checker passes evidence files
"""
import tempfile
import unittest
from pathlib import Path

from auto_harness.utils.redaction import check_redaction, check_file, check_directory


class TestRedactionChecker(unittest.TestCase):
    """Test redaction checker."""

    def test_detects_hf_token(self):
        """Detects unredacted Hugging Face token."""
        text = 'export HF_TOKEN="hf_abcdefghijklmnopqrstuvwx"'
        findings = check_redaction(text)
        self.assertGreater(len(findings), 0)
        self.assertIn("Hugging Face", findings[0]["pattern_name"])

    def test_detects_bearer_token(self):
        """Detects unredacted Bearer token."""
        text = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
        findings = check_redaction(text)
        self.assertGreater(len(findings), 0)

    def test_detects_home_path(self):
        """Detects unredacted /home/<user> path."""
        text = "cache_path: /home/myuser/.cache/huggingface"
        findings = check_redaction(text)
        self.assertGreater(len(findings), 0)
        self.assertIn("home", findings[0]["pattern_name"].lower())

    def test_detects_users_path(self):
        """Detects unredacted /Users/<user> path."""
        text = "path: /Users/johndoe/projects/model"
        findings = check_redaction(text)
        self.assertGreater(len(findings), 0)

    def test_detects_aws_key(self):
        """Detects unredacted AWS access key."""
        # Construct the documented synthetic example at runtime so a static
        # repository scan does not mistake the fixture for a live credential.
        text = "aws_access_key_id: " + "AKIA" + "IOSFODNN7EXAMPLE"
        findings = check_redaction(text)
        self.assertGreater(len(findings), 0)
        self.assertIn("AWS", findings[0]["pattern_name"])

    def test_detects_public_ip(self):
        """Detects public IP address."""
        text = "server_ip: 8.8.8.8"
        findings = check_redaction(text)
        self.assertGreater(len(findings), 0)
        self.assertIn("IP", findings[0]["pattern_name"])

    def test_passes_redacted_text(self):
        """Passes text with proper redaction markers."""
        text = 'export HF_TOKEN="hf_<REDACTED>"\nserver_ip: <REDACTED_IP>\npath: /home/<USER>/.cache'
        findings = check_redaction(text)
        self.assertEqual(len(findings), 0, "Redacted text should have no findings")

    def test_passes_clean_evidence(self):
        """Passes properly redacted evidence content."""
        text = """{
  "model_name": "Qwen/Qwen2.5-0.5B",
  "download_token_redacted": "hf_<REDACTED>",
  "cache_path_redacted": "/home/<USER>/.cache/huggingface",
  "deploy_url_redacted": "http://<REDACTED_IP>:7860"
}"""
        findings = check_redaction(text)
        self.assertEqual(len(findings), 0)

    def test_check_file(self):
        """check_file reads and checks a file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            path.write_text('{"token": "hf_<REDACTED>"}', encoding="utf-8")
            findings = check_file(path)
            self.assertEqual(len(findings), 0)

    def test_check_file_detects_violation(self):
        """check_file detects violations in a file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"token": "hf_abcdefghijklmnopqrstuvwx"}', encoding="utf-8")
            findings = check_file(path)
            self.assertGreater(len(findings), 0)

    def test_check_directory(self):
        """check_directory scans all files in a directory."""
        with tempfile.TemporaryDirectory() as tmp:
            # Clean file
            (Path(tmp) / "clean.json").write_text('{"token": "hf_<REDACTED>"}', encoding="utf-8")
            # Bad file
            (Path(tmp) / "bad.json").write_text('{"token": "hf_abcdefghijklmnopqrstuvwx"}', encoding="utf-8")

            results = check_directory(Path(tmp))
            self.assertIn(str(Path(tmp) / "bad.json"), results)
            self.assertNotIn(str(Path(tmp) / "clean.json"), results)

    def test_private_ips_not_flagged(self):
        """Private IP addresses (10.x, 192.168.x) are not flagged."""
        text = "local_url: http://127.0.0.1:7860\ninternal: http://10.0.0.1:8080\nlan: http://192.168.1.100:3000"
        findings = check_redaction(text)
        # 127.0.0.1 and 10.0.0.1 and 192.168.1.100 should not be flagged as public IPs
        ip_findings = [f for f in findings if "IP" in f["pattern_name"]]
        self.assertEqual(len(ip_findings), 0, "Private IPs should not be flagged")


if __name__ == "__main__":
    unittest.main()
