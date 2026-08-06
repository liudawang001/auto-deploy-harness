# Security policy

## Supported versions

Security fixes are applied to the latest release and the current default branch.

## Reporting a vulnerability

Use GitHub Private Vulnerability Reporting for this repository. Do not open a public issue containing credentials, exploit details, private logs, or deployment artifacts. If private reporting is not enabled, open a minimal issue requesting a private contact channel without including sensitive details.

Never submit API keys. Revoke any credential that has been pasted into a terminal, chat, issue, log, or test fixture.

## Runtime boundary

This tool launches third-party project code. The local backend is intended for trusted repositories only. Prefer the Docker backend with least privilege for untrusted code, and review every permission that enables dependency installation, source edits, network access, GPU access, or service startup. Child processes receive a minimal allowlisted environment; provider credentials are not forwarded.

Dry-run is analysis, not proof that a deployment is safe or successful. External network, Docker/GPU, and real-provider validation remain separate release gates.
