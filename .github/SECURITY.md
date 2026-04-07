# Security Policy

## Scope

ObbyAI is designed to run on a **private local network** and is not hardened for public internet exposure without additional security layers (nginx + SSL + firewall, which the default setup includes).

## Reporting a Vulnerability

If you find a security vulnerability, please **do not open a public issue**.

Instead, use GitHub's private vulnerability reporting:
**[Report a vulnerability](https://github.com/obnoxiousmods/obbyai/security/advisories/new)**

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Known Considerations

- ArangoDB credentials must be set in `.env` (never committed)
- The Ollama API on port 11434 should be firewalled from the internet
- The chat web UI proxies all Ollama requests server-side (no direct browser → Ollama)
- File uploads are processed in-memory; no files are written to disk
- No authentication layer is implemented — add nginx basic auth or OAuth if exposing publicly
