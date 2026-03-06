# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| latest (main branch) | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in PaperBanana, please report it responsibly.

**Do not open a public issue.** Instead, email **dip@llmsresearch.com** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix, if you have one

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days for critical issues.

## Security Considerations

PaperBanana interacts with external APIs and processes user-provided text. Users should be aware of the following:

**API Keys**: Your Google Gemini API key is stored locally in `.env`. Never commit this file to version control. The `.gitignore` already excludes it.

**Generated Code Execution**: The `paperbanana plot` command generates and executes Matplotlib code via Gemini. While the execution environment is sandboxed to plotting functions, users should review generated code before running it in production environments.

**MCP Server**: The MCP server exposes PaperBanana's functionality to IDE clients. It runs locally and does not accept remote connections by default. Do not expose the MCP server to untrusted networks.

**Reference Data**: The reference dataset contains publicly available figures from arXiv papers, used under fair use for research purposes. No proprietary or restricted data is included.

## Dependencies

We monitor dependencies for known vulnerabilities through GitHub's Dependabot. If you notice a vulnerability in one of our dependencies, please report it through the process above.
