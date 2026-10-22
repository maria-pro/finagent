# Security

FinAgent is a teaching project, but it handles API keys and can be served to other people, so
security reports are welcome.

## Reporting a problem

Please report security problems privately through GitHub's private vulnerability reporting (the
repository's *Security* tab, then *Report a vulnerability*) rather than in a public issue.

## Running it safely

- The web UI has no login and listens on your own computer only. Read *Sharing the web UI* in the
  README before serving it to anyone else.
- Keep API keys in environment variables or `.env`, which git ignores. The web UI never sends a key
  held by the server to the browser, and uses it only with the endpoint it is configured for.
- Run records and notes contain the questions people asked; treat `reports/` accordingly.
- Filing text is untrusted input, and the guardrails are a teaching example, not a compliance
  control; see the README's limitations.
