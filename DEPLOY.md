# Deploying VaultPay Guard

VaultPay Guard is a zero-dependency Python app (standard library
only). The web entrypoint is `app.py`; it binds `0.0.0.0:$PORT`
(Render supplies `PORT`). Nothing needs to be pip-installed and
there is no build step.

## Option A — Render Blueprint (recommended)

1. Push this repository to GitHub (public repo; it already contains
   an MIT `LICENSE`).
2. In the [Render dashboard](https://dashboard.render.com/) choose
   **New → Blueprint**, connect the repository, and select
   `render.yaml`.
3. When prompted for the `sync: false` variables, paste the PayPal
   **sandbox** credentials (step 1 of "Getting sandbox credentials"
   below). Leave them empty to deploy in SIMULATED mode — the UI
   will say so loudly on every page.
4. Render creates the free web service `vaultpay-guard` with
   health checks on `/healthz`. The public URL appears in the
   dashboard once the first deploy finishes.

## Option B — Render manual web service

1. **New → Web Service**, connect this repository.
2. Runtime: **Python 3**; Plan: **Free**.
3. Build command: `echo 'No build step: Python standard library only'`
4. Start command: `python3 app.py`
5. Health check path: `/healthz`
6. Add the environment variables listed in `render.yaml`
   (same names, same meanings).

## Option C — run it anywhere else

```bash
PORT=8000 python3 app.py        # then open http://localhost:8000/
```

## Getting sandbox credentials

1. Log in at <https://developer.paypal.com/> and open
   **Apps & Credentials → Sandbox**.
2. Create an app (any name, e.g. `vaultpay-guard`); copy its
   **Client ID** and **Secret**.
3. Set them as `PAYPAL_CLIENT_ID` / `PAYPAL_CLIENT_SECRET` (Render
   env vars, or a local `.env` copied from `.env.example`).
4. Restart the service. The badge at the top of every page switches
   from "SANDBOX / SIMULATED MODE" to "PayPal SANDBOX API", and
   payments now create real **sandbox** orders — test money only.
   Capture follows payer approval as a separate phase-2 step (see
   "Execution states" in the README). Never put live credentials into
   a hackathon demo; `PAYPAL_ENV` stays `sandbox`.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `PORT` | Port to bind (Render sets this) | `8000` |
| `HOST` | Interface to bind | `0.0.0.0` |
| `PAYPAL_CLIENT_ID` / `PAYPAL_CLIENT_SECRET` | Sandbox app credentials; empty = simulated mode | (empty) |
| `PAYPAL_ENV` | `sandbox` or `live` | `sandbox` |
| `LLM_API_KEY` | OpenAI-compatible API key; empty = deterministic mock model | (empty) |
| `LLM_BASE_URL` / `LLM_MODEL` | Model endpoint / name | `https://api.openai.com/v1` / `gpt-4o-mini` |
| `POLICY_PATH` | Policy JSON (caps + allowlist) | `config/policy.example.json` |
| `LEDGER_PATH` | JSONL audit ledger file | `data/audit-ledger.jsonl` |

## Known free-tier caveat

Render's free plan has an **ephemeral filesystem**: the audit ledger
(`LEDGER_PATH`) and any local edits reset when the service redeploys
or restarts, and the service spins down after inactivity (the first
request afterwards takes ~a minute). That is acceptable for judging
a demo — the ledger rebuilds as payments are proposed, and daily-cap
accounting is derived from the ledger file itself. For a persistent
deployment, attach a Render persistent disk (paid plans) and point
`LEDGER_PATH` at it, or move the ledger behind a database in a later
iteration.
