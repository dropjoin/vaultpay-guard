# VaultPay Guard

**A policy-guarded AI payment agent on the PayPal sandbox.**

> VaultPay Guard is an AI payment agent that can pay invoices and
> vendors through PayPal — but every payment it proposes must pass an
> independent AI checker and a deterministic policy engine
> (per-transaction caps, daily caps, payee allowlist) before a single
> cent moves, with every decision written to an auditable ledger.

Built for the [PayPal AI Hackathon](https://paypalaihackathon.devpost.com/)
on Devpost. Solo project by M Aziz Yahya (GitHub:
[dropjoin](https://github.com/dropjoin)).

## Why

Businesses and freelancers are about to hand AI agents the ability to
pay invoices and vendors. The failure mode is obvious and scary: one
hallucinated amount, one fake invoice, one prompt-injected memo — and
real money is gone. VaultPay Guard is the seatbelt: the agent can
*propose* payments, but it cannot *decide* them alone.

## How it works

```
 request (text or structured)
   -> PROPOSER   AI role A extracts a structured payment proposal
   -> GUARDRAIL  deterministic prompt-injection scan of all
                 untrusted text; high-severity hits block outright
   -> POLICY     deterministic engine: per-tx cap, daily cap,
                 payee allowlist, allowed currencies, balance,
                 owner freeze switch. Code, not a model.
   -> CHECKER    AI role B, independent of the proposer, reviews the
                 proposal against the policy facts and the original
                 request. It can tighten, never loosen.
   -> EXECUTE    PayPal Orders v2: create order — then, after the
                 payer approves on PayPal, capture (two-phase; the
                 simulated client captures immediately)
   -> LEDGER     every decision, pass or fail, appended as one JSONL
                 record with the full reasoning trail
```

Governance rule, enforced in code (`src/agents.py`): **a model can
tighten the policy's decision, never loosen it.** If the policy engine
rejects — or the guardrail blocks — the checker model is not even
consulted. If the checker returns anything unparseable, the payment is
rejected: fail closed.

Daily-cap accounting is read back from the ledger itself, so
restarting the agent cannot reset today's budget.

## Execution states

Against the real Orders API, payment execution is **two-phase** —
that is how PayPal works, and the ledger models it honestly instead
of pretending a created order is a finished payment:

| State | Meaning |
|---|---|
| `ORDER_CREATED` | Policy + checker PASSed and the PayPal order exists (status CREATED), but the payer has **not approved yet**. Nothing has been captured. The decision verdict is PASS — the guard approved the payment — and the card shows the payer approval URL. The amount **already counts against the daily cap**: the order commits the payer, so letting it sit outside the cap would let two pending orders each pass the check and both be captured later. |
| `COMPLETED` | The payer approved and the capture succeeded. Phase 2 (`vaultpay.py capture <order_id>`, or the "Complete capture" button in the web UI) appends a `kind="capture"` record referencing the original decision; the JSONL ledger stays append-only and readers reconcile the pair. Capture is idempotent, and a failed attempt (e.g. approval still pending) is recorded as status `ERROR` and can be retried — the order stays `ORDER_CREATED`. |
| `ERROR` (on a decision) | Creating the order itself failed (API error, network). The verdict becomes REJECT — a payment the guard approved but could not execute is never reported as a success. |

In **simulated mode** there is no payer, so the simulated client
captures immediately after creating the order: a PASS ends as
`COMPLETED` in one step, exactly as the offline demo shows.

## Quick start (offline, no credentials)

Requires Python 3.10+. Standard library only — nothing to install.

```bash
python3 scripts/demo_offline.py     # 4 canonical scenarios end to end
python3 -m unittest discover -s tests -v
```

Offline mode uses the deterministic **mock** AI backend and the
**simulated** PayPal client. Both are labeled as such in every output
and ledger record; no network calls are made and no money moves.

## CLI

```bash
# natural language (proposer extracts the fields)
python3 vaultpay.py propose --text "Pay $120.00 to billing@northwind.example for hosting invoice INV-1042"

# structured
python3 vaultpay.py propose --payee vendor-hosting --amount 120.00 --currency USD --reference INV-1042

# live mode only: after the payer approves the order on PayPal,
# complete phase 2 (capture) — appends a capture record to the ledger
python3 vaultpay.py capture 12884569JK067594D

python3 vaultpay.py policy          # caps, allowlist, spent-today
python3 vaultpay.py ledger          # audit ledger summary
```

Exit codes: `0` PASS/executed, `3` REJECTED, `1` error.

## Web UI

```bash
python3 app.py        # binds 0.0.0.0:$PORT (default 8000)
```

Three pages, served by the Python standard library — still zero
dependencies:

- **`/` — New Payment.** A natural-language prompt box (with one-click
  example scenarios: a legit invoice, an over-cap payment, a
  non-allowlisted payee, a prompt-injection attempt) or structured
  fields. Submitting runs the exact same pipeline as the CLI and
  renders the decision card inline. In live mode the card for a
  passed payment shows the PayPal payer-approval link and a
  "Complete capture" button (phase 2, see Execution states).
- **`/ledger` — Audit Ledger.** The full JSONL audit trail in an
  [AG Grid](https://www.ag-grid.com/) table: sortable, filterable,
  paginated, with custom cell renderers for the verdict pill, the
  execution status pill (`AWAITING APPROVAL` vs `COMPLETED`), and the
  simulated/sandbox mode tag, a currency-aware amount formatter, and
  a per-row approval link + capture button for orders still awaiting
  capture. Decision and capture records are reconciled into one row
  per payment. If the AG Grid CDN is unreachable, a server-rendered
  table is shown instead — the audit trail is never hidden behind a
  script tag.
- **`/policy` — Policy.** The caps, balance, today's spend (read back
  from the ledger) and the payee allowlist, read live from the policy
  file.

Whenever the PayPal client runs in simulated mode (no credentials),
every page carries a prominent **SANDBOX / SIMULATED** badge — a
simulated order can never be mistaken for a real sandbox call.

JSON API: `GET /api/status`, `GET /api/ledger`, `GET /api/policy`,
`POST /api/payments`, `POST /api/capture` (phase 2 for a live
order), `GET /healthz`. State paths are configurable via
`POLICY_PATH` / `LEDGER_PATH` (or `VAULTPAY_POLICY` /
`VAULTPAY_LEDGER`).

Deployment on Render (free tier) is pre-wired: see `render.yaml`
and [DEPLOY.md](DEPLOY.md).

> Screenshots (placeholders — captured from a local run in simulated
> mode): `docs/screenshots/new-payment.png` (request form + PASS
> decision card), `docs/screenshots/ledger-ag-grid.png` (AG Grid
> audit ledger), `docs/screenshots/policy.png` (caps and allowlist).

## Going live on the PayPal sandbox

1. Create a sandbox app at
   <https://developer.paypal.com/dashboard/applications/sandbox>.
2. Copy `.env.example` to `.env` and fill in `PAYPAL_CLIENT_ID` /
   `PAYPAL_CLIENT_SECRET`. (`.env` is git-ignored; credentials are
   read from the environment only and are never logged.)
3. Optionally point the agents at a real model with `LLM_API_KEY`,
   `LLM_BASE_URL`, `LLM_MODEL` (any OpenAI-compatible endpoint).
4. Edit `config/policy.example.json` (or your own file via
   `VAULTPAY_POLICY`) to set your caps, allowlist and balance.

The PayPal integration is the **Orders API v2**: OAuth2
client-credentials token, `POST /v2/checkout/orders`,
`POST /v2/checkout/orders/{id}/capture`, `GET /v2/checkout/orders/{id}`.

## Project layout

| Path | What |
|---|---|
| `src/policy.py` | Deterministic policy engine (caps, allowlist, fail-closed) |
| `src/guardrail.py` | Prompt-injection scan for untrusted request text |
| `src/agents.py` | Proposer + checker roles and the tighten-only governance |
| `src/llm.py` | Provider-agnostic LLM layer (mock + OpenAI-compatible) |
| `src/paypal_client.py` | PayPal Orders v2 client + labeled simulated twin |
| `src/flow.py` | The end-to-end pipeline and the decision card |
| `src/ledger.py` | JSONL audit ledger; also the spent-today source of truth |
| `src/money.py` | Integer minor-unit money math (no floats) |
| `src/web.py` | Web UI + JSON API (stdlib `http.server`, AG Grid ledger) |
| `app.py` | Web entrypoint for hosting (`0.0.0.0:$PORT`) |
| `config/policy.example.json` | Example policy: caps, allowlist, currencies |
| `render.yaml` / `DEPLOY.md` | Render free-tier deployment (Blueprint + manual steps) |
| `tests/` | Unit + flow + web tests, all offline (unittest) |

## Status

Core engine, AI layer, PayPal sandbox client (real + simulated),
end-to-end flow with two-phase live execution, CLI (including the
`capture` phase-2 command), web UI with AG Grid audit ledger, and
Render deployment files are in place. Live-sandbox verification is
done: with real sandbox credentials a $25.00 in-policy payment
produced order `12884569JK067594D` (status CREATED, awaiting payer
approval) in the audit ledger on 2026-11-04. Remaining build steps:
the demo's payer-approval walkthrough (needs a sandbox buyer
login), demo video and submission.

## License

MIT — see [LICENSE](LICENSE).
