# ETI — Evolutionary Trading Intelligence

ETI is a research-first quantitative and AI trading intelligence project. The v0.2 architecture provides reproducible experiments, deterministic risk controls, transparent strategy comparisons, and a mobile-friendly research dashboard.

## Safety status

- Research-only: no live order placement routes or authenticated exchange actions exist.
- Training is not automatic: model fitting is an explicit CLI action and is not part of startup or CI.
- No profitability claim: backtests do not establish future performance.
- Independent risk: model decisions must pass deterministic risk and portfolio checks; defaults disable shorts and leverage above 1x.
- No API secrets: market discovery uses public CCXT endpoints only.

## Architecture implemented

| Layer | Current state |
|---|---|
| Core | Modular configuration, provider interface and documented service boundaries |
| Market data | Public OHLCV for Binance/MEXC, completed-candle filtering, CSV input |
| Market catalog | Canonical venue-independent instrument specifications and filtered instrument catalog; current live discovery is crypto spot only |
| Adapter registry | Venue adapters advertise asset-class/market-type capabilities; unsupported combinations fail closed |
| Quantitative | Causal technical features, MA and momentum baselines |
| AI | From-scratch HistGradientBoosting baseline; nested CLI evaluation compares it with a majority-class baseline; no model trained on startup or in CI |
| Evolution | Constrained candidate search on caller-supplied validation data |
| Risk | Independent trade risk gate, directional stops, leverage, drawdown and exposure checks |
| Portfolio | Gross, single-asset, correlated-group and position-count limits; now mandatory in combined decision gate |
| Backtesting | Bar-based simulator with strict index/input validation, fees, slippage, optional funding cost and equity accounting |
| Evaluation | Purged nested walk-forward, validation-only candidate selection, untouched test predictions, classification diagnostics, cost-aware spot and hypothetical long/short simulation, drawdown/Sharpe/Sortino/trade statistics, buy-and-hold and no-trade baselines |
| Decision | Probability-to-action conversion followed by independent trade-risk AND portfolio approval |
| API | Research FastAPI endpoints for risk, portfolio, combined decision and provider capability status; no order route |
| UI | Responsive PWA dashboard with combined decision gate simulator and provider capability display |
| Mobile | Browser-installable PWA foundation; native Android and on-device inference remain later milestones |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
uvicorn eti.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`; health check is `/health`.

## Authentication foundation (new)

- Invite-only account creation at `POST /v1/auth/register`; login at `POST /v1/auth/token`; protected identity at `GET /v1/profile`.
- Passwords are hashed with scrypt; sessions are HMAC-SHA256 signed bearer tokens with a 15-minute default expiry.
- The server must set `ETI_AUTH_SECRET` (at least 32 bytes) and `ETI_REGISTRATION_KEY` (at least 16 characters). Never place either value in frontend code, Git, notebooks, or client settings.
- Registration remains closed when the invite key is absent. This is deliberately not open self-service signup.
- Login attempts are throttled per client peer address and normalized username; registration attempts are throttled per peer address. Counters are persisted in SQLite and identifiers are HMACed before storage. The ASGI peer address is used; arbitrary forwarded-IP headers are not trusted.
- Example server environment:
  ```bash
  export ETI_AUTH_SECRET='replace-with-a-long-random-secret-of-at-least-32-bytes'
  export ETI_REGISTRATION_KEY='set-a-private-invite-key-of-at-least-16-chars'
  export ETI_DATABASE_PATH='data/eti.sqlite3'
  uvicorn eti.api:app --host 0.0.0.0 --port 8000
  ```
- The account identity in protected routes comes from a verified bearer token, never a client-supplied owner id.
- This is a foundation, not public-production identity management. Before public signup/deployment, add email verification, edge-level rate limiting (in addition to the application throttle), trusted-proxy configuration, HTTPS, secure secret management, backups, monitoring, and PostgreSQL for multi-instance use.
- Workspace memory and agent task/message routes are protected by account-scoped bearer authentication. Model registry mutations require the server-configured administrator allowlist.

## Existing research workflow

Training remains an explicit CLI action and is not started by API startup or CI. No live order placement endpoints exist. Current live public market discovery is crypto spot on Binance/MEXC only; other labelled asset classes are available only through validated local CSV OHLCV data.

The model registry stores immutable version metadata and checksums, not model binaries. A separate allowlisted worker can execute the first deterministic public-market research capability; this is not yet a general LLM agent network. The PWA remains an undeployed frontend shell until a backend and HTTPS host are configured.


## Authenticated workspace and agent APIs (new)

All routes in this section require `Authorization: Bearer <access_token>`. The account ID is derived from the verified token; clients cannot submit an owner ID.

- `GET/POST /v1/workspaces`: list or create private workspaces.
- `GET /v1/workspaces/{workspace_id}/memory`: list memory items (optional `namespace` filter).
- `GET/PUT/DELETE /v1/workspaces/{workspace_id}/memory/{namespace}/{key}`: read, write or delete a JSON memory item.
- `GET/POST /v1/workspaces/{workspace_id}/agents/tasks`: list or queue durable task records.
- `GET/PATCH /v1/workspaces/{workspace_id}/agents/tasks/{task_id}`: read a task or apply a guarded status transition.
- `GET/POST /v1/workspaces/{workspace_id}/agents/messages`: list or append typed JSON messages.
- `GET /v1/audit/events`: list the authenticated account's audit history.

Every workspace route checks account ownership and returns not-found for another account's workspace. Task and message payloads are stored as typed JSON data, never executed as code or instructions. Task records can be processed by the separately started allowlisted worker described below.


## PWA account workspace

The PWA now includes sign-in, invite-only account creation, workspace selection, JSON memory editing and task queue visibility. The bearer token is held in browser `sessionStorage` for the current tab session and is not written into the service-worker cache. Sign out clears the token from the current session.

The service worker caches only the static app shell. It bypasses all `/v1/` endpoints, and the API sends `Cache-Control: no-store` for authenticated and dynamic routes. Security headers include a restrictive same-origin Content Security Policy, frame protection and MIME sniffing protection.

The PWA still requires a deployed HTTPS backend and configured server secrets before sign-in can work from a public URL. A native Android application is not yet implemented.


## Authenticated model registry API

- Signed-in accounts can list approved model versions and retrieve approved metadata.
- Mutating the shared registry requires a username explicitly listed in server-only `ETI_ADMIN_USERNAMES` (comma-separated).
- Admins can register candidate metadata, request independent approval and retire approved versions. Promotion still requires a passing evaluation report, calibrated status and an approver different from the creator.
- The API derives creator and approver identities from the verified account token; clients cannot set these fields.
- Candidate and retired versions are hidden from ordinary accounts.
- The registry remains metadata-only: it does not host/download model binaries, run inference, train models or auto-promote candidates.


## Durable agent worker (first execution capability)

The API exposes the authenticated capability list at `GET /v1/agents/capabilities`. The first built-in capability is `market.snapshot`: it fetches completed public crypto spot candles from Binance or MEXC and returns a descriptive snapshot with causal technical indicators. MEXC is the default public venue; a task may explicitly select Binance. The PWA populates its capability picker from the authenticated server allowlist. It does not train a model, recommend a trade, access private exchange accounts or place orders.

Start a worker explicitly from the same server environment and database as the API:

```bash
python -m eti.worker --once
python -m eti.worker --poll-seconds 2
```

- `--once` claims and processes at most one queued task; the polling mode runs continuously until stopped.
- Task claiming uses an atomic SQLite transaction and a lease. Expired worker leases are requeued until the task's configured attempt limit (default 3, maximum 10); exhausted tasks fail closed.
- Unknown capability names fail as tasks; user payloads cannot register new handlers or execute code.
- The worker is not started automatically by FastAPI. Deploy it as a separate process only after configuring the shared persistent database and monitoring.
- SQLite remains a local/single-instance foundation. Multi-instance worker fleets require a PostgreSQL-backed queue and operational monitoring before production use.
- This is deterministic tool execution, not yet an LLM-driven agent planner, multi-agent reasoning system or autonomous trading system.


The PWA also displays the authenticated account's audit trail alongside its workspace memory and task ledger.


## Model artifacts and approved inference

- `ModelArtifactStore` stores fitted ETI `BaselineModel` objects under a content-addressed SHA-256 path and returns an `artifact://sha256/<digest>` reference.
- Artifacts are immutable by digest; inference verifies the registry checksum before loading and rejects path-based or mismatched references.
- The artifact directory is server-owned. Joblib/pickle deserialization can execute code, so **never accept model files from users or place untrusted files in `ETI_ARTIFACT_DIR`**. Only trusted server-side training code may write there. The API has no artifact upload endpoint.
- Set `ETI_ARTIFACT_DIR` to a persistent private server directory. It must be backed up and writable only by the trusted ETI service account.
- `POST /v1/inference/{model_id}/{version}` is available only for approved model versions. The request must specify a supported asset class and exactly the feature names in the registered input schema; values and returned probabilities are checked for finite valid numbers. Requests are authenticated and rate-limited.
- Inference returns short/flat/long probabilities only. It does not size a position, approve a trade, bypass the risk engine or place an order.
- The repository has not automatically trained or promoted a production model. The training-to-artifact-to-registry workflow and model drift monitoring remain to be completed before using inference in paper autonomy.
