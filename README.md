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
- Signup and login use persistent SQLite request windows with hashed subject identifiers; the API returns HTTP 429 and `Retry-After` when a limit is reached. The transport peer address is used rather than trusting client-supplied forwarding headers.
- Example server environment:
  ```bash
  export ETI_AUTH_SECRET='replace-with-a-long-random-secret-of-at-least-32-bytes'
  export ETI_REGISTRATION_KEY='set-a-private-invite-key-of-at-least-16-chars'
  export ETI_DATABASE_PATH='data/eti.sqlite3'
  uvicorn eti.api:app --host 0.0.0.0 --port 8000
  ```
- The account identity in protected routes comes from a verified bearer token, never a client-supplied owner id.
- This is a foundation, not public-production identity management. Before public signup/deployment, add email verification, login/register rate limiting at the edge, HTTPS, secure secret management, backups, monitoring, and PostgreSQL for multi-instance use.
- Workspace memory, agent task/message and model registry HTTP routes are intentionally not exposed until their authorization policies are implemented and tested.

## Existing research workflow

Training remains an explicit CLI action and is not started by API startup or CI. No live order placement endpoints exist. Current live public market discovery is crypto spot on Binance/MEXC only; other labelled asset classes are available only through validated local CSV OHLCV data.

The model registry currently stores immutable version metadata and checksums, not model binaries. Agent tasks/messages are durable ledgers, not an autonomous running agent network. The PWA remains an undeployed frontend shell until a backend and HTTPS host are configured.


## Authenticated workspace and agent APIs (new)

All routes in this section require `Authorization: Bearer <access_token>`. The account ID is derived from the verified token; clients cannot submit an owner ID.

- `GET/POST /v1/workspaces`: list or create private workspaces.
- `GET /v1/workspaces/{workspace_id}/memory`: list memory items (optional `namespace` filter).
- `GET/PUT/DELETE /v1/workspaces/{workspace_id}/memory/{namespace}/{key}`: read, write or delete a JSON memory item.
- `GET/POST /v1/workspaces/{workspace_id}/agents/tasks`: list or queue durable task records.
- `GET/PATCH /v1/workspaces/{workspace_id}/agents/tasks/{task_id}`: read a task or apply a guarded status transition.
- `GET/POST /v1/workspaces/{workspace_id}/agents/messages`: list or append typed JSON messages.

Every workspace route checks account ownership and returns not-found for another account's workspace. Agent tasks/messages remain a durable ledger only: creating a task does not run an agent, and message payloads are stored as data, never executed as code or instructions. Model registry APIs, worker dispatch, permissioned agent capabilities and production deployment remain future work.


## PWA account workspace

The PWA now includes sign-in, invite-only account creation, workspace selection, JSON memory editing and task queue visibility. The bearer token is held in browser `sessionStorage` for the current tab session and is not written into the service-worker cache. Sign out clears the token from the current session.

The service worker caches only the static app shell. It bypasses all `/v1/` endpoints, and the API sends `Cache-Control: no-store` for authenticated and dynamic routes. Security headers include a restrictive same-origin Content Security Policy, frame protection and MIME sniffing protection.

The PWA still requires a deployed HTTPS backend and configured server secrets before sign-in can work from a public URL. A native Android application is not yet implemented.


## Authenticated model registry API

- Signed-in accounts can list approved model versions and retrieve an approved version's metadata.
- Mutating the shared model registry requires the account username to be explicitly listed in the server-only `ETI_ADMIN_USERNAMES` environment variable (comma-separated).
- Admins can register candidate metadata, request independent approval and retire approved versions. The creator cannot approve their own candidate; the existing storage policy still requires a passing evaluation report and calibrated status.
- The API derives `created_by` and `approved_by` from the authenticated account. Those identities cannot be supplied in the request body.
- Candidate and retired versions are hidden from non-admin accounts.
- The registry remains metadata-only. It does not host/download model binaries, run inference, train models or auto-promote candidates.
