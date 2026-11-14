# ETI — Evolutionary Trading Intelligence

ETI is a research-first quantitative and AI trading intelligence project. The v0.2 architecture provides reproducible experiments, deterministic risk controls, transparent strategy comparisons, and a mobile-friendly research dashboard.

## Safety status

- Live spot execution is available but disabled by default. It requires owner-managed credentials, saved capital/risk limits and explicit one-time owner authorization; no per-trade approval is required.
- Live execution remains Binance/MEXC spot only and disabled by default. Research data now distinguishes crypto spot, perpetual swaps and dated futures where each exchange lists the requested contract. Derivative execution is not implemented.
- Public Yahoo Finance OHLCV research adapter is available for forex, equities, indices and commodities (Yahoo ticker notation such as EURUSD=X, AAPL, ^GSPC and GC=F). The owner-controlled autonomous loop discovers active crypto spot, perpetual-swap and dated-future listings across Binance/MEXC each day, then queues up to one top-ranked instrument per discovery. It also schedules forex, equity, commodity and index reference models, for up to ten candidate-training runs per daily cycle. This is public market-data access, not an authenticated broker feed; history and interval coverage vary. Validated CSV remains available as a fallback.
- Derivative and non-crypto candidate training is exploratory: historical funding and open-interest features are used when supplied, but liquidation data is not yet a model feature and side-aware funding settlement, financing, corporate actions, futures rolls and contract-specific margin/fee schedules remain incomplete. A quality gate prevents unsupported candidates from passing approval.
- Training does not run on application startup or in CI. It can be launched manually through the research capability/CLI or scheduled by the owner in the autonomous research settings; scheduled candidates remain subject to the same quality gates.
- No profitability claim: backtests do not establish future performance.
- Independent risk: model decisions must pass deterministic risk and portfolio checks; defaults disable shorts and leverage above 1x.
- Exchange credentials: owner-only Binance/MEXC keys are encrypted with AES-256-GCM using a server-side key; the API never returns plaintext credentials. Connection verification checks private read access only.

## Owner-managed exchange credential vault

- Only usernames configured in `ETI_ADMIN_USERNAMES` can add or remove exchange credentials.
- Initial adapters are Binance and MEXC spot. The connection test calls private `fetch_balance` only; it does not place orders and does not prove that trading permission is enabled.
- API keys and secrets are encrypted with AES-256-GCM before they reach SQLite. Each ciphertext is authenticated against the owner, venue and credential field. The API returns only a masked key suffix.
- Configure `ETI_CREDENTIAL_ENCRYPTION_KEY` as URL-safe base64 of 32 cryptographically random bytes. Generate it with `python -c 'import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())'`. Keep a secure offline copy: losing or rotating this key without re-encrypting the stored credentials makes them unrecoverable.
- Use exchange keys with withdrawals disabled and IP restrictions enabled. Do not paste exchange keys into chat.
- This vault supplies the owner-managed connection layer. Live execution remains off until separately configured and explicitly authorized by the owner.

## Autonomous live spot execution

- Live execution is off by default. The owner must connect a Binance/MEXC key, configure a USDT spot watchlist and capital allocation, then complete a one-time authorization with explicit confirmations for spot trading permission, disabled withdrawals and exchange IP restrictions.
- After authorization, the worker schedules one evaluation per completed candle. It selects the latest approved, calibrated model for each eligible market and executes model-driven entries/exits automatically; there is no per-trade approval prompt.
- The first release is spot-only and long/flat, with at most one ETI-managed position open across the watchlist: no shorts, margin, leverage or withdrawals. Hard risk ceilings are enforced server-side: max 0.5% equity risk per trade, 10% single-position size, 25% gross exposure, 2% daily loss stop and 8% drawdown stop. Owner settings may be lower, never higher.
- Every order starts with a durable client-order intent. Duplicate task retries are blocked; uncertain exchange responses trigger a fail-closed stop rather than resubmitting blindly. Open orders are reconciled before another signal can trade. The owner kill switch stops new scheduling; an order already accepted by an exchange may still need exchange-side cancellation/reconciliation.
- The initial watchlist is owner-bounded. ETI only trades symbols with an approved, calibrated model; symbols without one are skipped. Candidate model approval remains a separate model-governance step.
- Market orders can experience slippage and losses. This is an execution system, not a profitability guarantee. Start with a small allocation and verify exchange permissions and order history before increasing capital.

## Market coverage diagnostics

The authenticated agent capability market.coverage returns a machine-readable matrix of implemented research sources, supported market types, live execution boundaries and unsupported products. It reports implementation capability only; it does not probe venue uptime, guarantee historical candle depth, or certify a symbol as tradable. Use market.discover and market.snapshot to test a specific instrument and its actual data response.

## Architecture implemented

| Layer | Current state |
|---|---|
| Core | Modular configuration, provider interface and documented service boundaries |
| Market data | Public OHLCV for Binance/MEXC spot, swaps and dated futures where listed; Yahoo Finance public OHLCV for forex, equities, commodities and indices; validated CSV input |
| Market catalog | Canonical venue-independent instrument specifications and filtered instrument catalog; live public OHLCV currently supports crypto spot/swaps/dated futures where listed |
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

Training remains an explicit CLI action and is not started by API startup or CI. No live order placement endpoints exist. Public crypto OHLCV supports spot, swaps and dated futures on Binance/MEXC only where each venue lists the requested market. Yahoo Finance public OHLCV supports forex, equities, commodities and indices with ticker and interval coverage limits; authenticated broker feeds are not connected. Derivative research does not yet include complete funding-history, open-interest, liquidation, financing, roll or contract-cost datasets, and unsupported-cost candidates are blocked from approval.

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


## Self-hosted container deployment (private beta foundation)

The repository includes a Docker image and Compose stack for a single-host deployment. It runs the FastAPI/PWA service and the durable worker as separate containers sharing one persistent SQLite volume.

1. Install Docker Engine and the Docker Compose plugin on a Linux host.
2. Copy `.env.example` to `.env` on the host and replace both placeholder secrets with independently generated random values. Keep `.env` private and out of Git.
3. Keep `ETI_BIND_ADDRESS=127.0.0.1` when using a TLS reverse proxy on the same host. Do not expose the raw API port directly to the public internet.
4. Run `docker compose up --build -d` from the repository directory. The container honors the platform-provided `PORT` variable (default 8000) for managed Docker hosts.
5. Check `docker compose ps` and `docker compose logs -f api worker`. The API health check is `/health`; the worker should appear in the signed-in PWA after its first heartbeat.

The default binding is localhost intentionally. Configure a domain and HTTPS reverse proxy (for example, Caddy or Nginx) before remote access. Use a firewall, secure host updates, encrypted backups, and a recovery test. Invite-only registration requires sharing the registration key only with intended testers.

**Deployment limits:** this is a single-host private-beta deployment template, not a managed public service. The API and worker share SQLite on one host; do not scale them across machines or replicas. There is no managed PostgreSQL, email verification, monitoring/alerting, or public Android app yet. Encrypted object-store backups are optional and must be configured; they are not enabled by default. Do not add exchange credentials or enable live trading. Before a public launch, complete a security review, TLS/domain setup, backup/restore test, email verification and edge rate limiting.


## Single-instance production runtime and encrypted backups

- Railway's initial deployment uses one API service with a persistent volume mounted at `/app/data`. The durable worker runs inside that same service so API and worker share the same SQLite file; do not scale this service horizontally or split the worker onto another service while SQLite is in use.
- Set `ETI_EMBEDDED_WORKER=1` to enable the worker in the API lifespan. Local development and Compose retain their separate worker service by default.
- Set `ETI_BACKUP_ENABLED=1` to run an immediate and then daily encrypted SQLite snapshot to the configured S3-compatible object bucket. Snapshots use SQLite's online backup API, pass `PRAGMA integrity_check`, and are encrypted with AES-256-GCM before upload. Old objects are removed after the configured retention period (7–365 days).
- Required backup variables: `ETI_BACKUP_S3_ENDPOINT`, `ETI_BACKUP_S3_BUCKET`, `ETI_BACKUP_S3_ACCESS_KEY_ID`, `ETI_BACKUP_S3_SECRET_ACCESS_KEY`, and `ETI_BACKUP_ENCRYPTION_KEY` (URL-safe base64 encoding of 32 random bytes; generate with `python -c 'import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())'`). Optional: `ETI_BACKUP_S3_REGION`, `ETI_BACKUP_S3_ADDRESSING_STYLE` (`path` or `virtual`), `ETI_BACKUP_S3_PREFIX`, `ETI_BACKUP_RETENTION_DAYS`, and `ETI_BACKUP_INTERVAL_SECONDS`.
- Restore with `python -m eti.backup restore --key '<exact-object-key>' --destination /app/data/eti.sqlite3`. Restore decrypts into a temporary file, verifies SQLite integrity, then atomically replaces the destination. Stop API/worker writes before running a restore.
- Backups are encrypted independently of the storage provider. Keep the encryption key in Railway's sealed variables and in a separate secure offline recovery location; losing that key makes backups unrecoverable. Never put it in Git or chat.
- This is a single-instance deployment design, not high availability. PostgreSQL migration is required before multiple API/worker replicas or horizontal scaling.

## PWA account workspace

The PWA now includes sign-in, invite-only account creation, workspace selection, JSON memory editing, task queue and worker status, account export/deletion controls, and a model registry panel. Administrators can register candidate metadata, inspect versions, request approval and retire models; the API remains the authority for permissions and lifecycle checks. The bearer token is held in browser `sessionStorage` for the current tab session and is not written into the service-worker cache. Sign out clears the token from the current session.

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

The API exposes the authenticated capability list at `GET /v1/agents/capabilities` and worker heartbeat status at `GET /v1/agents/status`. The PWA shows workers as online only when a heartbeat was received within the last 60 seconds. The first built-in capability is `market.snapshot`: it fetches completed public crypto spot candles from Binance or MEXC and returns a descriptive snapshot with causal technical indicators. MEXC is the default public venue; a task may explicitly select Binance. The PWA populates its capability picker from the authenticated server allowlist. It does not train a model, recommend a trade, access private exchange accounts or place orders.

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
- The model approval endpoint verifies that the server-owned artifact exists, matches its registered checksum and deserializes as a fitted ETI model before allowing independent approval.
- Artifacts are never accepted as user uploads. Joblib/pickle deserialization can execute code, so **never place untrusted files in `ETI_ARTIFACT_DIR`**. Only trusted server-side training code may write there.
- Set `ETI_ARTIFACT_DIR` to a persistent private server directory writable only by the trusted ETI service account.
- `POST /v1/inference/{model_id}/{version}` serves approved versions only. Requests must specify a supported asset class and exactly the feature names in the registered schema; values and returned probabilities are validated. Requests are authenticated and rate-limited.
- Inference returns short/flat/long probabilities only. It does not size a position, approve a trade, bypass the risk engine or place an order.
- Model approval and inference also require calibration evidence embedded in the artifact: a passing temperature-scaling report, at least 30 calibration samples and a finite positive temperature. Registry metadata alone cannot make an uncalibrated artifact approvable. The current export CLI marks artifacts uncalibrated because the time-series calibration fitter is still pending.
- No production model is trained or promoted by this change. The training-to-artifact-to-registry workflow and model drift monitoring remain to be completed before paper autonomy.


## Account data portability and workspace deletion

- `GET /v1/account/export` returns a JSON export of the authenticated account's workspaces, memory, agent tasks, agent messages and audit history. It excludes password hashes, session signing secrets and other accounts' data.
- `DELETE /v1/workspaces/{workspace_id}` permanently deletes an owned workspace and its dependent memory, tasks and messages. Audit events are retained with the workspace reference cleared and a `workspace.deleted` event appended.
- The mobile PWA exposes a signed-in JSON export download and a confirmation-gated delete button for the selected workspace.
- These controls do not delete the account itself, shared model-registry records, or server backups. Account closure and backup-retention policy require a separately defined operational/legal process; do not promise complete erasure from backups until that process is configured.


## Explicit model artifact export from the training CLI

The normal research command does not export a deployable model. The opt-in `--save-model` flag runs only after the requested evaluation completes and writes a candidate to the private `ETI_ARTIFACT_DIR` content-addressed store. The candidate classifier is fit on the earlier 80% of completed labelled rows; the label horizon is purged between the fit and calibration windows to prevent future-price leakage. Temperature scaling is fit only on the later chronological 20% calibration segment. The metrics report includes the artifact URI, SHA-256, feature schema, training/calibration windows and calibration evidence.

Example command shape (do not run until the project is ready for the planned training phase):

```bash
ETI_ARTIFACT_DIR=/srv/eti-private-artifacts \
python -m eti.pipeline --exchange mexc --symbol BTC/USDT \
  --timeframe 1h --evaluation nested --save-model \
  --model-id eti-crypto-core --model-version 0.1.0
```

If the calibration window has fewer than 30 rows, fewer than five examples of any class, or temperature scaling does not improve calibration-window log loss, the artifact remains `uncalibrated` and cannot be approved. A passing calibration report is not evidence of profitability and must be reviewed alongside out-of-sample trading metrics. The export does not register or approve the model; an independent evaluation report and reviewer approval are still required before inference. The export flag is explicit; API startup, CI and worker startup never train or export models.


## Global market discovery and derivatives research

- Authenticated dashboard discovery can list active Binance/MEXC spot, perpetual-swap and dated-future instruments, ranking by public quote volume when the venue supports bulk tickers.
- Derivatives research can request current funding/open-interest values and historical funding/open-interest observations where CCXT and the selected venue expose those endpoints. Unsupported endpoints are reported explicitly; missing values are never synthesized.
- When historical funding and open-interest data are available, ETI aligns them to completed OHLCV candles with backward-only as-of joins and trains dedicated funding-rate and open-interest-change features. No future observation is joined to an earlier candle.
- Nested derivative evaluation applies a conservative absolute funding drag estimated from observed funding intervals, plus explicit fee and slippage assumptions. This approximation is not side-aware exchange settlement.
- Derivative candidates remain blocked from approval because liquidation, contract-specific fee schedules, funding-side settlement, margin tiers, mark/index-price risk and venue execution controls are not yet independently validated. Live execution remains spot-only.
