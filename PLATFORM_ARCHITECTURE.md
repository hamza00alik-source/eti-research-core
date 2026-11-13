# ETI shared platform architecture

## Product contract

The Android application and web/PWA are two clients of one ETI platform. They do not own separate models, agents, databases, watchlists or trading state.

```text
Android app ─┐
             ├── HTTPS + authenticated API ── ETI backend
Web / PWA ───┘                                  ├── Identity and access control
                                                ├── Shared memory / workspace
                                                ├── Agent orchestrator + message ledger
                                                ├── Model registry + inference service
                                                ├── Market/provider adapter registry
                                                ├── Risk and portfolio policy engine
                                                └── Audit/event store
```

The backend is the source of truth. Both clients display and update the same account-scoped state. A device may cache non-sensitive read-only data for offline display, but it must reconcile through the backend and must never become an alternate source of truth.

## Shared memory

Memory is account/workspace scoped and typed. It is not one unrestricted prompt transcript.

- **Profile/configuration:** user preferences, risk limits, enabled markets and app settings.
- **Research memory:** watchlists, hypotheses, notes, experiments and reviewed findings.
- **Portfolio context:** account references, positions and exposure snapshots from authorized providers.
- **Agent working memory:** task state, handoffs, intermediate artifacts and bounded context.
- **Decision/audit history:** model version, input snapshot reference, agent actions, risk result and timestamp.

Every record needs an owner/workspace, type, schema version, created/updated timestamps and provenance. Sensitive credentials are not memory records. Reads and writes are permission checked and audited. Agents receive least-privilege scoped context, not unrestricted access to every user's data. Retention, export and deletion policies must be explicit.

## Agent-to-agent communication

Agents do not call each other through hidden direct coupling. They communicate through a durable orchestrator/message ledger:

1. An agent creates a typed task with a unique ID, requester, target capability, deadline and allowed context.
2. The orchestrator checks permissions and routes the task.
3. The receiving agent posts a typed response or progress event linked to the task.
4. The requester consumes the response; failures, retries and timeouts are recorded.
5. All messages are attributable to an agent/version and are auditable.

Messages are data, not executable instructions. Tool permissions and risk policy are enforced outside the model. Agents cannot grant themselves permissions, change immutable risk limits, approve their own model promotion or trigger live orders.

## Shared model lifecycle

A central model registry stores immutable model versions and metadata: model ID, version, artifact checksum/location, input schema, supported asset/instrument classes, training-data lineage, evaluation report, calibration status and lifecycle state.

The inference service resolves a model by capability and approved version. Android and web clients call the same inference API. They do not download private model artifacts or independently choose unapproved model versions. Device inference is an optional later optimization for explicitly exportable models; server inference remains the common baseline.

A model can be proposed by training, but promotion requires independent evaluation, calibration/drift checks and explicit approval. Agents cannot silently replace production models.

## Market and broker plug-in contract

Market integrations belong in the backend, not separately in Android and web.

Each provider adapter must declare:
- venue/provider and supported asset classes;
- instrument types (spot, margin, perpetual swap, dated future, option or CFD);
- public versus authenticated data capabilities;
- supported symbols and normalized instrument metadata;
- market-data endpoints, rate limits and freshness;
- fee, funding, margin, contract multiplier, tick/lot and trading-calendar semantics where applicable;
- whether execution is unsupported, paper-only or separately approved.

Adapters translate provider-specific APIs into stable canonical ETI contracts. The client APIs expose normalized instruments and capabilities, not provider-specific payloads. Adding a supported venue should therefore be a backend adapter/configuration task; clients should not need a new release unless a new user-facing feature is introduced.

A new asset class or materially different contract may require a specialist model and independent validation. Adding an API connection does not make the model automatically competent in that market.

## Security boundary

- HTTPS/TLS is mandatory outside local development.
- Both clients authenticate users and send short-lived access tokens; refresh tokens are protected by platform secure storage.
- Every endpoint enforces account/workspace authorization and least privilege.
- Provider credentials are submitted only to backend credential-management endpoints, encrypted at rest using managed keys, masked in responses and never logged.
- Prefer read-only keys initially. Withdrawal permissions are never requested. Trading permissions remain disabled until a separately reviewed execution milestone.
- Rate limits, request validation, replay protection where applicable, audit logs and secret rotation are required.
- CORS is an origin policy, not authentication.
- Public GitHub repository contents must never include credentials, tokens, private datasets or production configuration.

## Stable client API families

The future versioned API should provide stable, client-neutral resources:

- `/v1/session` and `/v1/profile`
- `/v1/memory` and `/v1/workspaces`
- `/v1/agents/tasks` and `/v1/agents/messages`
- `/v1/models` and `/v1/inference`
- `/v1/markets`, `/v1/instruments` and `/v1/providers`
- `/v1/portfolio`, `/v1/risk` and `/v1/audit`

These are target contracts, not a claim that these endpoints are already implemented. The existing unversioned research API remains research-only until authentication, persistence, authorization and migration are implemented and tested.

## Delivery sequence

1. Shared persistence and account/workspace isolation.
2. Authentication, authorization, secret management and audit foundations.
3. Durable agent task/message orchestration and scoped shared memory.
4. Model registry, versioned inference contract and approval lifecycle.
5. Versioned client API consumed by both PWA and native Android.
6. Native Android client and cloud-built APK.
7. Backend market adapters added incrementally, followed by instrument-specific validation.
8. Paper-only autonomous orchestration; live execution is a separate reviewed milestone.

## Current implementation boundary

The current repository has a research FastAPI app, a mobile-first PWA shell, provider and instrument adapter extension points, and deterministic risk gates. It does not yet have production authentication, persistent shared memory, an agent message service, a model registry service, encrypted broker credential storage, a deployed backend or a native Android app. Those must be implemented and verified before describing ETI as a connected multi-client platform.
