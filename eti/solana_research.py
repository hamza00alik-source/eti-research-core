"""Read-only Solana/Pump.fun research adapters. No wallet signing or order execution.

The module deliberately treats incomplete evidence as UNKNOWN. It is not a
"safe token" oracle and is not wired to ETI's live execution capability.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator, Iterable

import httpx


SOLANA_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
SOLANA_TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
DEX_BASE = "https://api.dexscreener.com"
RUGCHECK_BASE = "https://api.rugcheck.xyz"


class ProviderError(RuntimeError):
    """A provider failed, returned malformed data, or exceeded the response contract."""


@dataclass(frozen=True)
class Evidence:
    name: str
    status: str  # PASS | FAIL | UNKNOWN
    value: Any
    source: str
    reason: str | None = None


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


class SolanaRpcClient:
    """Small JSON-RPC client; configure a dedicated endpoint for sustained workloads."""

    def __init__(self, url: str | None = None, timeout: float = 15.0):
        self.url = url or os.getenv("ETI_SOLANA_RPC_URL", "https://api.mainnet.solana.com")
        self.timeout = timeout
        self._next_id = int(time.time() * 1000)

    def call(self, method: str, params: list[Any]) -> Any:
        self._next_id += 1
        try:
            response = httpx.post(
                self.url,
                json={"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Solana RPC {method} request failed: {type(exc).__name__}") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            raise ProviderError(f"Solana RPC {method} returned an error")
        if "result" not in payload:
            raise ProviderError(f"Solana RPC {method} response omitted result")
        return payload["result"]

    def account_info(self, address: str) -> dict[str, Any] | None:
        result = self.call("getAccountInfo", [address, {"encoding": "jsonParsed", "commitment": "confirmed"}])
        return result.get("value") if isinstance(result, dict) else None

    def token_supply(self, mint: str) -> dict[str, Any] | None:
        result = self.call("getTokenSupply", [mint, {"commitment": "confirmed"}])
        return result.get("value") if isinstance(result, dict) else None

    def token_program_accounts(self, mint: str) -> list[dict[str, Any]]:
        """Fetch mint-owned token accounts from both SPL Token programs."""
        accounts: list[dict[str, Any]] = []
        for program in (SOLANA_TOKEN_PROGRAM, SOLANA_TOKEN_2022_PROGRAM):
            result = self.call("getProgramAccounts", [
                program,
                {
                    "encoding": "jsonParsed",
                    "commitment": "confirmed",
                    "filters": [{"memcmp": {"offset": 0, "bytes": mint}}],
                },
            ])
            if not isinstance(result, list):
                raise ProviderError("getProgramAccounts returned a non-list result")
            accounts.extend(result)
        return accounts


class DexScreenerClient:
    """Public DexScreener token-pair lookup, respecting its 30-mint batch limit."""

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    def fetch_pairs(self, mints: Iterable[str]) -> list[dict[str, Any]]:
        values = list(dict.fromkeys(mints))
        if not values or any(not isinstance(mint, str) or not mint for mint in values):
            raise ValueError("at least one non-empty mint is required")
        pairs: list[dict[str, Any]] = []
        for offset in range(0, len(values), 30):
            batch = values[offset:offset + 30]
            url = f"{DEX_BASE}/tokens/v1/solana/{','.join(batch)}"
            try:
                response = httpx.get(url, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise ProviderError(f"DexScreener token lookup failed: {type(exc).__name__}") from exc
            if not isinstance(payload, list):
                raise ProviderError("DexScreener token lookup returned an unexpected shape")
            pairs.extend(item for item in payload if isinstance(item, dict))
        return pairs

    @staticmethod
    def select_pair(mint: str, pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = [
            pair for pair in pairs
            if pair.get("chainId") == "solana"
            and isinstance(pair.get("baseToken"), dict)
            and pair["baseToken"].get("address") == mint
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: _number((pair.get("liquidity") or {}).get("usd")) or 0.0)


class RugCheckClient:
    """Optional public risk-intelligence source; response fields are not trusted blindly."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def _get(self, path: str) -> dict[str, Any]:
        try:
            response = httpx.get(f"{RUGCHECK_BASE}{path}", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"RugCheck request failed: {type(exc).__name__}") from exc
        if not isinstance(payload, dict):
            raise ProviderError("RugCheck returned an unexpected response shape")
        return payload

    def report(self, mint: str) -> dict[str, Any]:
        return self._get(f"/v1/tokens/{mint}/report")

    def insider_graph(self, mint: str) -> dict[str, Any]:
        return self._get(f"/v1/tokens/{mint}/insiders/graph")


class PumpPortalStream:
    """One reconnecting WebSocket stream for new-token and migration events.

    Requires the optional runtime dependency `websockets`. API keys are read
    server-side from ETI_PUMPPORTAL_API_KEY and never returned in event payloads.
    """

    def __init__(self, api_key: str | None = None, max_backoff: float = 60.0):
        self.api_key = api_key if api_key is not None else os.getenv("ETI_PUMPPORTAL_API_KEY")
        self.max_backoff = max_backoff

    async def events(self, stop: asyncio.Event | None = None) -> AsyncIterator[dict[str, Any]]:
        try:
            import websockets
        except ImportError as exc:
            raise ProviderError("Install the websockets package to use PumpPortalStream") from exc
        endpoint = "wss://pumpportal.fun/api/data"
        if self.api_key:
            endpoint += "?api-key=" + self.api_key
        delay = 1.0
        while stop is None or not stop.is_set():
            try:
                async with websockets.connect(endpoint, ping_interval=20, ping_timeout=20, open_timeout=15) as ws:
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    await ws.send(json.dumps({"method": "subscribeMigration"}))
                    delay = 1.0
                    async for raw in ws:
                        if stop is not None and stop.is_set():
                            return
                        try:
                            event = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(event, dict):
                            event["_eti_received_at"] = time.time()
                            yield event
            except asyncio.CancelledError:
                raise
            except Exception:
                if stop is not None and stop.is_set():
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_backoff)


def scan_pair(
    mint: str,
    pair: dict[str, Any] | None,
    *,
    min_liquidity_usd: float = 40_000.0,
    min_pool_age_minutes: float = 10.0,
    max_h1_change_pct: float = 80.0,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Apply transparent market filters; no heuristic is described as a safety check."""
    if not isinstance(mint, str) or not mint:
        raise ValueError("mint is required")
    if pair is None:
        return {"mint": mint, "verdict": "WAIT", "reason": "no DexScreener pair yet", "evidence": {}}
    liquidity = _number((pair.get("liquidity") or {}).get("usd"))
    h1_change = _number((pair.get("priceChange") or {}).get("h1"))
    created = _number(pair.get("pairCreatedAt"))
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    age = (now - created) / 60_000 if created is not None else None
    checks = {
        "liquidity_usd": liquidity,
        "h1_change_pct": h1_change,
        "pool_age_minutes": age,
        "pair_address": pair.get("pairAddress"),
        "price_usd": _number(pair.get("priceUsd")),
        "volume_h24_usd": _number((pair.get("volume") or {}).get("h24")),
        "txns_h1_buys": _number(((pair.get("txns") or {}).get("h1") or {}).get("buys")),
        "txns_h1_sells": _number(((pair.get("txns") or {}).get("h1") or {}).get("sells")),
        "dex_id": pair.get("dexId"),
        "quote_token_address": ((pair.get("quoteToken") or {}).get("address")
                                if isinstance(pair.get("quoteToken"), dict) else None),
    }
    failures = []
    unknown = []
    if liquidity is None:
        unknown.append("liquidity unavailable")
    elif liquidity < min_liquidity_usd:
        failures.append("liquidity below configured floor")
    if h1_change is None:
        unknown.append("1h price change unavailable")
    elif h1_change > max_h1_change_pct:
        failures.append("1h price change exceeds chase ceiling")
    if age is None:
        unknown.append("pool creation time unavailable")
    elif age < min_pool_age_minutes:
        failures.append("pool younger than configured minimum")
    verdict = "REJECT" if failures else "UNKNOWN" if unknown else "PASS"
    return {
        "mint": mint, "verdict": verdict,
        "reason": "; ".join(failures + unknown) or None,
        "evidence": checks, "pair_snapshot_sha256": _hash_json(pair),
        "source": "dexscreener", "observed_at_ms": now,
        "note": "Scanner filters are market heuristics, not token-safety findings or a profitability signal.",
    }


def _parsed_info(account: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(account, dict):
        return None
    data = account.get("data")
    parsed = data.get("parsed") if isinstance(data, dict) else None
    info = parsed.get("info") if isinstance(parsed, dict) else None
    return info if isinstance(info, dict) else None


def audit_mint(
    mint: str,
    *,
    rpc: SolanaRpcClient | None = None,
    rugcheck: RugCheckClient | None = None,
    pool_created_at_ms: int | None = None,
    pool_address: str | None = None,
    creator_address: str | None = None,
    excluded_token_accounts: Iterable[str] = (),
    now_ms: int | None = None,
    max_top10_pct: float = 25.0,
    min_pool_age_minutes: float = 10.0,
) -> dict[str, Any]:
    """Run available read-only checks and return explicit PASS/FAIL/UNKNOWN layers.

    Creator sell-history and pool-specific LP-lock semantics remain UNKNOWN until
    an indexed history/LP source with a verified schema is configured.
    """
    rpc = rpc or SolanaRpcClient()
    rugcheck = rugcheck or RugCheckClient()
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    layers: list[Evidence] = []

    try:
        mint_account = rpc.account_info(mint)
        info = _parsed_info(mint_account)
        if info is None:
            layers.append(Evidence("mint_authority", "UNKNOWN", None, "solana_rpc", "mint account missing or not jsonParsed"))
            layers.append(Evidence("freeze_authority", "UNKNOWN", None, "solana_rpc", "mint account missing or not jsonParsed"))
        else:
            mint_authority = info.get("mintAuthority", "MISSING")
            freeze_authority = info.get("freezeAuthority", "MISSING")
            layers.append(Evidence("mint_authority", "PASS" if mint_authority is None else "FAIL" if mint_authority != "MISSING" else "UNKNOWN",
                                   mint_authority if mint_authority != "MISSING" else None, "solana_rpc"))
            layers.append(Evidence("freeze_authority", "PASS" if freeze_authority is None else "FAIL" if freeze_authority != "MISSING" else "UNKNOWN",
                                   freeze_authority if freeze_authority != "MISSING" else None, "solana_rpc"))
    except ProviderError as exc:
        layers.extend([
            Evidence("mint_authority", "UNKNOWN", None, "solana_rpc", str(exc)),
            Evidence("freeze_authority", "UNKNOWN", None, "solana_rpc", str(exc)),
        ])

    try:
        supply = rpc.token_supply(mint)
        supply_amount = int(supply["amount"]) if isinstance(supply, dict) and supply.get("amount") is not None else 0
        if supply_amount <= 0:
            raise ProviderError("token supply is zero or unavailable")
        accounts = rpc.token_program_accounts(mint)
        excluded = set(excluded_token_accounts)
        owner_balances: dict[str, int] = {}
        delegated_owners: dict[str, set[str]] = {}
        for account in accounts:
            address = account.get("pubkey")
            parsed = _parsed_info(account.get("account"))
            if not parsed or address in excluded:
                continue
            owner = parsed.get("owner")
            token_amount = parsed.get("tokenAmount") or {}
            raw_amount = token_amount.get("amount")
            try:
                amount = int(raw_amount)
            except (TypeError, ValueError):
                continue
            if not isinstance(owner, str) or not owner or amount <= 0:
                continue
            owner_balances[owner] = owner_balances.get(owner, 0) + amount
            delegate = parsed.get("delegate")
            delegated_amount = (parsed.get("delegatedAmount") or {}).get("amount")
            try:
                active_delegate = isinstance(delegate, str) and int(delegated_amount or 0) > 0
            except (TypeError, ValueError):
                active_delegate = False
            if active_delegate:
                delegated_owners.setdefault(delegate, set()).add(owner)
        if not owner_balances:
            raise ProviderError("no parsed token holders returned")
        top10_raw = sum(sorted(owner_balances.values(), reverse=True)[:10])
        top10_pct = top10_raw / supply_amount * 100.0
        layers.append(Evidence("top10_holder_concentration", "PASS" if top10_pct <= max_top10_pct else "FAIL",
                               round(top10_pct, 4), "solana_rpc",
                               "Aggregated by token-account owner; excluded accounts are only those explicitly supplied."))
        layers.append(Evidence("delegated_holder_accounts", "UNKNOWN", {
            "delegated_owner_counts": {key: len(value) for key, value in delegated_owners.items()},
            "holder_count": len(owner_balances),
        }, "solana_rpc", "Delegate presence is reported for review; delegate authority is not automatically classified as malicious."))
    except (ProviderError, KeyError, TypeError, ValueError) as exc:
        layers.append(Evidence("top10_holder_concentration", "UNKNOWN", None, "solana_rpc", str(exc)))
        layers.append(Evidence("delegated_holder_accounts", "UNKNOWN", None, "solana_rpc", str(exc)))

    if pool_created_at_ms is None:
        layers.append(Evidence("pool_age", "UNKNOWN", None, "dexscreener", "pool creation timestamp unavailable"))
    else:
        age = max(0.0, (now - pool_created_at_ms) / 60_000)
        layers.append(Evidence("pool_age", "PASS" if age >= min_pool_age_minutes else "FAIL",
                               round(age, 2), "dexscreener"))

    try:
        report = rugcheck.report(mint)
        report_hash = _hash_json(report)
        layers.append(Evidence("lp_lock", "UNKNOWN", {"report_sha256": report_hash}, "rugcheck",
                               "Pool-specific LP lock/burn schema and lock contract were not independently verified."))
    except ProviderError as exc:
        layers.append(Evidence("lp_lock", "UNKNOWN", None, "rugcheck", str(exc)))

    try:
        graph = rugcheck.insider_graph(mint)
        nodes = graph.get("nodes")
        links = graph.get("links")
        valid_graph = isinstance(nodes, list) and isinstance(links, list)
        layers.append(Evidence("insider_graph", "UNKNOWN", {
            "nodes": len(nodes) if isinstance(nodes, list) else None,
            "links": len(links) if isinstance(links, list) else None,
            "response_sha256": _hash_json(graph),
        } if valid_graph else None, "rugcheck",
            "Graph relationships are review evidence, not proof of common beneficial ownership."))
    except ProviderError as exc:
        layers.append(Evidence("insider_graph", "UNKNOWN", None, "rugcheck", str(exc)))

    layers.append(Evidence("creator_holdings_and_sales", "UNKNOWN", creator_address, "not_configured",
                           "Current creator identity and complete post-migration sell history are not proven by this RPC snapshot."))
    statuses = [item.status for item in layers]
    return {
        "kind": "solana_token_audit",
        "mint": mint,
        "pool_address": pool_address,
        "observed_at_ms": now,
        "passed": bool(statuses) and all(status == "PASS" for status in statuses),
        "blocked": any(status != "PASS" for status in statuses),
        "status_counts": {status: statuses.count(status) for status in ("PASS", "FAIL", "UNKNOWN")},
        "layers": [asdict(item) for item in layers],
        "warning": "An incomplete audit is UNKNOWN, never safe. A completed checklist is not a safety guarantee.",
    }
