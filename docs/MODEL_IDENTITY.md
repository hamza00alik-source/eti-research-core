# Canonical model identity

ETI model IDs are instrument identities, not display labels. Every new model ID must encode:

- venue
- asset class
- market type
- normalized symbol
- timeframe

Use `eti.model_identity.model_id_for_instrument` for every new model registration and lookup. The canonical form is:

```text
eti-{venue}-{asset_class}-{market_type}-{symbol}-{timeframe}
```

Each component is lowercased and normalized to hyphen-separated alphanumeric text. Empty or unusable components are rejected. This prevents a spot model, swap model, equity model or another instrument class from accidentally sharing a registry identity.

## Legacy compatibility

Live execution currently supports crypto spot only. Its approved-model resolver first searches the canonical crypto-spot ID and then, only if no canonical model exists, checks the exact pre-canonical ID:

```text
eti-{venue}-{symbol}-{timeframe}
```

This fallback exists only to avoid orphaning previously approved crypto-spot models. New registrations must never write legacy IDs. New asset classes and market types must not use the legacy resolver. Legacy registrations should be reviewed and migrated to canonical IDs before that compatibility path is removed.

## Approval boundary

Model identity and model approval are separate concerns. Autonomous workers may register immutable candidate versions, but cannot approve them. Calibration, schema matching and a training-time quality report do not replace locked out-of-sample economic evidence or human promotion review. See [MODEL_PROMOTION.md](MODEL_PROMOTION.md).
