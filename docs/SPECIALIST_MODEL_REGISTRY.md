# Specialist model registry metadata

The model registry continues to use immutable candidate/approved/retired
versions. Specialist metadata is stored additively in the existing
`input_schema` JSON, so existing feature lists and old database files remain
compatible without a schema migration.

Optional metadata:

- `department`: one of the eight ETI departments.
- `model_role`: a role declared for that department.
- `experiment_id`: stable identifier for the code/config/data experiment.
- `lineage`: SHA-256 hashes for code, config and dataset, plus data cutoff and
  optional source/instrument metadata.

The registration API accepts these fields directly or through the input schema.
Conflicts, unknown departments/roles and malformed hashes fail closed. Legacy
models without these declarations remain unassigned; no model ID guessing is
used.

## Current exported baseline

The existing deployable classifier is tagged as:

- Department: `prediction`
- Role: `direction_classifier`
- Supported class: `crypto_spot`
- Version: base version plus the first 12 characters of the artifact SHA-256
- Experiment ID: deterministic hash of source-code, training configuration and
  dataset hashes

A repeat run trains from scratch. If it produces identical artifact bytes, it
reuses the same content-addressed model version; it is not misrepresented as a
newly improved model. This does not implement incremental learning.

## Governance

New registrations remain candidates. Existing independent calibration and
human approval rules still apply. Department metadata does not confer approval,
promotion, budget or execution authority.
