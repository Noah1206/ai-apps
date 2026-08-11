# Benchmark identity decision

No new benchmark version is needed: the two raw manifests validate to the same Pydantic
`BenchmarkManifest`. The stored file contains 10 quote-combined label objects; validation
expands them into the same 15 labels written by the export. Audio identities, transcripts,
annotations, and all other validated fields are equal.

The auditable decision is therefore:

```text
benchmark_id: busan-surface-v0
benchmark_version: 1.0.0
canonical package: artifacts/task-002/busan-surface-v0--1.0.0.zip
package SHA-256: 151c1e28804627bea69bbd7f6632f4d3558ebf076147e42c1d168d508467233c
canonical raw manifest SHA-256: 7dd7ad855da2fa5f3a9611b46315c4655f932e37a3706973070c526a6c804bdc
semantic SHA-256: 700d352edb4a4e9321b48ec6cd312bec6ad1d4c48fa2bedbcf80a2ca23a67f8c
```

Raw-byte hashes remain provenance identifiers; the semantic hash is the identity used to
prove equivalence. A future semantic change must use a new benchmark version. Neither
historical file was overwritten.

`task-005-canonical-reevaluation.json` was produced from the original two Prediction JSONL
files. It did not run the models or alter predictions.
