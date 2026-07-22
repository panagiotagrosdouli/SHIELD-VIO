## Summary

Describe the change and the problem it addresses.

## Scientific or engineering motivation

Explain why this change is needed and which research question, failure mode, or maintenance concern it supports.

## Validation level

Select every level supported by this PR:

- [ ] Static analysis only
- [ ] Unit validation
- [ ] Analytical or numerical-invariant validation
- [ ] Synthetic validation
- [ ] Public-dataset execution
- [ ] Simulation validation
- [ ] Hardware validation

## Reproduction

```bash
# Exact commands used
```

Record configuration files, random seeds, data sequences, runtime environment, and output paths.

## Results

Summarize the evidence. Include aggregate results when applicable and identify any generated figures or tables.

## Claim boundary

State explicitly what this PR does **not** establish. Distinguish synthetic, mocked, dataset, simulation, and hardware evidence.

## Checklist

- [ ] The change is focused and auditable.
- [ ] Tests cover new or changed behavior.
- [ ] Units, coordinate frames, assumptions, and thresholds are documented.
- [ ] Deployable diagnostics remain separate from privileged labels or oracle metadata.
- [ ] Experiments use recorded seeds and reproducible commands.
- [ ] Generated artifacts can be traced to their source data.
- [ ] Documentation reflects any change to the implementation or evidence boundary.
- [ ] `ruff check shield_vio scripts tests` passes.
- [ ] `black --check .` passes.
- [ ] `pytest -q` passes.
