# Domain template library

ScopeBench now ships domain presets under `scopebench/templates/` for:

- `swe`
- `ops`
- `finance`
- `health`
- `marketing`

Each domain contains:

- `contract.yaml`
- `plan.yaml`
- `notes.md`

Use the CLI:

```bash
scopebench template list
scopebench template show finance/contract
scopebench template generate health
```

`generate` emits valid YAML (contract+plan bundle by default), suitable for piping to tooling.
