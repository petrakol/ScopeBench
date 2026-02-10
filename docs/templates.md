# Domain template library

ScopeBench ships domain presets under `scopebench/templates/` for:

- `swe`
- `ops`
- `finance`
- `health`
- `marketing`

Each domain contains a default bundle and can optionally include **named variants**.

Core files:

- `contract.yaml`
- `plan.yaml`
- `notes.md`

Variant files:

- `<variant>.contract.yaml`
- `<variant>.plan.yaml`
- `<variant>.notes.md`

Included variants cover real-world verticals such as payments (`finance/payments`),
medical data (`health/medical_data`), marketing outreach (`marketing/outreach`),
database operations (`ops/database`), and release stabilization (`swe/release_fix`).

Use the CLI:

```bash
scopebench template list
scopebench template show finance/contract
scopebench template show finance/payments/plan
scopebench template generate health/medical_data
```

Name format:

- `<domain>` (defaults to `bundle` + `default` variant)
- `<domain>/<kind>` where `kind` is `bundle|contract|plan|notes`
- `<domain>/<variant>` (defaults to `bundle`)
- `<domain>/<variant>/<kind>`

`generate` emits valid YAML (contract+plan bundle by default), suitable for piping to tooling.
