# Policy Backends (Python, OPA, Cedar)

ScopeBench supports pluggable policy backends selected by:

- `SCOPEBENCH_POLICY_BACKEND=python|opa|cedar`
- CLI flag `--policy-backend`
- API request field `policy_backend`

When `opa` or `cedar` is selected, the external engine decision is authoritative. ScopeBench sends
normalized policy input to the engine and uses the returned `decision/reasons/exceeded/asked`
verbatim as the policy result.

`SCOPEBENCH_POLICY_FAIL_OPEN=1` switches external backends to fail-open (`ASK`) behavior if the
engine is unreachable. Default is fail-closed (`DENY`).

## OPA quickstart

```bash
opa run --server --addr :8181 scopebench/policy/opa/policy.rego
scopebench serve --host 0.0.0.0 --port 8080 --policy-backend opa
curl -s http://localhost:8080/evaluate \
  -H 'content-type: application/json' \
  -d '{"contract":{"goal":"Fix failing unit test","preset":"team"},"plan":{"task":"Fix failing unit test","steps":[{"id":"1","description":"Read test","tool":"git_read"},{"id":"2","description":"Apply patch","tool":"git_patch","depends_on":["1"]}]},"include_telemetry":true}'
```

## Cedar policy assets

- `scopebench/policy/cedar/policy.cedar`
- `scopebench/policy/cedar/schema.json`

The Cedar backend loads versioned policy files at runtime and reports `policy_hash` in API responses.
Default Cedar endpoint: `SCOPEBENCH_CEDAR_URL=http://localhost:8182/v1/decision`.
