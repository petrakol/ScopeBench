# Python integration SDK

ScopeBench includes a lightweight integration SDK for agent runtimes.

## Guard call

```python
from scopebench.integrations import guard

result = guard(plan=plan_dict, contract=contract_dict, shadow_mode=False)
print(result.decision, result.recommended_patch)
```

`guard(...)` returns:

- `decision`, `effective_decision`
- `reasons`, `aggregate`, `exceeded`, `asked`
- `recommended_patch`

## Framework adapters

Use adapters to normalize common framework payloads into ScopeBench `PlanDAG`-compatible dicts.

```python
from scopebench.integrations import from_langchain_plan, from_autogen_messages, guard

langchain_plan = {
    "goal": "Fix flaky test",
    "steps": [
        {"action": "Read failing tests", "tool": "git_read"},
        {"action": "Patch retry logic", "tool": "git_patch", "depends_on": ["1"]},
    ],
}

plan = from_langchain_plan(langchain_plan)
result = guard(plan=plan, contract={"goal": "Fix flaky test", "preset": "team"})
print(result.effective_decision)

messages = [
    {"content": "Inspect logs", "tool": "analysis"},
    {"content": "Apply patch and run tests", "tool": "pytest"},
]
plan_from_messages = from_autogen_messages(messages, task="incident mitigation")
```

## Mock agent loop

```python
from scopebench.integrations import guard

contract = {"goal": "Fix failing unit test", "preset": "team"}
plan = {
    "task": "Fix failing unit test",
    "steps": [
        {"id": "1", "description": "Apply patch directly", "tool": "git_patch"},
    ],
}

guard_result = guard(plan=plan, contract=contract)
if guard_result.effective_decision == "ALLOW":
    print("execute tools")
else:
    print("replan first", guard_result.recommended_patch)
```
