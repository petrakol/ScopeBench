# Python integration SDK

ScopeBench now includes a minimal integration entrypoint:

```python
from scopebench.integrations import guard

result = guard(plan=plan_dict, contract=contract_dict, shadow_mode=False)
print(result.decision, result.recommended_patch)
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
