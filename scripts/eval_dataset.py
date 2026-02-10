from __future__ import annotations


from scopebench.bench.dataset import default_cases_path, load_cases
from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate


def main():
    cases_path = default_cases_path()
    cases = load_cases(cases_path)
    correct = 0
    for c in cases:
        contract = TaskContract.model_validate(c.contract)
        plan = PlanDAG.model_validate(c.plan)
        res = evaluate(contract, plan)
        pred = res.policy.decision.value
        ok = pred == c.expected_decision
        correct += int(ok)
        print(f"{c.id:24s} expected={c.expected_decision:5s} got={pred:5s} ok={ok}")
    print(f"Accuracy: {correct}/{len(cases)} = {correct / len(cases):.2%}")


if __name__ == "__main__":
    main()
