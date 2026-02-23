# Integration Starters (Adoption)

These copy-paste starters are intended to get ScopeBench running in an existing orchestration stack quickly.

## Airflow starter

```python
from scopebench.integrations import from_airflow_dag

plan = from_airflow_dag(dag, task="daily ingestion")
```

## Prefect starter

```python
from scopebench.integrations import from_prefect_tasks

plan = from_prefect_tasks(flow_tasks, task="daily ingestion")
```

## Dagster starter

```python
from scopebench.integrations import from_dagster_ops

plan = from_dagster_ops(ops, task="daily ingestion")
```

## Quick validation loop

```bash
scopebench quickstart --json --out-dir .scopebench-artifacts
```
