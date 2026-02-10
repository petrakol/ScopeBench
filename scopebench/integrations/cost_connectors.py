from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from scopebench.plan import EstimateMetric, EstimateSource, PlanDAG, RealtimeEstimate


@dataclass
class ConnectorEstimate:
    step_id: str
    metric: EstimateMetric
    value: float
    source: EstimateSource
    confidence: float = 0.7
    metadata: Optional[Dict[str, str]] = None


class CostConnector:
    source: EstimateSource

    def estimate(self, plan: PlanDAG) -> List[ConnectorEstimate]:
        raise NotImplementedError


class AwsCostExplorerConnector(CostConnector):
    source = EstimateSource.AWS_COST_EXPLORER

    def estimate(self, plan: PlanDAG) -> List[ConnectorEstimate]:
        estimates: List[ConnectorEstimate] = []
        for step in plan.steps:
            if not step.tool:
                continue
            tool = step.tool.lower()
            if "deploy" in tool or "infra" in tool:
                estimates.append(
                    ConnectorEstimate(
                        step_id=step.id,
                        metric=EstimateMetric.COST_USD,
                        value=8.0,
                        source=self.source,
                        confidence=0.65,
                        metadata={"pricing_model": "m5.large-hourly"},
                    )
                )
                estimates.append(
                    ConnectorEstimate(
                        step_id=step.id,
                        metric=EstimateMetric.TIME_DAYS,
                        value=0.25,
                        source=self.source,
                        confidence=0.6,
                    )
                )
        return estimates


class GcpBillingConnector(CostConnector):
    source = EstimateSource.GCP_BILLING

    def estimate(self, plan: PlanDAG) -> List[ConnectorEstimate]:
        estimates: List[ConnectorEstimate] = []
        for step in plan.steps:
            tool = (step.tool or "").lower()
            if "bigquery" in tool or "gcp" in tool:
                estimates.append(
                    ConnectorEstimate(
                        step_id=step.id,
                        metric=EstimateMetric.COST_USD,
                        value=4.5,
                        source=self.source,
                        confidence=0.7,
                        metadata={"pricing_model": "on-demand-query"},
                    )
                )
        return estimates


class AzureCostManagementConnector(CostConnector):
    source = EstimateSource.AZURE_COST_MANAGEMENT

    def estimate(self, plan: PlanDAG) -> List[ConnectorEstimate]:
        estimates: List[ConnectorEstimate] = []
        for step in plan.steps:
            tool = (step.tool or "").lower()
            if "azure" in tool:
                estimates.append(
                    ConnectorEstimate(
                        step_id=step.id,
                        metric=EstimateMetric.COST_USD,
                        value=5.0,
                        source=self.source,
                        confidence=0.68,
                    )
                )
        return estimates


class StripeBillingConnector(CostConnector):
    source = EstimateSource.STRIPE

    def estimate(self, plan: PlanDAG) -> List[ConnectorEstimate]:
        estimates: List[ConnectorEstimate] = []
        for step in plan.steps:
            text = step.description.lower()
            if "invoice" in text or "billing" in text or "payment" in text:
                estimates.append(
                    ConnectorEstimate(
                        step_id=step.id,
                        metric=EstimateMetric.COST_USD,
                        value=1.25,
                        source=self.source,
                        confidence=0.75,
                        metadata={"pricing_model": "per-invoice"},
                    )
                )
                estimates.append(
                    ConnectorEstimate(
                        step_id=step.id,
                        metric=EstimateMetric.LABOR_HOURS,
                        value=0.4,
                        source=self.source,
                        confidence=0.6,
                    )
                )
        return estimates


def apply_realtime_estimates(plan: PlanDAG, connectors: List[CostConnector]) -> PlanDAG:
    step_lookup = {step.id: step for step in plan.steps}
    for connector in connectors:
        for estimate in connector.estimate(plan):
            step = step_lookup.get(estimate.step_id)
            if step is None:
                continue
            step.realtime_estimates.append(
                RealtimeEstimate(
                    metric=estimate.metric,
                    value=estimate.value,
                    source=estimate.source,
                    captured_at=datetime.now(timezone.utc),
                    confidence=estimate.confidence,
                    metadata=estimate.metadata or {},
                )
            )
    return plan


def default_cost_connectors() -> List[CostConnector]:
    return [
        AwsCostExplorerConnector(),
        GcpBillingConnector(),
        AzureCostManagementConnector(),
        StripeBillingConnector(),
    ]
