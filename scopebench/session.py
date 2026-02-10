from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG


class SessionAgent(BaseModel):
    agent_id: str = Field(..., min_length=1)
    contract: Optional[TaskContract] = Field(
        default=None,
        description="Optional per-agent contract override. Falls back to global_contract.",
    )


class AgentPlan(BaseModel):
    agent_id: str = Field(..., min_length=1)
    plan: PlanDAG


class MultiAgentSession(BaseModel):
    global_contract: TaskContract
    agents: List[SessionAgent] = Field(..., min_length=1)
    plans: List[AgentPlan] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_bindings(self) -> "MultiAgentSession":
        agent_ids = [agent.agent_id for agent in self.agents]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("agents[].agent_id values must be unique")

        known_agents = set(agent_ids)
        plan_count: Dict[str, int] = {agent_id: 0 for agent_id in agent_ids}
        for plan in self.plans:
            if plan.agent_id not in known_agents:
                raise ValueError(
                    f"plan for agent_id '{plan.agent_id}' does not match any session agent"
                )
            plan_count[plan.agent_id] += 1

        missing = [agent_id for agent_id, count in plan_count.items() if count == 0]
        if missing:
            raise ValueError(
                f"each agent must have at least one plan; missing: {', '.join(sorted(missing))}"
            )
        return self

    def contract_for(self, agent_id: str) -> TaskContract:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent.contract or self.global_contract
        raise KeyError(f"Unknown agent_id: {agent_id}")

    def plans_for(self, agent_id: str) -> List[PlanDAG]:
        return [entry.plan for entry in self.plans if entry.agent_id == agent_id]

    def to_agent_plan_map(self) -> Dict[str, List[PlanDAG]]:
        return {agent.agent_id: self.plans_for(agent.agent_id) for agent in self.agents}
