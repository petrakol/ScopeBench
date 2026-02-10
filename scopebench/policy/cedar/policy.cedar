permit(
    principal,
    action,
    resource
)
when {
    resource.aggregate.spatial <= resource.contract.thresholds.max_spatial
    && resource.aggregate.temporal <= resource.contract.thresholds.max_temporal
    && resource.aggregate.depth <= resource.contract.thresholds.max_depth
};

forbid(
    principal,
    action,
    resource
)
when {
    resource.aggregate.uncertainty > resource.contract.escalation.ask_if_uncertainty_over
};
