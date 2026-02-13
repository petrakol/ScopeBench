forbid(
  principal,
  action == ScopeBench::Action::"Evaluate",
  resource
)
when {
  resource.aggregate.spatial > resource.thresholds.max_spatial * resource.facts.contraction / resource.axis_weights.spatial
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.temporal > resource.thresholds.max_temporal * resource.facts.contraction / resource.axis_weights.temporal
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.depth > resource.thresholds.max_depth * resource.facts.contraction / resource.axis_weights.depth
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.irreversibility > resource.thresholds.max_irreversibility * resource.facts.contraction / resource.axis_weights.irreversibility
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.resource_intensity > resource.thresholds.max_resource_intensity * resource.facts.contraction / resource.axis_weights.resource_intensity
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.legal_exposure > resource.thresholds.max_legal_exposure * resource.facts.contraction / resource.axis_weights.legal_exposure
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.dependency_creation > resource.thresholds.max_dependency_creation * resource.facts.contraction / resource.axis_weights.dependency_creation
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.stakeholder_radius > resource.thresholds.max_stakeholder_radius * resource.facts.contraction / resource.axis_weights.stakeholder_radius
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.power_concentration > resource.thresholds.max_power_concentration * resource.facts.contraction / resource.axis_weights.power_concentration
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.aggregate.uncertainty > resource.thresholds.max_uncertainty / resource.axis_weights.uncertainty
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.facts.allowed_tools_violation
};

forbid(principal, action == ScopeBench::Action::"Evaluate", resource)
when {
  resource.facts.forbidden_tool_category_violation
};

permit(principal, action == ScopeBench::Action::"Evaluate", resource);
