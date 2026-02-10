package scopebench

default decision := {
  "decision": "ALLOW",
  "reasons": ["Within contract envelope"],
  "exceeded": {},
  "asked": {}
}

deny_axes[axis] {
  axis := "spatial"
  input.aggregate.spatial > input.contract.thresholds.max_spatial
}

deny_axes[axis] {
  axis := "temporal"
  input.aggregate.temporal > input.contract.thresholds.max_temporal
}

deny_axes[axis] {
  axis := "depth"
  input.aggregate.depth > input.contract.thresholds.max_depth
}

exceeded[axis] := {
  "value": input.aggregate[axis],
  "threshold": input.contract.thresholds[sprintf("max_%s", [axis])]
} {
  axis := deny_axes[_]
}

deny_reasons[r] {
  axis := deny_axes[_]
  r := sprintf("%s=%0.2f exceeds max %0.2f", [
    axis,
    input.aggregate[axis],
    input.contract.thresholds[sprintf("max_%s", [axis])],
  ])
}

ask_uncertainty {
  input.aggregate.uncertainty > input.contract.escalation.ask_if_uncertainty_over
}

ask_fields := {"uncertainty": input.aggregate.uncertainty} {
  ask_uncertainty
}

ask_reasons := ["Within max thresholds but triggers escalation/uncertainty thresholds"] {
  ask_uncertainty
}

decision := {
  "decision": "DENY",
  "reasons": deny_reasons,
  "exceeded": exceeded,
  "asked": {}
} {
  count(deny_axes) > 0
}

decision := {
  "decision": "ASK",
  "reasons": ask_reasons,
  "exceeded": {},
  "asked": ask_fields
} {
  count(deny_axes) == 0
  ask_uncertainty
}
