package scopebench

import future.keywords.if
import future.keywords.in

default decision := {
  "decision": "ALLOW",
  "reasons": ["Within contract envelope"],
  "exceeded": {},
  "asked": {}
}

contraction := max([0, 1 - input.aggregate.uncertainty])
ask_any := input.contract.escalation.ask_if_any_axis_over * contraction

max_threshold(axis) := t if {
  axis == "uncertainty"
  t := input.contract.thresholds.max_uncertainty
}

max_threshold(axis) := t if {
  axis != "uncertainty"
  t := input.contract.thresholds[sprintf("max_%s", [axis])] * contraction
}

axis_exceeded[axis] if {
  some axis in object.keys(input.aggregate)
  axis != "n_steps"
  input.aggregate[axis] > max_threshold(axis)
}

allowed_tools_violation if {
  input.facts.allowed_tools_active
  some vector in input.vectors
  vector.tool != null
  not vector.tool in input.contract.allowed_tools
}

forbidden_category_violation if {
  some vector in input.vectors
  vector.tool_category != null
  vector.tool_category in input.contract.forbidden_tool_categories
}

exceeded[axis] := {
  "value": input.aggregate[axis],
  "threshold": max_threshold(axis),
} if {
  some axis in axis_exceeded
}

exceeded["allowed_tools"] := {"value": 1, "threshold": 0} if {
  allowed_tools_violation
}

exceeded["forbidden_tool_categories"] := {"value": 1, "threshold": 0} if {
  forbidden_category_violation
}

denied_reasons[r] if {
  some axis in axis_exceeded
  r := sprintf("%s=%0.2f exceeds max %0.2f", [axis, input.aggregate[axis], max_threshold(axis)])
}

denied_reasons[r] if {
  input.facts.allowed_tools_active
  some vector in input.vectors
  vector.tool != null
  not vector.tool in input.contract.allowed_tools
  r := sprintf("Tool '%s' not in contract.allowed_tools", [vector.tool])
}

denied_reasons[r] if {
  some vector in input.vectors
  vector.tool_category != null
  vector.tool_category in input.contract.forbidden_tool_categories
  r := sprintf("Tool category '%s' is forbidden by contract", [vector.tool_category])
}

ask_fields[axis] := val if {
  some axis in object.keys(input.aggregate)
  axis != "n_steps"
  val := input.aggregate[axis]
  val > ask_any
}

ask_fields["uncertainty"] := input.aggregate.uncertainty if {
  input.aggregate.uncertainty > input.contract.escalation.ask_if_uncertainty_over
}

ask_fields["tool_category"] := 1 if {
  count(input.facts.present_high_risk_categories) > 0
}

ask_fields["read_before_write"] := 1 if {
  input.facts.missing_initial_read
}

ask_fields["validation_after_write"] := 1 if {
  input.facts.missing_validation_after_write
}

ask_reasons[r] if {
  some category in input.facts.present_high_risk_categories
  r := sprintf("Tool category '%s' triggers ASK by escalation rules", [category])
}

ask_reasons[r] if {
  input.facts.missing_initial_read
  r := "SWE write step appears before any read-only step; ask for initial inspection"
}

ask_reasons[r] if {
  input.facts.missing_validation_after_write
  r := "SWE write step is missing a downstream validation/test step"
}

decision := {
  "decision": "DENY",
  "reasons": [r | r := denied_reasons[_]],
  "exceeded": exceeded,
  "asked": {},
} if {
  count(exceeded) > 0
}

decision := {
  "decision": "ASK",
  "reasons": reasons,
  "exceeded": {},
  "asked": ask_fields,
} if {
  count(exceeded) == 0
  count(ask_fields) > 0
  reasons := [r | r := ask_reasons[_]]
  count(reasons) > 0
}

decision := {
  "decision": "ASK",
  "reasons": ["Within max thresholds but triggers escalation/uncertainty thresholds"],
  "exceeded": {},
  "asked": ask_fields,
} if {
  count(exceeded) == 0
  count(ask_fields) > 0
  count([r | r := ask_reasons[_]]) == 0
}
