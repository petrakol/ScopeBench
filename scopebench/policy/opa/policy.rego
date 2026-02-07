# OPA policy template (illustrative).
# This repo's MVP uses a Python policy engine; this file is a starting point
# if you want to shift enforcement to Open Policy Agent (OPA).

package scopebench

default decision := "ALLOW"

# input.aggregate has axis values in [0,1]
# input.contract.thresholds has max_*

deny_reasons[r] {
  input.aggregate.spatial > input.contract.thresholds.max_spatial
  r := sprintf("spatial %v > max_spatial %v", [input.aggregate.spatial, input.contract.thresholds.max_spatial])
}

deny_reasons[r] {
  input.aggregate.depth > input.contract.thresholds.max_depth
  r := sprintf("depth %v > max_depth %v", [input.aggregate.depth, input.contract.thresholds.max_depth])
}

decision := "DENY" {
  count(deny_reasons) > 0
}

ask := true {
  input.aggregate.uncertainty > input.contract.escalation.ask_if_uncertainty_over
}

decision := "ASK" {
  decision == "ALLOW"
  ask
}
