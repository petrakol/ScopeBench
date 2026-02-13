from __future__ import annotations

from scopebench.server.api import (
    DatasetSuggestRequest,
    DatasetValidateRequest,
    DatasetRenderRequest,
    DatasetWizardRequest,
    DatasetReviewCommentRequest,
    DatasetReviewSuggestEditRequest,
    DatasetReviewVoteRequest,
    DatasetReviewSubmitRequest,
    create_app,
)


def _endpoint(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"Missing endpoint: {path}")


def test_dataset_validate_and_suggest_endpoints():
    app = create_app()
    suggest_endpoint = _endpoint(app, "/dataset/suggest")
    validate_endpoint = _endpoint(app, "/dataset/validate")
    render_endpoint = _endpoint(app, "/dataset/render")
    wizard_endpoint = _endpoint(app, "/dataset/wizard")
    review_comment_endpoint = _endpoint(app, "/dataset/review/comment")
    review_edit_endpoint = _endpoint(app, "/dataset/review/suggest_edit")
    review_vote_endpoint = _endpoint(app, "/dataset/review/vote")
    review_state_endpoint = _endpoint(app, "/dataset/review/{case_id}")
    review_submit_endpoint = _endpoint(app, "/dataset/review/submit")

    contract = {"goal": "Fix flaky CI test", "preset": "team"}
    plan = {
        "task": "Fix flaky CI test",
        "steps": [
            {"id": "read", "description": "Inspect failing test", "tool": "git_read"},
            {"id": "patch", "description": "Patch timing", "tool": "git_patch", "depends_on": ["read"]},
        ],
    }

    suggest_req = DatasetSuggestRequest(
        id="community_case_api_001",
        domain="engineering",
        instruction="Fix flaky CI",
        contract=contract,
        plan=plan,
        expected_decision="ALLOW",
        expected_rationale="Bounded fix with validation.",
    )
    suggested = suggest_endpoint(suggest_req)
    case = suggested.case

    assert case["id"] == "community_case_api_001"
    assert len(case["expected_step_vectors"]) == 2

    validate_req = DatasetValidateRequest(case=case)
    validated = validate_endpoint(validate_req)
    assert validated.ok is True
    assert validated.case_id == "community_case_api_001"

    rendered_json = render_endpoint(DatasetRenderRequest(case=case, format="json"))
    assert rendered_json.filename == "community_case_api_001.json"
    assert '"id": "community_case_api_001"' in rendered_json.content

    rendered_yaml = render_endpoint(DatasetRenderRequest(case=case, format="yaml"))
    assert rendered_yaml.filename == "community_case_api_001.yaml"
    assert "id: community_case_api_001" in rendered_yaml.content

    wizarded = wizard_endpoint(DatasetWizardRequest(
        id="community_case_api_001",
        domain="engineering",
        instruction="Fix flaky CI",
        contract=contract,
        plan=plan,
        expected_decision="ALLOW",
        expected_rationale="Bounded fix with validation.",
        format="yaml",
    ))
    assert wizarded.ok is True
    assert wizarded.case_id == "community_case_api_001"
    assert wizarded.rendered.filename == "community_case_api_001.yaml"
    assert "id: community_case_api_001" in wizarded.rendered.content


    review_comment = review_comment_endpoint(DatasetReviewCommentRequest(
        case_id="community_case_api_001",
        reviewer="alice",
        comment="Please tighten rationale language.",
        draft_case=case,
    ))
    assert review_comment.case_id == "community_case_api_001"
    assert review_comment.comments[-1]["reviewer"] == "alice"

    review_edit = review_edit_endpoint(DatasetReviewSuggestEditRequest(
        case_id="community_case_api_001",
        reviewer="bob",
        field_path="expected_rationale",
        proposed_value="Scoped fix with explicit validation and rollback plan.",
        rationale="Improve reviewability before merge.",
        draft_case=case,
    ))
    assert review_edit.suggested_edits[-1]["field_path"] == "expected_rationale"


    try:
        review_submit_endpoint(DatasetReviewSubmitRequest(case_id="community_case_api_001", format="json"))
        raise AssertionError("Expected submission to fail before accepted vote threshold")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 409

    review_vote_one = review_vote_endpoint(DatasetReviewVoteRequest(
        case_id="community_case_api_001",
        reviewer="alice",
        vote="accept",
        draft_case=case,
    ))
    assert review_vote_one.acceptance["ready"] is False

    review_vote_two = review_vote_endpoint(DatasetReviewVoteRequest(
        case_id="community_case_api_001",
        reviewer="bob",
        vote="accept",
        draft_case=case,
    ))
    assert review_vote_two.acceptance["ready"] is True
    assert review_vote_two.acceptance["status"] == "accepted"

    review_state = review_state_endpoint("community_case_api_001")
    assert review_state.votes["alice"] == "accept"
    assert review_state.votes["bob"] == "accept"

    review_submit = review_submit_endpoint(DatasetReviewSubmitRequest(
        case_id="community_case_api_001",
        format="yaml",
    ))
    assert review_submit.ok is True
    assert review_submit.review_status == "accepted"
    assert review_submit.rendered.filename == "community_case_api_001.yaml"
