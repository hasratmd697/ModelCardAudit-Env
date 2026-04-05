import pytest
from env.models import (
    ChecklistItem, Finding, Observation, Action, ActionType, Reward
)

def test_checklist_item_creation():
    item = ChecklistItem(id="c1", section="model_description", requirement="Must have model description")
    assert item.id == "c1"
    assert item.section == "model_description"

def test_finding_creation():
    finding = Finding(
        id="F1",
        section="bias_analysis",
        type="missing",
        severity="high",
        description="Bias analysis missing"
    )
    assert finding.id == "F1"
    assert finding.regulation is None
    assert finding.suggested_fix is None

def test_finding_with_regulation():
    finding = Finding(
        id="F2",
        section="training_data",
        type="non_compliant",
        severity="critical",
        description="No data provenance",
        regulation="EU AI Act Article 10",
        suggested_fix="Document data provenance."
    )
    assert finding.regulation == "EU AI Act Article 10"
    assert finding.suggested_fix is not None

def test_action_types():
    assert ActionType.READ_SECTION.value == "read_section"
    assert ActionType.FLAG_ISSUE.value == "flag_issue"
    assert ActionType.SUGGEST_IMPROVEMENT.value == "suggest_improvement"
    assert ActionType.VERIFY_CLAIM.value == "verify_claim"
    assert ActionType.SUBMIT_AUDIT.value == "submit_audit"

def test_action_creation():
    action = Action(action_type=ActionType.READ_SECTION, section_name="bias_analysis")
    assert action.action_type == ActionType.READ_SECTION
    assert action.section_name == "bias_analysis"
    assert action.issue_type is None

def test_observation_creation():
    obs = Observation(
        task_id="basic_completeness",
        task_description="Test task",
        model_card_metadata={"model_name": "TestModel"},
        available_sections=["model_description", "limitations"],
        current_section=None,
        checklist=[],
        findings_so_far=[],
        sections_reviewed=[],
        steps_remaining=30,
        step_count=0
    )
    assert obs.task_id == "basic_completeness"
    assert obs.steps_remaining == 30
    assert len(obs.available_sections) == 2

def test_reward_creation():
    reward = Reward(
        total=0.75,
        precision_score=0.8,
        recall_score=0.7,
        coverage_score=1.0,
        efficiency_bonus=0.5,
        false_positive_penalty=-0.05
    )
    assert reward.total == 0.75
    assert reward.false_positive_penalty == -0.05
