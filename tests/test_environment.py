import pytest
from env.environment import ModelCardAuditEnv
from env.models import Action, ActionType

@pytest.fixture
def env():
    return ModelCardAuditEnv()

def test_reset_returns_observation(env):
    obs = env.reset("basic_completeness")
    assert obs.task_id == "basic_completeness"
    assert obs.step_count == 0
    assert obs.steps_remaining == 30
    assert len(obs.available_sections) > 0
    assert len(obs.checklist) > 0

def test_read_section_action(env):
    obs = env.reset("basic_completeness")
    section = obs.available_sections[0]
    action = Action(action_type=ActionType.READ_SECTION, section_name=section)
    obs2, reward, done, info = env.step(action)
    assert obs2.step_count == 1
    assert obs2.current_section is not None
    assert section in obs2.sections_reviewed
    assert done is False

def test_read_invalid_section(env):
    env.reset("basic_completeness")
    action = Action(action_type=ActionType.READ_SECTION, section_name="nonexistent_section")
    obs, reward, done, info = env.step(action)
    assert "Error" in obs.current_section

def test_flag_issue_action(env):
    env.reset("basic_completeness")
    action = Action(
        action_type=ActionType.FLAG_ISSUE,
        section_name="limitations",
        issue_type="missing",
        severity="high",
        description="limitations section is missing"
    )
    obs, reward, done, info = env.step(action)
    assert len(obs.findings_so_far) == 1
    assert obs.findings_so_far[0].section == "limitations"

def test_submit_audit_ends_episode(env):
    env.reset("basic_completeness")
    action = Action(action_type=ActionType.SUBMIT_AUDIT)
    obs, reward, done, info = env.step(action)
    assert done is True
    assert "score" in info

def test_max_steps_ends_episode(env):
    env.reset("basic_completeness")
    env.max_steps = 2
    
    action = Action(action_type=ActionType.READ_SECTION, section_name=env.model_card["sections"].keys().__iter__().__next__())
    env.step(action)
    obs, reward, done, info = env.step(action)
    assert done is True

def test_step_after_done(env):
    env.reset("basic_completeness")
    env.step(Action(action_type=ActionType.SUBMIT_AUDIT))
    obs, reward, done, info = env.step(Action(action_type=ActionType.SUBMIT_AUDIT))
    assert done is True

def test_state_returns_dict(env):
    env.reset("basic_completeness")
    state = env.state()
    assert "task_id" in state
    assert "step_count" in state
    assert "done" in state
    assert "findings" in state
    assert "action_history" in state

def test_verify_claim_action(env):
    env.reset("basic_completeness")
    action = Action(action_type=ActionType.VERIFY_CLAIM, claim_key="accuracy")
    obs, reward, done, info = env.step(action)
    assert "verification_result" in info

def test_suggest_improvement_action(env):
    env.reset("basic_completeness")
    action = Action(
        action_type=ActionType.SUGGEST_IMPROVEMENT,
        section_name="model_description",
        suggestion="Add more detail about architecture."
    )
    obs, reward, done, info = env.step(action)
    assert "message" in info

def test_medium_task_reset(env):
    obs = env.reset("technical_consistency")
    assert obs.task_id == "technical_consistency"
    assert obs.steps_remaining == 45

def test_hard_task_reset(env):
    obs = env.reset("regulatory_compliance")
    assert obs.task_id == "regulatory_compliance"
    assert obs.steps_remaining == 60

def test_reward_bounded(env):
    env.reset("basic_completeness")
    action = Action(action_type=ActionType.SUBMIT_AUDIT)
    obs, reward, done, info = env.step(action)
    assert 0.0 <= reward.total <= 1.0
