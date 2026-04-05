import pytest
from env.models import Finding
from env.reward import compute_matches, compute_reward
from env.graders import grade_easy_task, grade_medium_task, grade_hard_task, grade_task

def _make_finding(section, ftype, severity="medium", suggested_fix=None, regulation=None):
    return Finding(
        id="test",
        section=section,
        type=ftype,
        severity=severity,
        description="test",
        suggested_fix=suggested_fix,
        regulation=regulation
    )

# --- compute_matches ---

def test_compute_matches_exact():
    findings = [_make_finding("bias_analysis", "missing")]
    gt = [{"section": "bias_analysis", "type": "missing", "severity": "high"}]
    assert compute_matches(findings, gt) == 1

def test_compute_matches_no_match():
    findings = [_make_finding("bias_analysis", "missing")]
    gt = [{"section": "limitations", "type": "missing", "severity": "high"}]
    assert compute_matches(findings, gt) == 0

def test_compute_matches_empty():
    assert compute_matches([], []) == 0

# --- Easy grader ---

def test_easy_grader_perfect():
    findings = [
        _make_finding("limitations", "missing"),
        _make_finding("bias_analysis", "missing")
    ]
    gt = [
        {"section": "limitations", "type": "missing", "severity": "high"},
        {"section": "bias_analysis", "type": "missing", "severity": "high"}
    ]
    # perfect recall (1.0), perfect precision (1.0), full coverage
    score = grade_easy_task(findings, gt, ["a", "b", "c", "d"], 4)
    assert score > 0.9

def test_easy_grader_zero():
    score = grade_easy_task([], [{"section": "x", "type": "missing", "severity": "high"}], [], 4)
    assert score == 0.0

# --- Medium grader ---

def test_medium_grader_with_suggestions():
    findings = [
        _make_finding("evaluation_metrics", "inconsistent", "high", suggested_fix="Fix it")
    ]
    gt = [{"section": "evaluation_metrics", "type": "inconsistent", "severity": "high"}]
    score = grade_medium_task(findings, gt)
    assert score > 0.8

# --- Hard grader ---

def test_hard_grader_with_regulation():
    findings = [
        _make_finding("general", "missing", "critical", regulation="EU AI Act Article 9")
    ]
    gt = [{"section": "general", "type": "missing", "severity": "critical", "regulation": "EU AI Act Article 9"}]
    score = grade_hard_task(findings, gt, 10, 60)
    assert score > 0.7

# --- grade_task dispatcher ---

def test_grade_task_unknown():
    assert grade_task("unknown", [], [], [], 0, 0, 30) == 0.0

def test_grade_task_routes_correctly():
    findings = [_make_finding("limitations", "missing")]
    gt = [{"section": "limitations", "type": "missing", "severity": "high"}]
    score = grade_task("basic_completeness", findings, gt, ["a"], 4, 5, 30)
    assert 0.0 <= score <= 1.0
