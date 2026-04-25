from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChecklistItem(BaseModel):
    id: str
    requirement: str
    section: str

class Finding(BaseModel):
    id: str = Field(..., description="Unique ID for the finding")
    section: str = Field(..., description="Section where the issue was found")
    type: str = Field(..., description="Type: missing, inconsistent, insufficient, non_compliant")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    description: str = Field(..., description="Description of the issue")
    regulation: Optional[str] = Field(None, description="Related regulation, if applicable")
    suggested_fix: Optional[str] = Field(None, description="Suggested fix for the issue")

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    task_description: str
    model_card_metadata: Dict[str, Any]
    available_sections: List[str]
    current_section: Optional[str]
    checklist: List[ChecklistItem]
    findings_so_far: List[Finding]
    sections_reviewed: List[str]
    steps_remaining: int
    step_count: int

class ActionType(str, Enum):
    READ_SECTION = "read_section"
    FLAG_ISSUE = "flag_issue"
    SUGGEST_IMPROVEMENT = "suggest_improvement"
    VERIFY_CLAIM = "verify_claim"
    SUBMIT_AUDIT = "submit_audit"

class Action(BaseModel):
    """What the agent can do."""
    action_type: ActionType
    section_name: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    regulation: Optional[str] = None
    suggestion: Optional[str] = None
    claim_key: Optional[str] = None

class Reward(BaseModel):
    """Multi-dimensional reward signal."""
    total: float
    precision_score: float
    recall_score: float
    coverage_score: float
    efficiency_bonus: float
    false_positive_penalty: float
    progress_bonus: float = 0.0
    repetition_penalty: float = 0.0
