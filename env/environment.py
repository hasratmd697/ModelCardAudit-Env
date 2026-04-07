import os
import json
import random
import uuid
from typing import Dict, Any, Tuple
from .models import Observation, Action, ActionType, Finding, ChecklistItem, Reward
from .reward import compute_reward
from .graders import grade_task

# Paths relative to the project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class ModelCardAuditEnv:
    def __init__(self):
        self.task_id = None
        self.task_description = ""
        self.model_card = {}
        self.ground_truth = []
        self.checklist = []
        
        self.findings = []
        self.sections_reviewed = []
        self.action_history = []
        
        self.step_count = 0
        self.max_steps = 30
        self.done = False

    def load_data(self, task_id: str):
        if task_id == "basic_completeness":
            difficulty = "easy"
            self.max_steps = 30
            self.task_description = "Easy: Identify missing required sections from a basic model card checklist."
            cl_file = "basic_completeness.json"
            gt_file = "easy_annotations.json"
        elif task_id == "technical_consistency":
            difficulty = "medium"
            self.max_steps = 45
            self.task_description = "Medium: Review a model card for internal inconsistencies and insufficient documentation."
            cl_file = "technical_consistency.json"
            gt_file = "medium_annotations.json"
        elif task_id == "regulatory_compliance":
            difficulty = "hard"
            self.max_steps = 60
            self.task_description = "Hard: Full audit against EU AI Act documentation requirements and NIST AI RMF standards."
            cl_file = "regulatory_compliance.json"
            gt_file = "hard_annotations.json"
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        # Load Checklist
        with open(os.path.join(DATA_DIR, "checklists", cl_file), "r", encoding="utf-8") as f:
            cl_data = json.load(f)
            self.checklist = [ChecklistItem(**item) for item in cl_data.get("items", [])]

        # Load Ground Truth
        with open(os.path.join(DATA_DIR, "ground_truth", gt_file), "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        # Select a random model card
        mc_dir = os.path.join(DATA_DIR, "model_cards", difficulty)
        mc_files = [f for f in os.listdir(mc_dir) if f.endswith(".json")]
        if not mc_files:
            raise RuntimeError(f"No model cards found in {mc_dir}")
            
        chosen_file = random.choice(mc_files)
        with open(os.path.join(mc_dir, chosen_file), "r", encoding="utf-8") as f:
            self.model_card = json.load(f)
            
        # Filter ground truth to this specific card's issues
        card_id = self.model_card.get("id", chosen_file.replace(".json", ""))
        self.ground_truth = gt_data.get(card_id, [])

    def reset(self, task_id: str = "basic_completeness") -> Observation:
        self.task_id = task_id
        self.step_count = 0
        self.done = False
        self.findings = []
        self.sections_reviewed = []
        self.action_history = []
        
        self.load_data(task_id)
        
        return self._get_observation()

    def _get_observation(self, current_section_content: str = None) -> Observation:
        metadata = {
            "model_name": self.model_card.get("model_name"),
            "model_type": self.model_card.get("model_type"),
            "framework": self.model_card.get("framework")
        }
        
        sections = self.model_card.get("sections", {})
        available_sections = list(sections.keys())
        
        return Observation(
            task_id=self.task_id,
            task_description=self.task_description,
            model_card_metadata=metadata,
            available_sections=available_sections,
            current_section=current_section_content,
            checklist=self.checklist,
            findings_so_far=self.findings,
            sections_reviewed=self.sections_reviewed,
            steps_remaining=self.max_steps - self.step_count,
            step_count=self.step_count
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            return self._get_observation(), self._compute_current_reward(), True, {"score": self._compute_final_score()}

        self.step_count += 1
        self.action_history.append(action.action_type.value)
        
        current_section = None
        info = {}

        if action.action_type == ActionType.READ_SECTION:
            if action.section_name in self.model_card.get("sections", {}):
                current_section = self.model_card["sections"][action.section_name]
                if action.section_name not in self.sections_reviewed:
                    self.sections_reviewed.append(action.section_name)
            else:
                current_section = f"Error: Section '{action.section_name}' not found."
                
        elif action.action_type == ActionType.FLAG_ISSUE:
            finding = Finding(
                id=str(uuid.uuid4())[:8],
                section=action.section_name or "general",
                type=action.issue_type or "unknown",
                severity=action.severity or "medium",
                description=action.description or "",
                regulation=action.regulation,
                suggested_fix=action.suggestion
            )
            self.findings.append(finding)
            info["message"] = "Issue flagged successfully."

        elif action.action_type == ActionType.SUGGEST_IMPROVEMENT:
            info["message"] = "Improvement suggestion recorded for external usage."
            # In a more advanced env, might attach this to a specific finding

        elif action.action_type == ActionType.VERIFY_CLAIM:
            # Simple mock verification
            info["verification_result"] = f"Claim '{action.claim_key}' could not be automatically verified."

        elif action.action_type == ActionType.SUBMIT_AUDIT:
            self.done = True
            info["message"] = "Audit submitted."

        if self.step_count >= self.max_steps:
            self.done = True
            info["message"] = "Max steps reached."

        reward = self._compute_current_reward()
        obs = self._get_observation(current_section)
        
        if self.done:
            info["score"] = self._compute_final_score()

        return obs, reward, self.done, info

    def _compute_current_reward(self) -> Reward:
        total_sections = len(self.model_card.get("sections", {}))
        return compute_reward(
            findings=self.findings,
            ground_truth=self.ground_truth,
            steps_taken=self.step_count,
            max_steps=self.max_steps,
            sections_reviewed=self.sections_reviewed,
            total_sections=total_sections,
            action_history=self.action_history
        )
        
    def _compute_final_score(self) -> float:
        total_sections = len(self.model_card.get("sections", {}))
        return grade_task(
            task_id=self.task_id,
            findings=self.findings,
            ground_truth=self.ground_truth,
            sections_reviewed=self.sections_reviewed,
            total_sections=total_sections,
            steps_taken=self.step_count,
            max_steps=self.max_steps
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "done": self.done,
            "findings": [f.model_dump() for f in self.findings],
            "sections_reviewed": self.sections_reviewed,
            "action_history": self.action_history,
            "ground_truth_count": len(self.ground_truth)
        }
