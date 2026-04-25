import gymnasium as gym
from gymnasium import spaces
import json
from .environment import ModelCardAuditEnv
from .models import Action, ActionType

class ModelCardAuditGymEnv(gym.Env):
    """
    A Gymnasium-compatible wrapper for the ModelCardAuditEnv.
    Observation space: Text
    Action space: Text
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.env = ModelCardAuditEnv()
        self.observation_space = spaces.Text(max_length=16384)
        self.action_space = spaces.Text(max_length=4096)
        self.render_mode = render_mode

    def _format_observation(self, obs) -> str:
        """Format the Observation object into a string prompt."""
        task_id = obs.task_id
        parts = [
            f"## Task: {obs.task_description}",
            f"## Model: {obs.model_card_metadata.get('model_name', 'Unknown')} ({obs.model_card_metadata.get('model_type', 'Unknown')}, {obs.model_card_metadata.get('framework', 'Unknown')})",
            f"\n**Step**: {obs.step_count} | **Steps Remaining**: {obs.steps_remaining}",
            f"\n**Available Sections**: {', '.join(obs.available_sections)}",
            f"**Sections Already Reviewed**: {', '.join(obs.sections_reviewed) if obs.sections_reviewed else 'None'}",
        ]
        
        unreviewed = [s for s in obs.available_sections if s not in obs.sections_reviewed]
        parts.append(f"**Sections NOT Yet Reviewed**: {', '.join(unreviewed) if unreviewed else 'All reviewed'}")
        
        if obs.current_section:
            parts.append(f"\n**Current Section Content**:\n```\n{obs.current_section}\n```")
        
        if obs.checklist:
            if task_id == 'regulatory_compliance':
                # Group checklist by section
                from collections import defaultdict
                by_section = defaultdict(list)
                for item in obs.checklist:
                    by_section[item.section].append(item)
                checklist_lines = []
                for section, items in by_section.items():
                    checklist_lines.append(f"  [{section.upper()}]")
                    for item in items:
                        checklist_lines.append(f"    - [{item.id}] {item.requirement}")
                parts.append(f"\n**Regulatory Checklist**:\n" + "\n".join(checklist_lines))
            else:
                checklist_text = "\n".join([f"  - [{item.id}] {item.requirement} (section: {item.section})" for item in obs.checklist])
                parts.append(f"\n**Checklist Requirements**:\n{checklist_text}")
        
        if obs.findings_so_far:
            findings_text = "\n".join([
                f"  - [{f.id}] {f.section}: {f.type} ({f.severity}) — {f.description}"
                for f in obs.findings_so_far
            ])
            parts.append(f"\n**Issues Flagged So Far** ({len(obs.findings_so_far)} total):\n{findings_text}")
        else:
            parts.append("\n**Issues Flagged So Far**: None")
        
        if obs.findings_so_far:
            flagged_sections = set(f.section for f in obs.findings_so_far)
            parts.append(f"\n**Note**: You have already flagged issues in: {', '.join(flagged_sections)}. Avoid duplicate flags for the same section+type combination.")
        
        parts.append("\nChoose your next action. Respond with a JSON object only.")
        return "\n".join(parts)

    def _parse_action_str(self, action_str: str) -> Action:
        """Parse the JSON string action from the LLM into an Action object."""
        text = action_str.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        try:
            parsed = json.loads(text)
            return Action(**parsed)
        except (json.JSONDecodeError, ValueError) as e:
            return Action(action_type=ActionType.SUBMIT_AUDIT)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        task_id = "basic_completeness"
        hf_repo_id = None
        hf_revision = None
        
        if options:
            task_id = options.get("task_id", task_id)
            hf_repo_id = options.get("hf_repo_id", hf_repo_id)
            hf_revision = options.get("hf_revision", hf_revision)
            
        obs = self.env.reset(task_id=task_id, hf_repo_id=hf_repo_id, hf_revision=hf_revision)
        obs_str = self._format_observation(obs)
        return obs_str, {}

    def step(self, action_str: str):
        action = self._parse_action_str(action_str)
        obs, reward, done, info = self.env.step(action)
        
        obs_str = self._format_observation(obs)
        
        terminated = done and (action.action_type == ActionType.SUBMIT_AUDIT)
        truncated = done and not terminated
        
        reward_float = reward.total
        
        return obs_str, reward_float, terminated, truncated, info
