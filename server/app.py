import os
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from env.environment import ModelCardAuditEnv
from env.models import Action, ActionType, Observation

# ── RL agent state (loaded once at startup) ───────────────────────────────────
_rl_model: Any = None
_rl_tokenizer: Any = None


def _load_rl_agent():
    """Try to load the GRPO fine-tuned LoRA model from HuggingFace Hub."""
    global _rl_model, _rl_tokenizer
    hub_id = os.environ.get("RL_MODEL_ID", "Hasrathussain/audit-agent-rl")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"[RL] Loading base model for inference...")
        _rl_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        _rl_model = PeftModel.from_pretrained(base, hub_id)
        _rl_model.eval()
        print(f"[RL] RL agent loaded successfully from {hub_id}")
    except Exception as exc:
        print(f"[RL] Could not load RL agent ({exc}). Falling back to deterministic baseline.")
        _rl_model = None
        _rl_tokenizer = None


# ── System / task prompts ─────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = (
    "You are an expert AI model card auditor. Your job is to review ML model cards "
    "for completeness, technical consistency, and regulatory compliance.\n\n"
    "Respond with a JSON object containing your chosen action. Example:\n"
    '{"action_type": "read_section", "section_name": "bias_analysis"}'
)

TASK_STRATEGIES = {
    "basic_completeness": (
        "Strategy: identify MISSING required sections. "
        "Use issue_type='missing', severity='high' for absent sections."
    ),
    "technical_consistency": (
        "Strategy: find INTERNAL INCONSISTENCIES. "
        "Use issue_type='inconsistent' for contradictions, 'insufficient' for thin sections."
    ),
    "regulatory_compliance": (
        "Strategy: audit against EU AI Act & NIST AI RMF. "
        "Always include the 'regulation' field citing the exact article."
    ),
}


def _format_obs(obs_dict: dict) -> str:
    """Convert an observation dict into a compact prompt string."""
    parts = [
        f"Task: {obs_dict['task_description']}",
        f"Model: {obs_dict['model_card_metadata'].get('model_name', 'Unknown')}",
        f"Step {obs_dict['step_count']} / steps remaining {obs_dict['steps_remaining']}",
        f"Available: {', '.join(obs_dict['available_sections'])}",
        f"Reviewed: {', '.join(obs_dict['sections_reviewed']) or 'none'}",
    ]
    unreviewed = [s for s in obs_dict["available_sections"] if s not in obs_dict["sections_reviewed"]]
    parts.append(f"Not yet reviewed: {', '.join(unreviewed) or 'all done'}")
    if obs_dict.get("current_section"):
        parts.append(f"Current section content:\n{obs_dict['current_section'][:800]}")
    if obs_dict.get("findings_so_far"):
        parts.append(f"Findings so far: {len(obs_dict['findings_so_far'])}")
    parts.append("Respond with a JSON action object only.")
    return "\n".join(parts)


def _parse_action(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action_type": "submit_audit"}


def _rl_next_action(obs_dict: dict, task_id: str) -> dict:
    """Run one forward pass through the RL model to get the next action."""
    if _rl_model is None or _rl_tokenizer is None:
        return {"action_type": "submit_audit"}

    system_prompt = BASE_SYSTEM_PROMPT + "\n" + TASK_STRATEGIES.get(task_id, "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _format_obs(obs_dict)},
    ]
    text_prompt = _rl_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _rl_tokenizer(text_prompt, return_tensors="pt").to(_rl_model.device)
    with torch.no_grad():
        outputs = _rl_model.generate(
            **inputs, max_new_tokens=256, temperature=0.1, do_sample=False
        )
    completion = _rl_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return _parse_action(completion)


# ── Deterministic fallback ────────────────────────────────────────────────────
from inference import plan_findings


def _deterministic_next_action(obs_dict: dict, task_id: str, pending: list) -> tuple:
    unreviewed = [
        s for s in obs_dict["available_sections"] if s not in obs_dict["sections_reviewed"]
    ]
    if unreviewed:
        return {"action_type": "read_section", "section_name": unreviewed[0]}, pending
    if not pending:
        pending = plan_findings(task_id, obs_dict)
    if pending:
        return pending.pop(0), pending
    return {"action_type": "submit_audit"}, pending


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="ModelCardAudit-Env API")
env = ModelCardAuditEnv()


@app.on_event("startup")
async def startup_event():
    _load_rl_agent()


# ── Request models ─────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "basic_completeness"
    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None


class RunAuditRequest(BaseModel):
    task_id: str = "basic_completeness"
    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None


# ── Existing endpoints ─────────────────────────────────────────────────────────
@app.get("/api-root")
def api_root():
    return {
        "message": "Online",
        "rl_agent_loaded": _rl_model is not None,
    }


@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    try:
        task_id = req.task_id if req else "basic_completeness"
        hf_repo_id = req.hf_repo_id if req else None
        hf_revision = req.hf_revision if req else None
        obs = env.reset(task_id, hf_repo_id=hf_repo_id, hf_revision=hf_revision)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step_env(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state():
    return env.state()


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            "basic_completeness",
            "technical_consistency",
            "regulatory_compliance",
        ]
    }


# ── NEW: Autonomous RL-powered audit endpoint ──────────────────────────────────
@app.post("/run-audit")
def run_audit(req: RunAuditRequest):
    """
    Runs a complete autonomous audit episode using the RL agent (or the
    deterministic baseline if the model is not yet trained/published).
    Returns the full episode trajectory and final score.
    """
    episode_env = ModelCardAuditEnv()
    try:
        obs = episode_env.reset(
            req.task_id,
            hf_repo_id=req.hf_repo_id,
            hf_revision=req.hf_revision,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    mode = "rl-agent" if _rl_model is not None else "deterministic"
    steps_log = []
    pending = []
    done = False
    info: Dict[str, Any] = {}

    while not done:
        obs_dict = obs.model_dump()

        if _rl_model is not None:
            action_dict = _rl_next_action(obs_dict, req.task_id)
        else:
            action_dict, pending = _deterministic_next_action(obs_dict, req.task_id, pending)

        try:
            action_obj = Action(**action_dict)
        except Exception:
            action_obj = Action(action_type=ActionType.SUBMIT_AUDIT)

        obs, reward, done, info = episode_env.step(action_obj)

        steps_log.append({
            "step": obs_dict["step_count"] + 1,
            "action": action_dict,
            "reward": reward.model_dump(),
            "done": done,
        })

    final_score = info.get("score", 0.0)
    return {
        "mode": mode,
        "task_id": req.task_id,
        "final_score": final_score,
        "total_steps": len(steps_log),
        "findings": [f.model_dump() for f in obs.findings_so_far],
        "steps": steps_log,
    }


# ── Serve frontend static files ────────────────────────────────────────────────
frontend_dist = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
)
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def fallback_root():
        return {"message": "Frontend build not found. API is running."}


def main():
    """Entry point for running the ModelCardAudit-Env server."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
