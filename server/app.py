from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from env.environment import ModelCardAuditEnv
from env.models import Action, Observation

app = FastAPI(title="ModelCardAudit-Env API")
env = ModelCardAuditEnv()

class ResetRequest(BaseModel):
    task_id: str = "basic_completeness"

@app.get("/")
def read_root():
    return {"message": "Welcome to ModelCardAudit-Env"}

@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    try:
        task_id = req.task_id if req else "basic_completeness"
        obs = env.reset(task_id)
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
            "info": info
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
            "regulatory_compliance"
        ]
    }
