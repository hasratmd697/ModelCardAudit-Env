import os
import json
import argparse
from pathlib import Path
from openai import OpenAI
import time

# Ensure we can import from env and inference
from env.environment import ModelCardAuditEnv
from env.models import Action, ActionType
from inference import get_system_prompt, format_observation, get_model_name, build_proxy_client, proxy_credentials_available, plan_findings, parse_action

def collect_trajectories(num_rollouts=1, output_file="data/trajectories/expert.jsonl", task=None):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tasks = [task] if task else ["basic_completeness", "technical_consistency", "regulatory_compliance"]
    
    client = build_proxy_client() if proxy_credentials_available() else None
    if client:
        print(f"Using proxy client with model: {get_model_name()}")
    else:
        print("Using deterministic baseline for trajectory collection.")
    
    # We will just append to output_file or create it
    with open(output_path, "w", encoding="utf-8") as f:
        for t in tasks:
            print(f"Collecting trajectories for task: {t}")
            for rollout in range(num_rollouts):
                env = ModelCardAuditEnv()
                # Initialize observation
                obs = env.reset(task_id=t)
                
                system_prompt = get_system_prompt(t)
                done = False
                
                # Deterministic fallback state
                pending_actions = None
                
                while not done:
                    # Current observation formatted as user prompt
                    obs_dict = obs.model_dump()
                    user_prompt = format_observation(obs_dict)
                    
                    # For Qwen and other standard instruct models, ChatML format is common.
                    # GRPO trainer expects string prompts.
                    # We will format it as a standard chat list that the TRL trainer dataset parsing can handle or just raw text.
                    # TRL's GRPOTrainer can accept prompts as a list of dicts (messages) or string. We use list of dicts for safety if it gets tokenized, but string is fine. Let's save both or string.
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    action_text = ""
                    action_dict = None
                    
                    if client:
                        try:
                            completion = client.chat.completions.create(
                                model=get_model_name(),
                                messages=messages,
                                temperature=0.7,
                                max_tokens=256,
                            )
                            action_text = completion.choices[0].message.content
                            action_dict = parse_action(action_text)
                            time.sleep(1) # rate limit protection
                        except Exception as e:
                            print(f"LLM error: {e}, falling back to deterministic action.")
                            action_dict = None
                    
                    if action_dict is None:
                        # Deterministic fallback
                        unreviewed = [s for s in obs_dict['available_sections'] if s not in obs_dict['sections_reviewed']]
                        if unreviewed:
                            action_dict = {"action_type": "read_section", "section_name": unreviewed[0]}
                        elif pending_actions is None:
                            pending_actions = plan_findings(t, obs_dict)
                            if pending_actions:
                                action_dict = pending_actions.pop(0)
                            else:
                                action_dict = {"action_type": "submit_audit"}
                        elif pending_actions:
                            action_dict = pending_actions.pop(0)
                        else:
                            action_dict = {"action_type": "submit_audit"}
                        
                        action_text = json.dumps(action_dict)
                    
                    # Step the environment
                    try:
                        action_obj = Action(**action_dict)
                    except Exception:
                        action_obj = Action(action_type=ActionType.SUBMIT_AUDIT)
                    
                    # Before stepping, keep track of step count
                    current_step = obs.step_count
                    
                    obs, reward, done, info = env.step(action_obj)
                    
                    # Log the transition
                    # `prompt` should be the messages list for GRPOTrainer which uses `apply_chat_template`
                    log_entry = {
                        "prompt": messages,
                        "completion": action_text,
                        "reward": info.get("score", reward.total) if done else reward.total,
                        "task_id": t,
                        "step": current_step,
                        "episode_id": f"{t}_r{rollout}"
                    }
                    f.write(json.dumps(log_entry) + "\n")
                    
                print(f"  Rollout {rollout+1}/{num_rollouts} done. Final Score: {info.get('score', 0.0):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Collect trajectories for RL training.")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts per task.")
    parser.add_argument("--output", type=str, default="data/trajectories/expert.jsonl", help="Output JSONL file.")
    parser.add_argument("--task", type=str, default=None, help="Specific task to collect for (default: all).")
    args = parser.parse_args()
    
    collect_trajectories(args.num_rollouts, args.output, args.task)

if __name__ == "__main__":
    main()
