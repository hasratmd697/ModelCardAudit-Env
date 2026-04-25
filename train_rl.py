import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# vllm stub — trl calls importlib.util.find_spec("vllm") and then does
# `from vllm import LLM, SamplingParams` at module level.  We satisfy both
# checks by registering a proper types.ModuleType with a real ModuleSpec and
# stub LLM / SamplingParams classes.  use_vllm=False (the default) means the
# stubs are never actually instantiated at runtime.
# ---------------------------------------------------------------------------
try:
    import vllm  # noqa: F401  – already installed, nothing to do
except ImportError:
    import types
    import importlib.machinery

    def _make_stub(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__package__ = name
        mod.__path__ = []          # mark it as a package
        return mod

    # Top-level vllm stub with the classes trl imports
    _vllm = _make_stub("vllm")

    class _LLM:                                   # noqa: N801
        def __init__(self, *a, **kw): pass

    class _SamplingParams:                        # noqa: N801
        def __init__(self, *a, **kw): pass

    _vllm.LLM = _LLM
    _vllm.SamplingParams = _SamplingParams

    # Sub-package stubs (needed by older trl builds that import pynccl)
    _vllm_dist = _make_stub("vllm.distributed")
    _vllm_comm = _make_stub("vllm.distributed.device_communicators")
    _vllm_pync = _make_stub("vllm.distributed.device_communicators.pynccl")
    _vllm_pync.PyNcclCommunicator = type("PyNcclCommunicator", (), {})

    for _mod_name, _mod_obj in [
        ("vllm", _vllm),
        ("vllm.distributed", _vllm_dist),
        ("vllm.distributed.device_communicators", _vllm_comm),
        ("vllm.distributed.device_communicators.pynccl", _vllm_pync),
    ]:
        sys.modules[_mod_name] = _mod_obj


import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import warnings
warnings.filterwarnings("ignore")

from env.models import Action, ActionType

def parse_action_safe(completion_text):
    text = completion_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    try:
        parsed = json.loads(text)
        return Action(**parsed)
    except Exception:
        return Action(action_type=ActionType.SUBMIT_AUDIT)

def audit_reward_func(prompts, completions, **kwargs):
    """
    Evaluates completions using a heuristic reward based on the prompt state.
    This avoids the complexity of perfectly syncing an asynchronous environment
    for each generated completion, while still incentivizing valid actions.
    """
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        reward = 0.0
        
        # Convert prompt to string if it's a list of dicts (ChatML)
        prompt_str = str(prompt)
        
        # 1. Valid JSON reward
        try:
            # trl returns completions as plain str (standard) or as
            # [{"role": "assistant", "content": "..."}] (conversational).
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = completion
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(text)
            action_type = parsed.get("action_type")
            reward += 0.5  # Valid JSON format bonus
            
            # 2. Logic rewards based on prompt context
            if action_type == "read_section":
                section_name = parsed.get("section_name", "")
                if "Sections NOT Yet Reviewed" in prompt_str and section_name in prompt_str.split("Sections NOT Yet Reviewed")[1]:
                    reward += 1.0  # Good: reading a new valid section
                else:
                    reward -= 0.5  # Bad: reading an invalid or already read section
            elif action_type == "flag_issue":
                if parsed.get("issue_type") and parsed.get("severity"):
                    reward += 1.0  # Good: properly formatted flag
                else:
                    reward -= 0.5  # Bad: missing required fields
            elif action_type == "submit_audit":
                if "All reviewed" in prompt_str:
                    reward += 2.0  # Excellent: submitting when all sections read
                else:
                    reward -= 1.0  # Bad: submitting before reviewing all sections
            else:
                reward += 0.1 # Generic valid action
                
        except json.JSONDecodeError:
            reward -= 1.0  # Heavy penalty for unparseable output
            
        rewards.append(reward)
        
    return rewards

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

def keep_alive():
    """Starts a dummy web server on port 7860 to satisfy HF Spaces health checks."""
    class DummyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Training in progress...")
        def log_message(self, format, *args):
            pass  # Suppress HTTP logging

    def run_server():
        server_address = ('0.0.0.0', 7860)
        try:
            httpd = HTTPServer(server_address, DummyHandler)
            print("Started dummy server on port 7860 to keep HF Space alive.")
            httpd.serve_forever()
        except Exception as e:
            print(f"Dummy server failed to start: {e}")

    threading.Thread(target=run_server, daemon=True).start()

def main():
    keep_alive()  # Start the background server immediately
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trajectories", type=str, default="data/trajectories/expert.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="Hasrathussain/audit-agent-rl")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps for smoke testing")
    args = parser.parse_args()
    
    # 1. Load Dataset
    if not os.path.exists(args.trajectories):
        print(f"Trajectories file {args.trajectories} not found!")
        return
        
    import pandas as pd
    df = pd.read_json(args.trajectories, lines=True)
    # We only need the prompts for GRPO
    # Drop duplicates to avoid over-weighting identical states
    df['prompt_str'] = df['prompt'].astype(str)
    df = df.drop_duplicates(subset=['prompt_str'])
    df = df[['prompt']]
    dataset = Dataset.from_pandas(df)
    
    # 2. Setup Model with QLoRA
    print(f"Loading base model {args.base_model}...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            device_map="auto"
        )
    except Exception as e:
        print(f"Warning: Could not load with 4-bit quantization (missing bitsandbytes or GPU?): {e}")
        print("Loading without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA ourselves so we can fix the warnings_issued attribute before
    # GRPOTrainer.__init__ tries to access it.  trl==0.14.0 calls
    #   model.warnings_issued["estimate_tokens"] = True
    # but the installed transformers version doesn't pre-initialise this dict on
    # model instances, and the PEFT attribute-lookup chain doesn't find it either.
    from peft import get_peft_model as _get_peft_model
    model = _get_peft_model(model, peft_config)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}       # satisfy trl 0.14.0's assumption

    # 3. Setup GRPO Trainer
    # ── Memory budget for a T4 (16 GiB) ──────────────────────────────────────
    # batch=1 × generations=2 × prompt=512 + completion=128 → ~640 tokens/seq
    # gradient_checkpointing trades activation memory for recompute.
    # gradient_accumulation_steps=4 restores effective batch=4 without extra VRAM.
    training_args = GRPOConfig(
        output_dir="models/audit-agent-rl",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        # ── Sequence / generation dims (main OOM culprit) ──
        per_device_train_batch_size=1,   # was 4
        num_generations=2,               # was 4  (must divide batch evenly)
        max_prompt_length=512,           # was 2048
        max_completion_length=128,       # was 256
        # ── Gradient accumulation restores effective batch size ──
        gradient_accumulation_steps=4,
        # ── Memory savers ──
        gradient_checkpointing=True,
        fp16=True,                       # keep everything in float16
        # ── Optimiser / logging ──
        learning_rate=5e-5,
        beta=0.04,
        logging_steps=1,
        save_strategy="epoch",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        remove_unused_columns=False,
        report_to="none",
    )

    # peft_config is NOT passed here — the model is already wrapped with LoRA above.
    # GRPOTrainer will create a frozen reference-model copy (correct GRPO behaviour).
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[audit_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    # 4. Train
    print("Starting GRPO Training...")
    trainer.train()
    
    if args.push_to_hub:
        print(f"Pushing to Hub: {args.hub_model_id}")
        trainer.push_to_hub()
    else:
        print("Saving locally...")
        trainer.save_model("models/audit-agent-rl")

if __name__ == "__main__":
    main()
