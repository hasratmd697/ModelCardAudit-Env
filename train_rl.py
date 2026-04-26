import os
import sys
import csv
import json
import argparse
import time
import threading
import inspect
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import warnings

# ═══════════════════════════════════════════════════════════════
# VLLM STUB MUST BE BEFORE ANY `trl` IMPORTS
# ═══════════════════════════════════════════════════════════════
try:
    import vllm
except ImportError:
    import types, importlib.machinery
    def _make_stub(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__package__ = name; mod.__path__ = []; return mod
    _vllm = _make_stub("vllm")
    class _LLM:
        def __init__(self, *a, **kw): pass
    class _SamplingParams:
        def __init__(self, *a, **kw): pass
    _vllm.LLM = _LLM; _vllm.SamplingParams = _SamplingParams
    _vllm_dist = _make_stub("vllm.distributed")
    _vllm_comm = _make_stub("vllm.distributed.device_communicators")
    _vllm_pync = _make_stub("vllm.distributed.device_communicators.pynccl")
    _vllm_pync.PyNcclCommunicator = type("PyNcclCommunicator", (), {})
    for _n, _m in [("vllm",_vllm),("vllm.distributed",_vllm_dist),
                   ("vllm.distributed.device_communicators",_vllm_comm),
                   ("vllm.distributed.device_communicators.pynccl",_vllm_pync)]:
        sys.modules[_n] = _m

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig

warnings.filterwarnings("ignore")

from env.models import Action, ActionType

# Training state
TRAINING_STATE = {
    "status": "starting", "message": "Booting...", "phase": "init",
    "started_at": None, "ended_at": None, "error": None,
    "step": 0, "reward_mean": None,
}

def _set_state(status, message, phase=None, error=None):
    TRAINING_STATE.update({"status": status, "message": message, "error": error})
    if phase: TRAINING_STATE["phase"] = phase
    if status == "running" and TRAINING_STATE["started_at"] is None:
        TRAINING_STATE["started_at"] = time.time()
    if status in {"completed", "failed"}:
        TRAINING_STATE["ended_at"] = time.time()

def _render_status():
    s = TRAINING_STATE
    elapsed = f"{int(time.time()-s['started_at'])}s" if s["started_at"] else "unknown"
    lines = [
        f"status:  {s['status']}", f"phase:   {s['phase']}",
        f"message: {s['message']}", f"elapsed: {elapsed}", f"step:    {s['step']}",
    ]
    if s["reward_mean"] is not None: lines.append(f"reward:  {s['reward_mean']:.3f}")
    if s["error"]: lines.append(f"error:   {s['error']}")
    return "\n".join(lines)

SYSTEM_PROMPT = (
    "Output ONLY valid JSON, no markdown, no extra text.\n"
    'Formats: {"action_type":"read_section","section_name":"..."} '
    'or {"action_type":"flag_issue","issue_type":"...","severity":"low|medium|high","description":"..."} '
    'or {"action_type":"submit_audit","summary":"..."}'
)

class _CSVLogCallback(TrainerCallback):
    """
    Writes one row per training step to logs/training_log.csv.
    Columns: phase, step, loss, reward_mean, reward_std, kl
    This file is the source of truth for generating reward-curve plots.
    """
    def __init__(self, phase: str, log_dir: str = "logs"):
        self.phase = phase
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        # Write header only once (append mode so SFT + GRPO both land in one file)
        write_header = not os.path.exists(self.csv_path)
        self._f = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._f,
            fieldnames=["phase", "step", "loss", "reward_mean", "reward_std", "kl"],
        )
        if write_header:
            self._writer.writeheader()
            self._f.flush()

    def on_step_end(self, args, state, control, **kwargs):
        if not state.log_history:
            return
        last = state.log_history[-1]
        row = {
            "phase":       self.phase,
            "step":        state.global_step,
            "loss":        last.get("loss") or last.get("train/loss", ""),
            "reward_mean": last.get("reward") or last.get("rewards/mean") or last.get("train/reward", ""),
            "reward_std":  last.get("reward_std") or last.get("rewards/std", ""),
            "kl":          last.get("kl") or last.get("train/kl", ""),
        }
        self._writer.writerow(row)
        self._f.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._f.close()
        print(f"[CSV LOG] Saved training log → {self.csv_path}", flush=True)


class _StatusCallback(TrainerCallback):
    def __init__(self, phase="grpo"): self.phase = phase
    def on_step_end(self, args, state, control, **kwargs):
        TRAINING_STATE["step"] = state.global_step
        TRAINING_STATE["phase"] = self.phase
        if state.log_history:
            last = state.log_history[-1]
            for k in ("reward","rewards/mean","train/reward","train/loss"):
                if k in last: TRAINING_STATE["reward_mean"] = last[k]; break
        total = args.max_steps if args.max_steps > 0 else "?"
        pct = f"{int(state.global_step/args.max_steps*100)}%" if args.max_steps > 0 else "?"
        elapsed = int(time.time() - (TRAINING_STATE["started_at"] or time.time()))
        print(f"[{self.phase.upper()} step {state.global_step}/{total} | {pct} | {elapsed}s]", flush=True)

# 🔍 DEBUG CALLBACK: Prints actual generations every 30 steps
class _GenerationLogger(TrainerCallback):
    def __init__(self, tokenizer, interval=30):
        self.tokenizer = tokenizer
        self.interval = interval
        self.step_count = 0
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.interval != 0: return
        if not state.is_world_process_zero: return
        try:
            self.model.eval()
            with torch.no_grad():
                inp = self.tokenizer(SYSTEM_PROMPT + "\nUser: Audit section 1", return_tensors="pt").to(self.model.device)
                out = self.model.generate(**inp, max_new_tokens=64, do_sample=True, temperature=0.9)
                txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
                print(f"\n[DEBUG GEN STEP {self.step_count}] {txt[-200:]}\n", flush=True)
        except Exception as e:
            print(f"[DEBUG GEN ERROR] {e}")

def keep_alive():
    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = _render_status().encode()
            self.send_response(200); self.send_header("Content-type","text/plain")
            self.send_header("Content-Length",str(len(body))); self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a): pass
    def _run():
        try: HTTPServer(("0.0.0.0",7860),_H).serve_forever()
        except Exception as e: print(f"Keep-alive error: {e}")
    threading.Thread(target=_run, daemon=True).start()
    print("Keep-alive server started on :7860")

def _extract_text(c):
    if isinstance(c, list): return c[-1].get("content","") if c else ""
    return str(c)

def _extract_json(text):
    text = text.strip()
    for fence in ("```json","```"):
        if fence in text: text = text.split(fence)[1].split("```")[0].strip(); break
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s: text = text[s:e+1]
    try: return json.loads(text), text
    except: return None, text

def _inject_system(prompt):
    sys_msg = {"role":"system","content":SYSTEM_PROMPT}
    if isinstance(prompt, list):
        return [sys_msg] + [m for m in prompt if m.get("role") != "system"]
    return [sys_msg, {"role":"user","content":str(prompt)}]

def format_reward_func(prompts, completions, **kwargs):
    out = []
    for c in completions:
        text = _extract_text(c).strip()
        for fence in ("```json", "```"):
            if text.startswith(fence): text = text[len(fence):]
            if text.endswith(fence): text = text[:-len(fence)]
        text = text.strip()
        
        try:
            json.loads(text)
            out.append(1.0)
        except:
            if text.count("{") >= text.count("}") and "{" in text:
                out.append(0.5)
            else:
                out.append(-0.5)
    return out

def audit_reward_func(prompts, completions, **kwargs):
    out = []
    for prompt, c in zip(prompts, completions):
        pstr = str(prompt)
        parsed, _ = _extract_json(_extract_text(c))
        if parsed is None: out.append(-1.0); continue
        atype = parsed.get("action_type",""); r = 0.0
        if atype == "read_section":
            name = parsed.get("section_name","")
            if not name: r = -0.3
            elif "Sections NOT Yet Reviewed" in pstr:
                r = 1.5 if name in pstr.split("Sections NOT Yet Reviewed")[1] else -0.3
            else: r = 0.3
        elif atype == "flag_issue":
            r += 0.4 if parsed.get("issue_type") else 0
            r += 0.4 if parsed.get("severity") in ("low","medium","high") else 0
            desc = parsed.get("description","")
            r += 0.4 if isinstance(desc,str) and len(desc)>10 else 0
        elif atype == "submit_audit":
            done = "All reviewed" in pstr or "all sections" in pstr.lower()
            r = 2.0 if done else -0.8
        else: r = -0.3
        out.append(r)
    return out

def build_sft_dataset(path, max_samples, tokenizer):
    import pandas as pd
    df = pd.read_json(path, lines=True)
    print(f"JSONL columns: {list(df.columns)}")
    comp_col = next((c for c in ("completion","action","response","output","answer") if c in df.columns), None)
    if comp_col is None:
        print("WARNING: no completion column — skipping SFT warmup."); return None
    df["_pstr"] = df["prompt"].astype(str)
    df = df.drop_duplicates(subset=["_pstr"])
    if max_samples > 0: df = df.head(max_samples)

    records = []
    for _, row in df.iterrows():
        messages = _inject_system(row["prompt"])
        comp = row[comp_col]
        comp_str = json.dumps(comp) if isinstance(comp, dict) else str(comp).strip()
        messages.append({"role":"assistant","content":comp_str})
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            records.append({"text": text})
        except Exception as exc:
            print(f"Template error (skipped): {exc}")

    if not records:
        print("WARNING: SFT dataset empty."); return None
    print(f"SFT dataset: {len(records)} samples")

    ds = Dataset.from_list(records)

    def tokenize_fn(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=384, padding=False)
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds

def build_grpo_dataset(path, max_samples):
    import pandas as pd
    df = pd.read_json(path, lines=True)
    df["_pstr"] = df["prompt"].astype(str)
    df = df.drop_duplicates(subset=["_pstr"])
    if max_samples > 0: df = df.head(max_samples)
    ds = Dataset.from_pandas(df[["prompt"]])
    return ds.map(lambda ex: {"prompt": _inject_system(ex["prompt"])})

def main():
    keep_alive()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trajectories", type=str, default="data/trajectories/expert.jsonl")
    parser.add_argument("--log_dir",      type=str, default=os.environ.get("LOG_DIR", "logs"),
                        help="Directory for training_log.csv output")
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument("--push_to_hub",  action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="Hasrathussain/audit-agent-rl")
    parser.add_argument("--max_steps",    type=int, default=300)
    parser.add_argument("--sft_steps",    type=int, default=50,
                        help="SFT warmup steps before GRPO. 0 = skip.")
    parser.add_argument("--max_samples",  type=int, default=1500)
    args = parser.parse_args()

    if not os.path.exists(args.trajectories):
        print(f"ERROR: {args.trajectories} not found"); return

    _set_state("running", "Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    _set_state("running", "Loading model…")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            ),
            device_map="auto", trust_remote_code=True,
        )
    except Exception as exc:
        print(f"4-bit failed ({exc}), using fp16…")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    if not hasattr(model, "warnings_issued"): model.warnings_issued = {}
    model.print_trainable_parameters()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1 — SFT WARMUP
    # ═══════════════════════════════════════════════════════════════
    if args.sft_steps > 0:
        _set_state("running", f"SFT warmup ({args.sft_steps} steps)…", phase="sft")
        print(f"\n{'='*55}\nPHASE 1: SFT warmup ({args.sft_steps} steps)\n{'='*55}")
        sft_ds = build_sft_dataset(args.trajectories, args.max_samples, tokenizer)
        if sft_ds:
            orig_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "right"

            _sft_kw = "processing_class" if "processing_class" in set(inspect.signature(SFTTrainer.__init__).parameters) else "tokenizer"
            sft_trainer = SFTTrainer(
                model=model,
                **{_sft_kw: tokenizer},
                args=SFTConfig(
                    output_dir="models/audit-agent-sft",
                    max_steps=args.sft_steps,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    gradient_checkpointing=True,
                    fp16=True,
                    learning_rate=2e-4,
                    logging_steps=5,
                    save_strategy="no",
                    report_to="none",
                    remove_unused_columns=False,
                    max_seq_length=384,
                ),
                train_dataset=sft_ds,
                callbacks=[
                    _StatusCallback(phase="sft"),
                    _CSVLogCallback(phase="sft", log_dir=args.log_dir),
                ],
            )
            sft_trainer.train()
            del sft_trainer
            torch.cuda.empty_cache()

            tokenizer.padding_side = orig_padding_side
            print("SFT warmup complete.\n")
        else:
            print("SFT skipped (no completion column).\n")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2 — GRPO
    # ═══════════════════════════════════════════════════════════════
    _set_state("running", f"GRPO ({args.max_steps} steps)…", phase="grpo")
    print(f"\n{'='*55}\nPHASE 2: GRPO ({args.max_steps} steps)\n{'='*55}")

    grpo_ds = build_grpo_dataset(args.trajectories, args.max_samples)
    print(f"GRPO dataset: {len(grpo_ds)} prompts")

    _tok_kw = "processing_class" if "processing_class" in set(inspect.signature(GRPOTrainer.__init__).parameters) else "tokenizer"
    print(f"GRPOTrainer kwarg: {_tok_kw}")

    trainer = GRPOTrainer(
        model=model,
        **{_tok_kw: tokenizer},
        reward_funcs=[format_reward_func, audit_reward_func],
        args=GRPOConfig(
            output_dir="models/audit-agent-rl",
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            num_generations=8,          # ✅ CRITICAL for reward_std > 0
            max_prompt_length=256,
            max_completion_length=128,  # ✅ Prevents premature cutoff
            temperature=0.9,            # ✅ Ensures diversity
            # top_p=0.9,                # ❌ REMOVED: Not supported in GRPOConfig
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            fp16=True,
            learning_rate=3e-5,
            beta=0.04,
            max_grad_norm=1.0,          # ✅ Prevents NaN/KL spikes
            logging_steps=1,
            save_strategy="epoch",
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id if args.push_to_hub else None,
            remove_unused_columns=False,
            report_to="none",
            use_vllm=False,
        ),
        train_dataset=grpo_ds,
        callbacks=[
            _StatusCallback(phase="grpo"),
            _CSVLogCallback(phase="grpo", log_dir=args.log_dir),
            _GenerationLogger(tokenizer, interval=30),
        ],
    )

    _set_state("running", f"GRPO training started…", phase="grpo")
    try:
        trainer.train()
        if args.push_to_hub:
            _set_state("running", f"Pushing: {args.hub_model_id}")
            trainer.push_to_hub()
        else:
            _set_state("running", "Saving locally…")
            trainer.save_model("models/audit-agent-rl")
            tokenizer.save_pretrained("models/audit-agent-rl")
        _set_state("completed", "Training completed successfully.")
        print("Done.")
    except Exception as exc:
        _set_state("failed", "Training failed.", error=str(exc)); raise

if __name__ == "__main__":
    main()
