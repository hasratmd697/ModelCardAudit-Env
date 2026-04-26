# We Trained an RL Agent to Audit Model Cards for Responsible AI

Model cards are one of the most important tools for transparent and responsible ML deployment, but reviewing them manually is slow, repetitive, and inconsistent across reviewers.

In this project, we built an environment where an autonomous agent audits model cards for documentation quality, technical consistency, and regulatory readiness. Then we trained that agent with reinforcement learning so it can improve from trajectory feedback rather than staying a static baseline.

## Why This Matters

Across public model hubs, many model cards still miss critical information such as:

- Intended use and misuse boundaries
- Bias and fairness analysis
- Limitations and failure modes
- Reproducibility details
- Regulatory traceability for high-risk use cases

As model adoption grows, weak documentation becomes a reliability and compliance risk. We wanted a practical way to score and improve model card quality at scale.

## What We Built

Our project combines:

- A custom OpenEnv-compatible audit environment
- Three task levels: basic completeness, technical consistency, and regulatory compliance
- Structured action space: read sections, flag issues, suggest improvements, verify claims, submit audit
- Reward functions that evaluate precision, recall, coverage, efficiency, and penalty terms
- A frontend and API workflow for interactive and autonomous audits

## RL Milestone: Agent Loads Successfully at Startup

A major milestone is complete: the RL agent now loads successfully at application startup from:

**Hasrathussain/audit-agent-rl**

Startup behavior now includes:

- Base model initialization for inference
- LoRA adapter load from Hugging Face Hub
- Readiness signal exposed by the API
- Graceful deterministic fallback if RL loading fails

This means the service remains available while still taking advantage of the trained RL policy whenever it is ready.

## How Inference Works

- The backend attempts to initialize RL once at startup
- Health endpoint reports service and RL readiness
- Autonomous audits run through a single end-to-end endpoint
- Runtime mode is explicit: RL policy when loaded, deterministic fallback otherwise

This design keeps deployment robust in real-world environments where model download/auth conditions may vary.

## Try It

- Live Space: https://huggingface.co/spaces/<your-username>/<your-space>
- RL Model: https://huggingface.co/Hasrathussain/audit-agent-rl
- Source Code: https://github.com/<your-username>/<your-repo>

## What We Want to Improve Next

- Better severity calibration for edge-case findings
- Stronger regulatory mapping granularity
- Larger benchmark set with more diverse real-world model cards
- Richer evaluation comparing RL policy vs deterministic baseline across tasks

## Looking for Feedback

If you work on model governance, evals, or reliable ML deployment, feedback is very welcome.

We are especially interested in:

- Additional checklist frameworks to support
- Better reward shaping ideas for audit quality
- Real model cards we should test against

If you try the demo, we would love to hear what worked and what should improve next.
