---
title: ModelCardAudit-Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🏆 ModelCardAudit-Env

> An OpenEnv environment where an AI agent acts as a **responsible AI auditor**, reviewing ML model cards for completeness, technical consistency, bias documentation quality, and regulatory compliance against frameworks like the EU AI Act and NIST AI RMF.

---

## 🧠 Motivation

Model cards are the primary documentation for ML models, yet:

- **73% of HuggingFace model cards** are missing critical sections (bias analysis, limitations, intended use)
- Manual review is tedious and inconsistent
- Regulatory pressure is increasing (EU AI Act mandates documentation)
- No standardized tooling exists for automated model card quality assessment

**ModelCardAudit-Env** provides a realistic simulation where an AI agent:

1. Receives a model card document (structured into sections)
2. Reviews it against a compliance checklist (varying by task difficulty)
3. Flags issues, suggests improvements, and verifies technical claims
4. Produces a structured audit report scored against ground-truth annotations

---

## 📐 Environment Design

### Actions

| Action                | Description                                                         |
| :-------------------- | :------------------------------------------------------------------ |
| `read_section`        | Read a specific section of the model card                           |
| `flag_issue`          | Flag a compliance or quality issue with type, severity, description |
| `suggest_improvement` | Suggest content improvements for a section                          |
| `verify_claim`        | Verify a technical claim (e.g., metric accuracy)                    |
| `submit_audit`        | Submit the final audit report and end the episode                   |

### Observation Space

Each step returns an `Observation` containing:

- Task metadata and description
- Model card metadata (name, type, framework)
- Available sections and currently viewed section content
- Compliance checklist items
- Issues flagged so far
- Sections already reviewed
- Step count and steps remaining

### Reward Signal

Multi-dimensional reward combining:

- **Precision**: Correct findings / total findings
- **Recall**: Found issues / total real issues
- **Coverage**: Sections reviewed / total sections
- **Efficiency**: Bonus for fewer steps
- **Penalties**: For false positives and repeated reads

---

## 📋 Tasks

### Task 1: Basic Completeness Check (Easy)

- **Objective**: Identify missing required sections from a 10-item checklist
- **Max Steps**: 30
- **Grader**: `score = 0.5 * recall + 0.3 * precision + 0.2 * coverage`
- **Expected Baseline**: 0.60–0.70

### Task 2: Technical Consistency Audit (Medium)

- **Objective**: Find internal inconsistencies and insufficient documentation
- **Max Steps**: 45
- **Grader**: `score = 0.4 * recall + 0.3 * precision + 0.15 * suggestion_quality + 0.15 * severity_accuracy`
- **Expected Baseline**: 0.35–0.45

### Task 3: Regulatory Compliance Audit (Hard)

- **Objective**: Full audit against EU AI Act and NIST AI RMF standards
- **Max Steps**: 60
- **Grader**: `score = 0.35 * recall + 0.25 * precision + 0.15 * severity_accuracy + 0.15 * regulatory_mapping + 0.10 * efficiency`
- **Expected Baseline**: 0.15–0.25

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- (Optional) An OpenAI API key for LLM-driven inference

### Installation

```bash
# Clone and install
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Audit a Hugging Face Model Card

1. Open **New Audit** in the frontend.
2. Set **Source** to **Hugging Face repo**.
3. Enter a repo id in the format `owner/model-name` (example: `google/gemma-4-e2b`).
4. Optionally set a revision (defaults to `main`, then falls back to `master`).
5. Start the audit. The app fetches `README.md`, maps sections into the audit schema, and runs the normal workflow.

Notes:
- External Hugging Face cards do not have local ground-truth annotations in this project, so score values are best treated as directional.
- Checklist and audit actions still work exactly the same as local dataset cards.

### Running the Baseline Agent

```bash
# Deterministic local baseline
python inference.py

# Validator / LiteLLM proxy-backed run
export API_BASE_URL="https://your-litellm-proxy"
export API_KEY="your-validator-key"
export MODEL_NAME="gpt-4o-mini"        # optional, default is gpt-4o-mini
python inference.py
```

### Running Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
docker build -t modelcard-audit-env .
docker run -p 7860:7860 modelcard-audit-env
```

### HuggingFace Spaces

- Create an HF Space with `openenv` tag
- Use Docker SDK Space type
- The FastAPI server exposes on port 7860
- Endpoints: `/reset`, `/step`, `/state`, `/tasks`

---

## 📁 Project Structure

```
modelcard-audit-env/
├── openenv.yaml                 # OpenEnv metadata
├── Dockerfile                   # Container config
├── inference.py                 # Baseline agent script (naive + LLM)
├── README.md                    # This file
├── requirements.txt             # Dependencies
│
├── env/
│   ├── __init__.py
│   ├── environment.py           # Main environment (step/reset/state)
│   ├── models.py                # Pydantic typed models
│   ├── reward.py                # Reward function logic
│   └── graders.py               # Task-specific graders
│
├── data/
│   ├── model_cards/             # Synthetic model card documents (JSON)
│   │   ├── easy/                # 3 cards with obviously missing sections
│   │   ├── medium/              # 3 cards with subtle inconsistencies
│   │   └── hard/                # 3 cards for high-risk AI systems
│   ├── checklists/              # Compliance requirement checklists
│   │   ├── basic_completeness.json    (10 items)
│   │   ├── technical_consistency.json (15 items)
│   │   └── regulatory_compliance.json (26 items)
│   └── ground_truth/            # Annotated ground-truth audit findings
│       ├── easy_annotations.json
│       ├── medium_annotations.json
│       └── hard_annotations.json
│
├── server/
│   ├── __init__.py
│   └── app.py                   # FastAPI server
│
└── tests/
    ├── test_environment.py
    ├── test_models.py
    └── test_graders.py
```

---

## 🔌 API Reference

### `POST /reset`

Reset the environment for a new episode.

**Request Body:**

```json
{ "task_id": "basic_completeness" }
```

Task IDs: `basic_completeness`, `technical_consistency`, `regulatory_compliance`

**Response:** An `Observation` object.

### `POST /step`

Take an action in the environment.

**Request Body:**

```json
{
  "action_type": "flag_issue",
  "section_name": "bias_analysis",
  "issue_type": "insufficient",
  "severity": "high",
  "description": "Bias analysis lacks methodology and quantitative results."
}
```

**Response:**

```json
{
  "observation": { ... },
  "reward": { "total": 0.45, "precision_score": 0.8, ... },
  "done": false,
  "info": {}
}
```

### `GET /state`

Get the full internal state (for debugging).

### `GET /tasks`

List available task IDs.

---

## ⚙️ Environment Variables

| Variable          | Default                 | Description                           |
| :---------------- | :---------------------- | :------------------------------------ |
| `ENV_API_URL`     | `http://localhost:7860` | URL of the environment API server     |
| `API_BASE_URL`    | None                    | LiteLLM / validator proxy base URL    |
| `API_KEY`         | None                    | LiteLLM / validator proxy API key     |
| `MODEL_NAME`      | `gpt-4o-mini`           | Model to use for inference            |

---

## 📄 License

Open source for hackathon submission.
