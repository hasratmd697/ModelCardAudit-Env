# Software Requirements Specification (SRS)
## ModelCardAudit-Env

---

## 1. Introduction

### 1.1 Purpose
The purpose of this Software Requirements Specification (SRS) is to provide a comprehensive description of the **ModelCardAudit-Env** project. This document outlines the functional and non-functional requirements, system architecture, and operating environment of the platform. It is intended for software developers, QA testers, and project stakeholders.

### 1.2 Document Conventions
- **API**: Application Programming Interface
- **JSON**: JavaScript Object Notation
- **LLM**: Large Language Model
- **ML**: Machine Learning
- **OpenEnv**: A framework for standardizing agent environments

### 1.3 Intended Audience
- Developers and engineers involved in the project.
- Judges and reviewers of the OpenEnv Hackathon.
- Future contributors and maintainers.

### 1.4 Project Scope
**ModelCardAudit-Env** is an OpenEnv environment where an AI agent acts as a responsible AI auditor. The agent reviews ML model cards for completeness, technical consistency, bias documentation quality, and regulatory compliance (e.g., EU AI Act, NIST AI RMF). The system addresses the widespread issue of incomplete or inconsistent ML model documentation, specifically targeting the high percentage of inadequate model cards on platforms like HuggingFace.

---

## 2. Overall Description

### 2.1 Product Perspective
ModelCardAudit-Env is a standalone environment built to conform to the OpenEnv specification. It simulates an auditing workflow where an AI agent interacts with a provided model card, reviews its sections against predefined compliance checklists, and submits a final audit report. It includes a frontend for user interaction and an API server for agent integration.

### 2.2 Product Functions
- **Model Card Loading**: Load local synthetic model cards or external model cards from HuggingFace repositories.
- **RL Agent Startup**: On API startup, load a base language model plus a GRPO-trained LoRA adapter from HuggingFace Hub for autonomous auditing.
- **Section Reading**: Allow the agent to incrementally read specific sections of a model card.
- **Issue Flagging**: Provide mechanisms for the agent to flag issues with varying severities and types.
- **Improvement Suggestion**: Enable the agent to suggest actionable improvements for flagged issues.
- **Claim Verification**: Support the verification of technical and statistical claims present in the model card.
- **Audit Submission & Grading**: Conclude the audit episode and score the agent's performance using predefined rubrics and ground-truth annotations.

### 2.3 User Classes and Characteristics
- **AI Agents**: Automated scripts or LLM-driven agents interacting with the environment programmatically.
- **Human Auditors/Researchers**: Users accessing the environment via the frontend to test models manually or review agent performance.

### 2.4 Operating Environment
- **OS**: Cross-platform (Linux, Windows, macOS) via Docker.
- **Runtime**: Python 3.11+
- **Containerization**: Multi-stage Docker build that first compiles the Vite frontend using Node.js, and then serves both the compiled static assets and the API via the Python FastAPI container. Designed to run seamlessly on a single port (7860) within HuggingFace Spaces.
- **Hardware Resources**: Minimal for deterministic mode; RL startup and inference benefit from GPU but remain functional on CPU with slower initialization/inference.

### 2.5 Design and Implementation Constraints
- Must comply with the OpenEnv specification.
- Must run efficiently within constrained hardware environments (e.g., standard HuggingFace Spaces).
- Inference script relies on an external LLM via an OpenAI-compatible client, requiring proper API key management.
- RL model boot depends on HuggingFace Hub availability and adapter accessibility (`RL_MODEL_ID`); if unavailable, the system must degrade gracefully to deterministic mode.

---

## 3. External Interface Requirements

### 3.1 User Interfaces
- A web-based frontend (built with Vite, Vanilla JS/CSS) allowing users to initiate a "New Audit", specify a HuggingFace repository ID, set the revision branch, and interact with the audit workflow. The UI is served directly by the FastAPI backend on the root (`/`) path and periodically polls the `/api-root` endpoint for server health.

### 3.2 Software Interfaces
- **FastAPI Server**: Exposes RESTful endpoints for environment interaction and serves the frontend.
  - `GET /`: Serves the compiled frontend UI static assets (`index.html`).
  - `GET /api-root`: Returns health and RL readiness metadata (`message`, `rl_agent_loaded`) for frontend connectivity and agent status checks.
  - `POST /reset`: Resets the environment and returns the initial observation.
  - `POST /step`: Processes an action and returns the subsequent observation, reward, and status.
  - `POST /run-audit`: Executes a complete autonomous episode using RL policy when available, otherwise deterministic fallback.
  - `GET /state`: Retrieves the full internal state for debugging.
  - `GET /tasks`: Lists available task IDs (e.g., `basic_completeness`, `technical_consistency`, `regulatory_compliance`).
- **OpenAI API**: The baseline inference agent uses the standard OpenAI client to communicate with underlying LLMs (e.g., `gpt-4o-mini`).
- **HuggingFace Hub API**: Used to download the RL adapter (default `Hasrathussain/audit-agent-rl`), with optional authenticated requests via `HF_TOKEN`.

---

## 4. System Features

### 4.1 Task Modules
The environment consists of three graduated task difficulties:
- **Task 1: Basic Completeness Check (Easy)**: Identify missing required sections from a 10-item checklist. Max 30 steps.
- **Task 2: Technical Consistency Audit (Medium)**: Identify internal inconsistencies and inadequate documentation from a 15-item checklist. Max 45 steps.
- **Task 3: Regulatory Compliance Audit (Hard)**: Audit the model against complex regulatory requirements (EU AI Act, NIST AI RMF) across a 25+ item checklist. Max 60 steps.

### 4.2 Action Space
The agent can perform the following actions:
- `read_section`: Fetch the content of a target section.
- `flag_issue`: Document an issue (type, severity, description).
- `suggest_improvement`: Provide recommendations to fix an issue.
- `verify_claim`: Verify metric/statistical claims.
- `submit_audit`: Terminate the episode and trigger final grading.

### 4.3 Reward System
A multi-dimensional reward function evaluating:
- **Precision**: Valid findings out of total reported findings.
- **Recall**: Valid findings discovered out of total actual issues.
- **Coverage**: Percentage of sections reviewed.
- **Efficiency**: Step optimization.
- **Penalties**: Deductions for false positives and redundant actions.

### 4.4 Autonomous RL Audit Execution
- At service startup, the system attempts to initialize the RL policy (base model + LoRA adapter).
- Autonomous audit episodes are executed through `POST /run-audit`.
- Runtime mode is explicitly returned as either `rl-agent` or `deterministic`.
- If RL initialization fails, audit functionality remains available through deterministic fallback without service interruption.

### 4.5 Training Telemetry and Plot Generation
- The RL training pipeline shall write step-level metrics to `training_log.csv` in a configurable log directory.
- The log must capture phase, step, loss, reward mean, reward standard deviation, and KL divergence where available.
- A plotting utility shall transform the CSV into reward, loss, KL, combined, and baseline-vs-RL charts for reporting.
- When running on Hugging Face Spaces, the log directory may be redirected to persistent storage so that the CSV survives restarts.

---

## 5. Nonfunctional Requirements

### 5.1 Performance Requirements
- The FastAPI server must respond to state transitions (`/step`, `/reset`) within standard web timeout limits (< 1 second processing time).
- Agent evaluation must complete deterministically based on provided action trajectories.
- Startup sequence must complete with either: (a) RL model successfully loaded, or (b) deterministic fallback activated, while keeping API availability intact.

### 5.2 Security & Safety Requirements
- The application processes arbitrary model cards; external data fetched from HuggingFace must be safely parsed.
- No local code execution vulnerabilities during the evaluation of model card contents.
- Secure handling of external API keys via environment variables (`API_KEY`, `API_BASE_URL`).

### 5.3 Software Quality Attributes
- **Maintainability**: Clean project structure separating API, environment logic, and data. Typed data models using Pydantic.
- **Extensibility**: Modular graders and checklists allowing easy addition of new task difficulty levels or regulatory frameworks.
- **Reproducibility**: Pre-generated deterministic ground-truth JSON files for consistent scoring.

---

## 6. Assumptions and Dependencies
- The environment assumes external model cards follow a standard structure that can be successfully mapped to the internal schema.
- The baseline agent heavily relies on the availability and capability of the specified LLM (e.g., via OpenAI API proxy).
- Grading of external HuggingFace model cards is directional, as they lack localized ground-truth annotations compared to internal synthetic data.
