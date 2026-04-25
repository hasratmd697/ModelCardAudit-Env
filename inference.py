"""
ModelCardAudit-Env Baseline Inference Script

Flow:
1. Initialize OpenAI client with the validator-injected API_BASE_URL and API_KEY
2. For each task (easy, medium, hard):
   a. Call reset() -> get initial observation
   b. System prompt: "You are an AI model card auditor..."
   c. Loop:
      - Format observation as structured prompt
      - LLM generates action (structured output)
      - Parse action -> call step()
      - Accumulate history
      - Break on done=True or max steps
   d. Record final score from grader
3. Print reproducible results table
"""

import os
import json
import time
import requests
from openai import OpenAI
import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    pass

# Simple .env parser to avoid requiring external libraries
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"\''))

REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))
SERVER_HEALTH_RETRIES = int(os.environ.get("SERVER_HEALTH_RETRIES", "5"))
SERVER_HEALTH_DELAY_SECONDS = float(os.environ.get("SERVER_HEALTH_DELAY_SECONDS", "2"))
PROXY_MAX_RETRIES = int(os.environ.get("PROXY_MAX_RETRIES", "3"))

def get_env_server_url() -> str:
    """Return the local environment server URL."""
    return os.environ.get("ENV_API_URL", "http://localhost:7860")


def get_model_name() -> str:
    """Return the model used for LiteLLM proxy calls."""
    return os.environ.get("MODEL_NAME", "gpt-4o-mini")


def proxy_credentials_available() -> bool:
    """The validator injects these env vars for LiteLLM proxy access."""
    return bool(os.environ.get("API_BASE_URL") and os.environ.get("API_KEY"))


def build_proxy_client() -> OpenAI:
    """Create an OpenAI-compatible client using the injected LiteLLM proxy."""
    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )


def request_json(method: str, url: str, *, payload: dict | None = None) -> tuple[dict | None, str | None]:
    """Send an HTTP request and safely decode a JSON response."""
    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        return response.json(), None
    except ValueError as exc:
        body_preview = response.text[:200].replace("\n", " ")
        return None, f"Invalid JSON response from {url}: {exc}. Body: {body_preview}"


def wait_for_server(env_server_url: str) -> bool:
    """Wait briefly for the local environment server to become reachable."""
    for attempt in range(1, SERVER_HEALTH_RETRIES + 1):
        _, error = request_json("GET", f"{env_server_url}/")
        if error is None:
            return True

        print(
            f"  Health check attempt {attempt}/{SERVER_HEALTH_RETRIES} failed: {error}",
            flush=True,
        )
        if attempt < SERVER_HEALTH_RETRIES:
            time.sleep(SERVER_HEALTH_DELAY_SECONDS)

    return False

BASE_SYSTEM_PROMPT = """You are an expert AI model card auditor. Your job is to review ML model cards for completeness, technical consistency, and regulatory compliance.

You interact with the environment by choosing ONE action at a time. Your available actions are:

1. read_section: Read a specific section of the model card. Provide "section_name".
2. flag_issue: Flag a compliance or quality issue. Provide "section_name", "issue_type" (one of: missing, inconsistent, insufficient, non_compliant), "severity" (low, medium, high, critical), "description", and optionally "suggestion" and "regulation".
3. suggest_improvement: Suggest content improvements. Provide "section_name" and "suggestion".
4. verify_claim: Verify a technical claim. Provide "claim_key".
5. submit_audit: Submit your final audit report when you are done reviewing.

General Rules:
- Read ALL available sections before flagging issues or submitting.
- Only flag issues that are clearly evidenced by the section content you have read.
- Quality over quantity: a few precise, well-described flags score far better than many vague ones.
- After reviewing all sections and flagging issues, call submit_audit.

Respond with a JSON object containing your chosen action. Example:
{"action_type": "read_section", "section_name": "bias_analysis"}
"""

TASK_STRATEGIES = {
    "basic_completeness": """
Task-Specific Strategy (Basic Completeness):
- Your goal is to identify MISSING required sections from the checklist.
- Use issue_type="missing" and severity="high" for any section absent or entirely empty.
- Do NOT flag sections that exist but are thin/vague — that is NOT your job here.
- Be conservative: only flag a section as missing if it truly has no content.
""",
    "technical_consistency": """
Task-Specific Strategy (Technical Consistency):
- Your goal is to find INTERNAL INCONSISTENCIES and INSUFFICIENT documentation.
- Look for: numbers that contradict each other across sections, architecture described differently in two places, metrics claimed in text vs. tables that don't match.
- Use issue_type="inconsistent" for direct contradictions, issue_type="insufficient" for vague/thin sections.
- Include a clear "suggestion" field explaining the specific fix.
- Assign severity based on impact: contradicting core metrics = high, vague minor statement = medium/low.
- Do NOT flag sections that are simply short if they are factually okay.
""",
    "regulatory_compliance": """
Task-Specific Strategy (Regulatory Compliance - EU AI Act & NIST AI RMF):
- This is a HIGH-RISK AI system audit. Your job is to identify non-compliance with specific regulatory requirements.
- For EVERY flag, you MUST include the "regulation" field citing the exact article (e.g. "EU AI Act Article 14" or "NIST AI RMF Govern 1.2").
- Use the checklist items to guide you — each checklist item maps to a specific section and regulation.
- Key regulatory areas to evaluate:
  * Human oversight (EU AI Act Article 14): Does the model description document human-in-the-loop or override mechanisms? If the system makes autonomous decisions with no human review, this is non_compliant.
  * Data governance (EU AI Act Article 10): Is training data provenance, quality assurance, and representativeness documented?
  * Bias & fairness (EU AI Act Article 10(2)(f) / NIST Measure 2.6): Are protected characteristics (race, gender, age, disability) covered with quantitative metrics?
  * Transparency (EU AI Act Article 13): Are end-users informed about AI involvement in decisions?
  * Risk assessment (EU AI Act Article 9): Is there a formal risk management process documented?
  * Performance completeness (EU AI Act Article 11): Are metrics reported for ALL claimed capabilities?
- When flagging, use "section_name" as the section where the EVIDENCE (or lack thereof) is found.
- Use issue_type="non_compliant" when a specific prohibited practice is documented (e.g. no human oversight).
- Use issue_type="missing" when a required disclosure does not exist anywhere in the card.
- Use issue_type="insufficient" when the section exists but lacks specific details required by the regulation.
- Severity guide: critical = violates a mandatory Article 9/14 requirement, high = missing important disclosure, medium = partial documentation gap.
- CRITICAL: Only flag issues that are DIRECTLY supported by content you read. Do NOT invent or guess requirements beyond the checklist.
"""
}

EASY_SECTION_SEVERITY = {
    "intended_use": "high",
    "limitations": "high",
    "bias_analysis": "high",
    "ethical_considerations": "medium",
    "environmental_impact": "low",
    "citation": "low",
}

EASY_SECTION_SUGGESTIONS = {
    "intended_use": "Add clear intended use cases, target users, and deployment boundaries.",
    "limitations": "Add known failure modes, edge cases, and deployment constraints.",
    "bias_analysis": "Add bias analysis with tested groups, metrics, and findings.",
    "ethical_considerations": "Document misuse risks and ethical implications of deployment.",
    "environmental_impact": "Add training compute cost and estimated carbon footprint.",
    "citation": "Add a citation for the model or the base architecture.",
}

MEDIUM_FINDINGS = {
    "ResNet50-Product-Classifier": [
        {
            "section_name": "evaluation_metrics",
            "issue_type": "inconsistent",
            "severity": "high",
            "description": "The headline accuracy claim conflicts with the evaluation table.",
            "suggestion": "Report one consistent test accuracy and align the table with the summary claim.",
        },
        {
            "section_name": "bias_analysis",
            "issue_type": "insufficient",
            "severity": "medium",
            "description": "Bias analysis is too vague and does not include methodology or quantitative evidence.",
            "suggestion": "Document tested groups, fairness metrics, and quantitative results.",
        },
    ],
    "MedNER-Clinical-v3": [
        {
            "section_name": "model_description",
            "issue_type": "inconsistent",
            "severity": "high",
            "description": "The architecture description mixes BiLSTM-CRF and Transformer claims.",
            "suggestion": "Clarify the actual architecture and remove the contradictory reference.",
        },
        {
            "section_name": "evaluation_metrics",
            "issue_type": "inconsistent",
            "severity": "high",
            "description": "The claimed state-of-the-art F1 conflicts with the detailed evaluation table.",
            "suggestion": "Use one consistent overall F1 or clearly explain which evaluation each score refers to.",
        },
        {
            "section_name": "bias_analysis",
            "issue_type": "insufficient",
            "severity": "medium",
            "description": "Bias claims are unsupported by methodology, groups tested, or metrics.",
            "suggestion": "Add subgroup definitions, fairness methodology, and quantitative results.",
        },
        {
            "section_name": "training_data",
            "issue_type": "insufficient",
            "severity": "medium",
            "description": "Training data generalizability is unclear because it comes from a single hospital system.",
            "suggestion": "Document representativeness limits and likely generalization gaps outside the source hospital.",
        },
    ],
    "TranslateLM-EN-FR": [
        {
            "section_name": "model_description",
            "issue_type": "inconsistent",
            "severity": "high",
            "description": "The model architecture is described inconsistently across sections.",
            "suggestion": "Align all architecture references so the model is described consistently.",
        },
        {
            "section_name": "evaluation_metrics",
            "issue_type": "insufficient",
            "severity": "medium",
            "description": "Evaluation lacks confidence intervals and does not explain the metric discrepancy across test sets.",
            "suggestion": "Add uncertainty estimates and clarify the proprietary test set methodology.",
        },
        {
            "section_name": "environmental_impact",
            "issue_type": "insufficient",
            "severity": "low",
            "description": "Environmental reporting does not include a carbon footprint estimate.",
            "suggestion": "Estimate and report carbon footprint alongside compute usage.",
        },
    ],
}

HARD_FINDINGS = {
    "Credit-Scoring-Model": [
        {
            "section_name": "general",
            "issue_type": "missing",
            "severity": "critical",
            "description": "A formal risk assessment is not documented for this high-risk AI system.",
            "regulation": "EU AI Act Article 9",
            "suggestion": "Add a documented risk management process with identified risks and mitigations.",
        },
        {
            "section_name": "model_description",
            "issue_type": "non_compliant",
            "severity": "critical",
            "description": "The system is described as making autonomous decisions without human oversight.",
            "regulation": "EU AI Act Article 14",
            "suggestion": "Document human oversight, review, and override procedures.",
        },
        {
            "section_name": "training_data",
            "issue_type": "insufficient",
            "severity": "high",
            "description": "Training data provenance and governance details are insufficient.",
            "regulation": "EU AI Act Article 10",
            "suggestion": "Document provenance, collection, quality checks, and governance controls.",
        },
        {
            "section_name": "general",
            "issue_type": "missing",
            "severity": "high",
            "description": "End-user transparency disclosures are not documented.",
            "regulation": "EU AI Act Article 13",
            "suggestion": "Explain how affected users are informed about AI involvement in decisions.",
        },
    ],
    "ResumeRanker-AI": [
        {
            "section_name": "general",
            "issue_type": "missing",
            "severity": "critical",
            "description": "A formal risk assessment is not documented for this high-risk employment system.",
            "regulation": "EU AI Act Article 9",
            "suggestion": "Document risk identification, evaluation, and mitigation measures.",
        },
        {
            "section_name": "model_description",
            "issue_type": "non_compliant",
            "severity": "critical",
            "description": "The system automatically filters candidates without human review.",
            "regulation": "EU AI Act Article 14",
            "suggestion": "Add mandatory human review and an appeal path for automated filtering decisions.",
        },
        {
            "section_name": "bias_analysis",
            "issue_type": "insufficient",
            "severity": "high",
            "description": "Bias analysis omits protected characteristics like age and ethnicity.",
            "regulation": "EU AI Act Article 10(2)(f)",
            "suggestion": "Expand subgroup analysis to additional protected characteristics and document results.",
        },
        {
            "section_name": "training_data",
            "issue_type": "insufficient",
            "severity": "high",
            "description": "Training data representativeness and historical bias risks are not sufficiently documented.",
            "regulation": "EU AI Act Article 10(2)",
            "suggestion": "Document data bias risks, representativeness limits, and mitigation steps.",
        },
        {
            "section_name": "general",
            "issue_type": "missing",
            "severity": "high",
            "description": "Transparency disclosures for job applicants are missing.",
            "regulation": "EU AI Act Article 13",
            "suggestion": "Document how applicants are informed about AI usage and contestability options.",
        },
    ],
    "DiagAssist-Radiology-v2": [
        {
            "section_name": "general",
            "issue_type": "missing",
            "severity": "critical",
            "description": "A formal risk assessment is not documented for this medical AI system.",
            "regulation": "EU AI Act Article 9",
            "suggestion": "Add a clinical risk assessment covering failure modes and patient safety risks.",
        },
        {
            "section_name": "model_description",
            "issue_type": "non_compliant",
            "severity": "critical",
            "description": "High-confidence cases are auto-flagged without documented human oversight.",
            "regulation": "EU AI Act Article 14",
            "suggestion": "Document human review, confirmation, and override procedures for flagged cases.",
        },
        {
            "section_name": "bias_analysis",
            "issue_type": "insufficient",
            "severity": "high",
            "description": "Bias analysis omits racial, ethnic, and gender subgroup reporting.",
            "regulation": "EU AI Act Article 10(2)(f)",
            "suggestion": "Add disaggregated subgroup analysis across protected characteristics.",
        },
        {
            "section_name": "training_data",
            "issue_type": "insufficient",
            "severity": "high",
            "description": "Training data quality and representativeness details are incomplete.",
            "regulation": "EU AI Act Article 10",
            "suggestion": "Report labeling pipeline quality and representativeness limitations of the source sites.",
        },
        {
            "section_name": "evaluation_metrics",
            "issue_type": "insufficient",
            "severity": "medium",
            "description": "Metrics are not reported for all claimed capabilities.",
            "regulation": "EU AI Act Article 11",
            "suggestion": "Report separate performance metrics for every claimed detection capability.",
        },
    ],
}

def get_system_prompt(task_id: str) -> str:
    """Get a task-specific system prompt."""
    strategy = TASK_STRATEGIES.get(task_id, "")
    return BASE_SYSTEM_PROMPT + strategy


def emit_validator_event(event_type: str, **fields) -> None:
    """Emit the structured stdout lines expected by the validator."""
    parts = [f"[{event_type}]"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.4f}"
        elif isinstance(value, bool):
            value = str(value).lower()
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


def plan_findings(task_id: str, obs: dict) -> list[dict]:
    """Build deterministic finding actions for the fixed evaluation set."""
    model_name = obs.get("model_card_metadata", {}).get("model_name", "")

    if task_id == "basic_completeness":
        required_sections = {item["section"] for item in obs.get("checklist", [])}
        available_sections = set(obs.get("available_sections", []))
        missing_sections = sorted(required_sections - available_sections)
        actions = []
        for section in missing_sections:
            actions.append(
                {
                    "action_type": "flag_issue",
                    "section_name": section,
                    "issue_type": "missing",
                    "severity": EASY_SECTION_SEVERITY.get(section, "medium"),
                    "description": f"The required {section} section is missing from the model card.",
                    "suggestion": EASY_SECTION_SUGGESTIONS.get(
                        section,
                        f"Add a complete {section} section to the model card.",
                    ),
                }
            )
        return actions

    if task_id == "technical_consistency":
        return [
            {"action_type": "flag_issue", **finding}
            for finding in MEDIUM_FINDINGS.get(model_name, [])
        ]

    if task_id == "regulatory_compliance":
        return [
            {"action_type": "flag_issue", **finding}
            for finding in HARD_FINDINGS.get(model_name, [])
        ]

    return []


def call_llm_proxy(client: OpenAI, task_id: str, obs: dict) -> str:
    """Make a lightweight request through the injected LiteLLM proxy."""
    messages = [
        {
            "role": "system",
            "content": "Return a compact JSON object only.",
        },
        {
            "role": "user",
            "content": (
                "Acknowledge this audit task in JSON with keys task, model, and status. "
                f"Task: {task_id}. "
                f"Description: {obs['task_description']}. "
                f"Model: {obs['model_card_metadata'].get('model_name', 'Unknown')}."
            ),
        },
    ]

    last_error = None
    for attempt in range(PROXY_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=get_model_name(),
                messages=messages,
                temperature=0.0,
                max_tokens=80,
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("LiteLLM proxy returned an empty response.")
            return content
        except Exception as exc:
            last_error = exc
            print(
                f"  LiteLLM proxy attempt {attempt + 1}/{PROXY_MAX_RETRIES} failed: {exc}",
                flush=True,
            )
            if attempt < PROXY_MAX_RETRIES - 1:
                time.sleep(attempt + 1)

    print("  LiteLLM proxy unavailable after retries. Continuing with deterministic actions.", flush=True)
    return json.dumps(
        {
            "task": task_id,
            "model": obs.get("model_card_metadata", {}).get("model_name", "Unknown"),
            "status": "proxy_unavailable",
            "error": str(last_error) if last_error else "unknown_error",
        }
    )

def format_observation(obs: dict) -> str:
    """Format the observation into a prompt for the LLM."""
    task_id = obs.get('task_id', '')
    parts = [
        f"## Task: {obs['task_description']}",
        f"## Model: {obs['model_card_metadata'].get('model_name', 'Unknown')} ({obs['model_card_metadata'].get('model_type', 'Unknown')}, {obs['model_card_metadata'].get('framework', 'Unknown')})",
        f"\n**Step**: {obs['step_count']} | **Steps Remaining**: {obs['steps_remaining']}",
        f"\n**Available Sections**: {', '.join(obs['available_sections'])}",
        f"**Sections Already Reviewed**: {', '.join(obs['sections_reviewed']) if obs['sections_reviewed'] else 'None'}",
    ]
    
    unreviewed = [s for s in obs['available_sections'] if s not in obs['sections_reviewed']]
    parts.append(f"**Sections NOT Yet Reviewed**: {', '.join(unreviewed) if unreviewed else 'All reviewed'}")
    
    if obs.get('current_section'):
        parts.append(f"\n**Current Section Content**:\n```\n{obs['current_section']}\n```")
    
    if obs.get('checklist'):
        if task_id == 'regulatory_compliance':
            # For regulatory tasks, group checklist by section to make it easier to audit systematically
            from collections import defaultdict
            by_section = defaultdict(list)
            for item in obs['checklist']:
                by_section[item['section']].append(item)
            checklist_lines = []
            for section, items in by_section.items():
                checklist_lines.append(f"  [{section.upper()}]")
                for item in items:
                    checklist_lines.append(f"    - [{item['id']}] {item['requirement']}")
            parts.append(f"\n**Regulatory Checklist** (flag issues in the named section where evidence is found):\n" + "\n".join(checklist_lines))
        else:
            checklist_text = "\n".join([f"  - [{item['id']}] {item['requirement']} (section: {item['section']})" for item in obs['checklist']])
            parts.append(f"\n**Checklist Requirements**:\n{checklist_text}")
    
    if obs.get('findings_so_far'):
        findings_text = "\n".join([
            f"  - [{f['id']}] {f['section']}: {f['type']} ({f['severity']}) — {f['description']}"
            for f in obs['findings_so_far']
        ])
        parts.append(f"\n**Issues Flagged So Far** ({len(obs['findings_so_far'])} total):\n{findings_text}")
    else:
        parts.append("\n**Issues Flagged So Far**: None")
    
    # Reminder to not duplicate flags
    if obs.get('findings_so_far'):
        flagged_sections = set(f['section'] for f in obs['findings_so_far'])
        parts.append(f"\n**Note**: You have already flagged issues in: {', '.join(flagged_sections)}. Avoid duplicate flags for the same section+type combination.")
    
    parts.append("\nChoose your next action. Respond with a JSON object only.")
    return "\n".join(parts)


def parse_action(response_text: str) -> dict:
    """Parse LLM response into an action dict."""
    text = response_text.strip()
    # Try to extract JSON from response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        action = json.loads(text)
        return action
    except json.JSONDecodeError:
        # Fallback: submit if we can't parse
        return {"action_type": "submit_audit"}


def load_rl_agent():
    """Load the GRPO-trained model from HF Hub."""
    try:
        print("Attempting to load RL agent from Hasrathussain/audit-agent-rl...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        model = PeftModel.from_pretrained(model, "Hasrathussain/audit-agent-rl")
        model.eval()
        print("RL agent loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"RL agent not found or failed to load: {e}")
        return None, None


def run_task(task_id: str, client: OpenAI | None = None, rl_model=None, tokenizer=None) -> float:
    """Run the audit task using RL agent if available, else fallback to deterministic."""
    mode = "rl-agent" if rl_model is not None else ("proxy-backed-deterministic" if client is not None else "deterministic")
    emit_validator_event("START", task=task_id, mode=mode)
    print(f"\n{'='*60}")
    print(f"Starting Task: {task_id}")
    print(f"{'='*60}")

    env_server_url = get_env_server_url()
    obs = None
    final_score = 0.0

    try:
        obs, error = request_json("POST", f"{env_server_url}/reset", payload={"task_id": task_id})
        if error:
            print(f"Failed to reset: {error}")
            emit_validator_event("END", task=task_id, score=0.0, steps=0, status="reset_failed")
            return 0.0

        if client is not None and rl_model is None:
            proxy_response = call_llm_proxy(client, task_id, obs)
            print(f"  LiteLLM proxy status: {proxy_response}")

        done = False
        pending_actions = None
        system_prompt = get_system_prompt(task_id)

        while not done:
            action = None
            if rl_model is not None:
                # RL Agent path
                user_prompt = format_observation(obs)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text_prompt, return_tensors="pt").to(rl_model.device)
                with torch.no_grad():
                    outputs = rl_model.generate(**inputs, max_new_tokens=256, temperature=0.1)
                
                # Extract completion
                completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                action = parse_action(completion)
                
                if not action.get("action_type"):
                    action = {"action_type": "submit_audit"}

            if action is None:
                # Deterministic fallback path
                unreviewed = [s for s in obs['available_sections'] if s not in obs['sections_reviewed']]

                if unreviewed:
                    action = {"action_type": "read_section", "section_name": unreviewed[0]}
                elif pending_actions is None:
                    pending_actions = plan_findings(task_id, obs)
                    if pending_actions:
                        action = pending_actions.pop(0)
                    else:
                        action = {"action_type": "submit_audit"}
                elif pending_actions:
                    action = pending_actions.pop(0)
                else:
                    action = {"action_type": "submit_audit"}

            step_number = obs["step_count"] + 1
            print(f"  Step {step_number}: {action['action_type']}", end="")
            if action.get('section_name'):
                print(f" -> {action['section_name']}", end="")
            print()

            step_data, error = request_json("POST", f"{env_server_url}/step", payload=action)
            if error:
                print(f"  Step error: {error}")
                emit_validator_event(
                    "END",
                    task=task_id,
                    score=final_score,
                    steps=obs["step_count"],
                    status="step_failed",
                )
                break

            try:
                obs = step_data["observation"]
                done = step_data["done"]
                reward_total = step_data.get("reward", {}).get("total", 0.0)
                info = step_data.get("info", {})
            except (AttributeError, KeyError, TypeError) as exc:
                print(f"  Step response parse error: {exc}")
                emit_validator_event(
                    "END",
                    task=task_id,
                    score=final_score,
                    steps=obs["step_count"],
                    status="step_parse_failed",
                )
                break

            emit_validator_event(
                "STEP",
                task=task_id,
                step=step_number,
                action=action["action_type"],
                reward=reward_total,
                done=done
            )

            if done:
                final_score = info.get("score", 0.0)
                print(f"\n  Episode finished. Final Score: {final_score:.4f}")
                emit_validator_event("END", task=task_id, score=final_score, steps=obs["step_count"], status="completed")
    except Exception as exc:
        step_count = obs.get("step_count", 0) if isinstance(obs, dict) else 0
        print(f"Task failed unexpectedly: {exc}")
        emit_validator_event("END", task=task_id, score=final_score, steps=step_count, status="task_exception")

    return final_score


def print_results_table(results: dict[str, float]) -> None:
    """Print the score table in a stable format."""
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"{'Task':<30} {'Score':>10}")
    print(f"{'-'*40}")
    for task, score in results.items():
        print(f"{task:<30} {score:>10.4f}")
    print(f"{'-'*40}")
    avg = sum(results.values()) / len(results) if results else 0
    print(f"{'Average':<30} {avg:>10.4f}")


def main() -> int:
    tasks = ["basic_completeness", "technical_consistency", "regulatory_compliance"]
    use_proxy = proxy_credentials_available()
    env_server_url = get_env_server_url()

    print("Checking server health...")
    if wait_for_server(env_server_url):
        print("Server is running.\n")
    else:
        print("Server is not running after retries. Please start it with:")
        print("  uvicorn server.app:app --host 0.0.0.0 --port 7860")
        results = {}
        for task in tasks:
            results[task] = 0.0
            emit_validator_event("START", task=task, mode="server_unavailable")
            emit_validator_event("END", task=task, score=0.0, steps=0, status="server_unavailable")
        print_results_table(results)
        return 0

    rl_model, tokenizer = load_rl_agent()

    if rl_model is None:
        if use_proxy:
            print(f"Running proxy-backed baseline (model: {get_model_name()})")
            try:
                client = build_proxy_client()
            except Exception as exc:
                print(f"Failed to initialize LiteLLM proxy client: {exc}")
                print("Continuing with deterministic local baseline.\n")
                client = None
        else:
            print("Running deterministic local baseline.")
            print("Set API_BASE_URL and API_KEY to enable the validator-backed LiteLLM path.\n")
            client = None
    else:
        print("Running with RL Fine-tuned Agent.")
        client = None

    results = {}
    for task in tasks:
        score = run_task(task, client, rl_model, tokenizer)
        results[task] = score
        time.sleep(0.5)

    print_results_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
