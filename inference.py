"""
ModelCardAudit-Env Baseline Inference Script

Flow:
1. Initialize OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN
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

# Simple .env parser to avoid requiring external libraries
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"\''))

API_BASE_URL = os.environ.get("ENV_API_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# Support OpenRouter standard by allowing fallback to OPEN_API_KEY, using OpenRouter endpoint.
OPENAI_API_KEY = os.environ.get("OPEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

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

def get_system_prompt(task_id: str) -> str:
    """Get a task-specific system prompt."""
    strategy = TASK_STRATEGIES.get(task_id, "")
    return BASE_SYSTEM_PROMPT + strategy

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


def run_task_with_llm(task_id: str, client: OpenAI) -> float:
    """Run a task using the LLM agent."""
    print(f"\n{'='*60}")
    print(f"Starting Task: {task_id}")
    print(f"{'='*60}")
    
    # 1. Reset Environment
    res = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
    if res.status_code != 200:
        print(f"Failed to reset: {res.text}")
        return 0.0
    
    obs = res.json()
    done = False
    # Use task-specific system prompt
    system_prompt = get_system_prompt(task_id)
    messages = [{"role": "system", "content": system_prompt}]
    final_score = 0.0
    
    while not done:
        # 2. Format observation as prompt
        user_msg = format_observation(obs)
        messages.append({"role": "user", "content": user_msg})
        
        # 3. Get LLM action
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            response_text = completion.choices[0].message.content
            if response_text is None:
                print(f"  LLM returned empty response. Falling back to submit_audit.")
                response_text = '{"action_type": "submit_audit"}'
            messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            print(f"  LLM error: {e}. Falling back to submit_audit.")
            response_text = '{"action_type": "submit_audit"}'
        
        # 4. Parse and execute action
        action = parse_action(response_text)
        print(f"  Step {obs['step_count'] + 1}: {action.get('action_type', 'unknown')}", end="")
        if action.get('section_name'):
            print(f" -> {action['section_name']}", end="")
        if action.get('issue_type'):
            print(f" [{action['issue_type']}/{action.get('severity', '?')}]", end="")
        print()
        
        res = requests.post(f"{API_BASE_URL}/step", json=action)
        if res.status_code != 200:
            print(f"  Step error: {res.text}")
            break
            
        step_data = res.json()
        obs = step_data["observation"]
        done = step_data["done"]
        
        if done:
            final_score = step_data["info"].get("score", 0.0)
            print(f"\n  Episode finished. Final Score: {final_score:.4f}")
            print(f"  Reward breakdown: {step_data['reward']}")
    
    return final_score


def run_task_naive(task_id: str) -> float:
    """Run a task using a naive baseline (read all sections then submit)."""
    print(f"\n{'='*60}")
    print(f"Starting Task (naive baseline): {task_id}")
    print(f"{'='*60}")
    
    res = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
    if res.status_code != 200:
        print(f"Failed to reset: {res.text}")
        return 0.0
    
    obs = res.json()
    done = False
    final_score = 0.0
    
    while not done:
        unreviewed = [s for s in obs['available_sections'] if s not in obs['sections_reviewed']]
        
        if unreviewed:
            action = {"action_type": "read_section", "section_name": unreviewed[0]}
        else:
            action = {"action_type": "submit_audit"}
        
        print(f"  Step {obs['step_count'] + 1}: {action['action_type']}", end="")
        if action.get('section_name'):
            print(f" -> {action['section_name']}", end="")
        print()
        
        res = requests.post(f"{API_BASE_URL}/step", json=action)
        step_data = res.json()
        obs = step_data["observation"]
        done = step_data["done"]
        
        if done:
            final_score = step_data["info"].get("score", 0.0)
            print(f"\n  Episode finished. Final Score: {final_score:.4f}")
    
    return final_score


if __name__ == "__main__":
    tasks = ["basic_completeness", "technical_consistency", "regulatory_compliance"]
    
    print("Checking server health...")
    try:
        r = requests.get(f"{API_BASE_URL}/")
        r.raise_for_status()
        print("Server is running.\n")
    except Exception:
        print("Server is not running. Please start it with:")
        print("  uvicorn server.app:app --host 0.0.0.0 --port 7860")
        exit(1)
    
    # Decide mode based on API key availability
    use_llm = OPENAI_API_KEY is not None
    
    if use_llm:
        print(f"Running LLM-based agent (model: {MODEL_NAME})")
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**client_kwargs)
    else:
        print("OPENAI_API_KEY not set. Running naive baseline (read-all-then-submit).")
        print("Set OPENAI_API_KEY to enable LLM-driven auditing.\n")
        client = None
    
    results = {}
    for task in tasks:
        if use_llm:
            score = run_task_with_llm(task, client)
        else:
            score = run_task_naive(task)
        results[task] = score
        time.sleep(0.5)
    
    # Print results table
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
