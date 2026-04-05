from typing import List, Dict, Any
from .models import Finding, Reward

def compute_matches(findings: List[Finding], ground_truth: List[Dict[str, Any]], fuzzy_general: bool = False) -> int:
    true_positives = 0
    gt_issues = ground_truth.copy()
    
    for f in findings:
        matched = False
        for gt in gt_issues:
            gt_section = gt.get("section")
            gt_type = gt.get("type")
            # Standard match: same section AND same type
            if f.section == gt_section and f.type == gt_type:
                matched = True
                gt_issues.remove(gt)
                break
            # Fuzzy match for regulatory hard task:
            # Ground truth issues tagged 'general' don't map to a real card section.
            # Accept any finding with matching type against an unmatched 'general' GT issue.
            if fuzzy_general and gt_section == "general" and f.type == gt_type:
                matched = True
                gt_issues.remove(gt)
                break
        if matched:
            true_positives += 1
            
    return true_positives

def compute_reward(findings: List[Finding], ground_truth: List[Dict[str, Any]], steps_taken: int, max_steps: int, sections_reviewed: List[str], total_sections: int, action_history: List[str]) -> Reward:
    
    true_positives = compute_matches(findings, ground_truth)
            
    num_findings = max(len(findings), 1)
    precision = true_positives / num_findings
    
    # 2. Recall
    num_gt = max(len(ground_truth), 1)
    recall = true_positives / num_gt
    
    # 3. Coverage
    coverage = len(sections_reviewed) / total_sections if total_sections > 0 else 0
    
    # 4. Efficiency
    efficiency = max(0.0, 1.0 - (steps_taken / max_steps))
    
    # 5. Penalties
    false_positives = len(findings) - true_positives
    false_positive_penalty = -0.05 * false_positives
    
    # Progress
    progress = 0.1 * coverage + 0.3 * recall
    
    # Repetition Penalty (basic)
    read_actions = [a for a in action_history if "read_section" in a]
    repeated_reads = len(read_actions) - len(set(read_actions))
    repetition_penalty = -0.02 * repeated_reads
    
    total = (0.35 * precision + 0.35 * recall + 0.15 * coverage + 0.10 * efficiency 
             + 0.05 * progress + false_positive_penalty + repetition_penalty)
             
    return Reward(
        total=max(0.0, min(1.0, total)),
        precision_score=precision,
        recall_score=recall,
        coverage_score=coverage,
        efficiency_bonus=efficiency,
        false_positive_penalty=false_positive_penalty
    )
