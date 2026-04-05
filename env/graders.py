from typing import List, Dict, Any
from .models import Finding
from .reward import compute_matches

def grade_easy_task(findings: List[Finding], ground_truth: List[Dict[str, Any]], sections_reviewed: List[str], total_sections: int) -> float:
    """
    Task 1: Basic Completeness Check (Easy)
    score = 0.5 * recall + 0.3 * precision + 0.2 * coverage
    """
    true_positives = compute_matches(findings, ground_truth)
    
    num_findings = max(len(findings), 1)
    precision = true_positives / num_findings
    
    num_gt = max(len(ground_truth), 1)
    recall = true_positives / num_gt
    
    coverage = len(sections_reviewed) / total_sections if total_sections > 0 else 0
    
    score = 0.5 * recall + 0.3 * precision + 0.2 * coverage
    return max(0.0, min(1.0, score))

def grade_medium_task(findings: List[Finding], ground_truth: List[Dict[str, Any]]) -> float:
    """
    Task 2: Technical Consistency Audit (Medium)
    score = 0.4 * recall + 0.3 * precision + 0.15 * suggestion_quality + 0.15 * severity_accuracy
    """
    true_positives = compute_matches(findings, ground_truth)
    
    num_findings = max(len(findings), 1)
    precision = true_positives / num_findings
    
    num_gt = max(len(ground_truth), 1)
    recall = true_positives / num_gt
    
    severity_correct = 0
    gt_issues = ground_truth.copy()
    for f in findings:
        for gt in gt_issues:
            if f.section == gt.get("section") and f.type == gt.get("type"):
                if f.severity == gt.get("severity"):
                    severity_correct += 1
                gt_issues.remove(gt)
                break
                
    severity_accuracy = severity_correct / num_findings if num_findings > 0 else 0
    
    suggestion_quality = sum([1 for f in findings if f.suggested_fix]) / num_findings if num_findings > 0 else 0
    
    score = 0.4 * recall + 0.3 * precision + 0.15 * suggestion_quality + 0.15 * severity_accuracy
    return max(0.0, min(1.0, score))

def grade_hard_task(findings: List[Finding], ground_truth: List[Dict[str, Any]], steps_taken: int, max_steps: int) -> float:
    """
    Task 3: Regulatory Compliance Audit (Hard)
    score = 0.35 * recall + 0.25 * precision + 0.15 * severity_accuracy + 0.15 * regulatory_mapping + 0.10 * efficiency
    """
    # Use fuzzy_general so 'general'-section GT issues can be matched by section-specific findings of the same type
    true_positives = compute_matches(findings, ground_truth, fuzzy_general=True)
    
    num_findings = max(len(findings), 1)
    precision = true_positives / num_findings
    
    num_gt = max(len(ground_truth), 1)
    recall = true_positives / num_gt
    
    severity_correct = 0
    regulatory_correct = 0
    
    gt_issues = ground_truth.copy()
    for f in findings:
        for gt in gt_issues:
            gt_section = gt.get("section")
            gt_type = gt.get("type")
            # Match by section+type OR by type when GT section is 'general'
            is_match = (f.section == gt_section and f.type == gt_type) or \
                       (gt_section == "general" and f.type == gt_type)
            if is_match:
                if f.severity == gt.get("severity"):
                    severity_correct += 1
                if f.regulation and gt.get("regulation") and f.regulation in gt.get("regulation"):
                    regulatory_correct += 1
                gt_issues.remove(gt)
                break
                
    severity_accuracy = severity_correct / num_findings if num_findings > 0 else 0
    regulatory_mapping = regulatory_correct / num_findings if num_findings > 0 else 0
    
    efficiency = max(0.0, 1.0 - (steps_taken / max_steps))
    
    score = 0.35 * recall + 0.25 * precision + 0.15 * severity_accuracy + 0.15 * regulatory_mapping + 0.10 * efficiency
    return max(0.0, min(1.0, score))

def grade_task(task_id: str, findings: List[Finding], ground_truth: List[Dict[str, Any]], sections_reviewed: List[str], total_sections: int, steps_taken: int, max_steps: int) -> float:
    if task_id == "basic_completeness":
        return grade_easy_task(findings, ground_truth, sections_reviewed, total_sections)
    elif task_id == "technical_consistency":
        return grade_medium_task(findings, ground_truth)
    elif task_id == "regulatory_compliance":
        return grade_hard_task(findings, ground_truth, steps_taken, max_steps)
    else:
        return 0.0
