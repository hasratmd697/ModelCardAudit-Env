const EASY_SECTION_SEVERITY = {
  intended_use: "high",
  limitations: "high",
  bias_analysis: "high",
  ethical_considerations: "medium",
  environmental_impact: "low",
  citation: "low",
};

const EASY_SECTION_SUGGESTIONS = {
  intended_use: "Add clear intended use cases, target users, and deployment boundaries.",
  limitations: "Add known failure modes, edge cases, and deployment constraints.",
  bias_analysis: "Add bias analysis with tested groups, metrics, and findings.",
  ethical_considerations: "Document misuse risks and ethical implications of deployment.",
  environmental_impact: "Add training compute cost and estimated carbon footprint.",
  citation: "Add a citation for the model or the base architecture.",
};

const MEDIUM_FINDINGS = {
  "ResNet50-Product-Classifier": [
    {
      section_name: "evaluation_metrics",
      issue_type: "inconsistent",
      severity: "high",
      description: "The headline accuracy claim conflicts with the evaluation table.",
      suggestion: "Report one consistent test accuracy and align the table with the summary claim.",
    },
    {
      section_name: "bias_analysis",
      issue_type: "insufficient",
      severity: "medium",
      description: "Bias analysis is too vague and does not include methodology or quantitative evidence.",
      suggestion: "Document tested groups, fairness metrics, and quantitative results.",
    },
  ],
  "MedNER-Clinical-v3": [
    {
      section_name: "model_description",
      issue_type: "inconsistent",
      severity: "high",
      description: "The architecture description mixes BiLSTM-CRF and Transformer claims.",
      suggestion: "Clarify the actual architecture and remove the contradictory reference.",
    },
    {
      section_name: "evaluation_metrics",
      issue_type: "inconsistent",
      severity: "high",
      description: "The claimed state-of-the-art F1 conflicts with the detailed evaluation table.",
      suggestion: "Use one consistent overall F1 or clearly explain which evaluation each score refers to.",
    },
    {
      section_name: "bias_analysis",
      issue_type: "insufficient",
      severity: "medium",
      description: "Bias claims are unsupported by methodology, groups tested, or metrics.",
      suggestion: "Add subgroup definitions, fairness methodology, and quantitative results.",
    },
    {
      section_name: "training_data",
      issue_type: "insufficient",
      severity: "medium",
      description: "Training data generalizability is unclear because it comes from a single hospital system.",
      suggestion: "Document representativeness limits and likely generalization gaps outside the source hospital.",
    },
  ],
  "TranslateLM-EN-FR": [
    {
      section_name: "model_description",
      issue_type: "inconsistent",
      severity: "high",
      description: "The model architecture is described inconsistently across sections.",
      suggestion: "Align all architecture references so the model is described consistently.",
    },
    {
      section_name: "evaluation_metrics",
      issue_type: "insufficient",
      severity: "medium",
      description: "Evaluation lacks confidence intervals and does not explain the metric discrepancy across test sets.",
      suggestion: "Add uncertainty estimates and clarify the proprietary test set methodology.",
    },
    {
      section_name: "environmental_impact",
      issue_type: "insufficient",
      severity: "low",
      description: "Environmental reporting does not include a carbon footprint estimate.",
      suggestion: "Estimate and report carbon footprint alongside compute usage.",
    },
  ],
};

const HARD_FINDINGS = {
  "Credit-Scoring-Model": [
    {
      section_name: "general",
      issue_type: "missing",
      severity: "critical",
      description: "A formal risk assessment is not documented for this high-risk AI system.",
      regulation: "EU AI Act Article 9",
      suggestion: "Add a documented risk management process with identified risks and mitigations.",
    },
    {
      section_name: "model_description",
      issue_type: "non_compliant",
      severity: "critical",
      description: "The system is described as making autonomous decisions without human oversight.",
      regulation: "EU AI Act Article 14",
      suggestion: "Document human oversight, review, and override procedures.",
    },
    {
      section_name: "training_data",
      issue_type: "insufficient",
      severity: "high",
      description: "Training data provenance and governance details are insufficient.",
      regulation: "EU AI Act Article 10",
      suggestion: "Document provenance, collection, quality checks, and governance controls.",
    },
    {
      section_name: "general",
      issue_type: "missing",
      severity: "high",
      description: "End-user transparency disclosures are not documented.",
      regulation: "EU AI Act Article 13",
      suggestion: "Explain how affected users are informed about AI involvement in decisions.",
    },
  ],
  "ResumeRanker-AI": [
    {
      section_name: "general",
      issue_type: "missing",
      severity: "critical",
      description: "A formal risk assessment is not documented for this high-risk employment system.",
      regulation: "EU AI Act Article 9",
      suggestion: "Document risk identification, evaluation, and mitigation measures.",
    },
    {
      section_name: "model_description",
      issue_type: "non_compliant",
      severity: "critical",
      description: "The system automatically filters candidates without human review.",
      regulation: "EU AI Act Article 14",
      suggestion: "Add mandatory human review and an appeal path for automated filtering decisions.",
    },
    {
      section_name: "bias_analysis",
      issue_type: "insufficient",
      severity: "high",
      description: "Bias analysis omits protected characteristics like age and ethnicity.",
      regulation: "EU AI Act Article 10(2)(f)",
      suggestion: "Expand subgroup analysis to additional protected characteristics and document results.",
    },
    {
      section_name: "training_data",
      issue_type: "insufficient",
      severity: "high",
      description: "Training data representativeness and historical bias risks are not sufficiently documented.",
      regulation: "EU AI Act Article 10(2)",
      suggestion: "Document data bias risks, representativeness limits, and mitigation steps.",
    },
    {
      section_name: "general",
      issue_type: "missing",
      severity: "high",
      description: "Transparency disclosures for job applicants are missing.",
      regulation: "EU AI Act Article 13",
      suggestion: "Document how applicants are informed about AI usage and contestability options.",
    },
  ],
  "DiagAssist-Radiology-v2": [
    {
      section_name: "general",
      issue_type: "missing",
      severity: "critical",
      description: "A formal risk assessment is not documented for this medical AI system.",
      regulation: "EU AI Act Article 9",
      suggestion: "Add a clinical risk assessment covering failure modes and patient safety risks.",
    },
    {
      section_name: "model_description",
      issue_type: "non_compliant",
      severity: "critical",
      description: "High-confidence cases are auto-flagged without documented human oversight.",
      regulation: "EU AI Act Article 14",
      suggestion: "Document human review, confirmation, and override procedures for flagged cases.",
    },
    {
      section_name: "bias_analysis",
      issue_type: "insufficient",
      severity: "high",
      description: "Bias analysis omits racial, ethnic, and gender subgroup reporting.",
      regulation: "EU AI Act Article 10(2)(f)",
      suggestion: "Add disaggregated subgroup analysis across protected characteristics.",
    },
    {
      section_name: "training_data",
      issue_type: "insufficient",
      severity: "high",
      description: "Training data quality and representativeness details are incomplete.",
      regulation: "EU AI Act Article 10",
      suggestion: "Report labeling pipeline quality and representativeness limitations of the source sites.",
    },
    {
      section_name: "evaluation_metrics",
      issue_type: "insufficient",
      severity: "medium",
      description: "Metrics are not reported for all claimed capabilities.",
      regulation: "EU AI Act Article 11",
      suggestion: "Report separate performance metrics for every claimed detection capability.",
    },
  ],
};

export function getAutoAction(observation) {
  if (!observation) {
    return null;
  }

  const unreviewed = observation.available_sections.filter(
    (section) => !observation.sections_reviewed.includes(section),
  );

  if (unreviewed.length > 0) {
    return {
      action_type: "read_section",
      section_name: unreviewed[0],
    };
  }

  const pendingFindings = planFindings(observation).filter((candidate) => {
    return !observation.findings_so_far.some(
      (finding) =>
        finding.section === candidate.section_name &&
        finding.type === candidate.issue_type,
    );
  });

  if (pendingFindings.length > 0) {
    return pendingFindings[0];
  }

  return {
    action_type: "submit_audit",
  };
}

export function planFindings(observation) {
  const taskId = observation.task_id;
  const modelName = observation.model_card_metadata?.model_name || "";

  if (taskId === "basic_completeness") {
    const requiredSections = new Set(observation.checklist.map((item) => item.section));
    const availableSections = new Set(observation.available_sections);
    return [...requiredSections]
      .filter((section) => !availableSections.has(section))
      .sort()
      .map((section) => ({
        action_type: "flag_issue",
        section_name: section,
        issue_type: "missing",
        severity: EASY_SECTION_SEVERITY[section] || "medium",
        description: `The required ${section} section is missing from the model card.`,
        suggestion:
          EASY_SECTION_SUGGESTIONS[section] ||
          `Add a complete ${section} section to the model card.`,
      }));
  }

  if (taskId === "technical_consistency") {
    return (MEDIUM_FINDINGS[modelName] || []).map((finding) => ({
      action_type: "flag_issue",
      ...finding,
    }));
  }

  if (taskId === "regulatory_compliance") {
    return (HARD_FINDINGS[modelName] || []).map((finding) => ({
      action_type: "flag_issue",
      ...finding,
    }));
  }

  return [];
}
