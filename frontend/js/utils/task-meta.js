export const TASK_ORDER = [
  "basic_completeness",
  "technical_consistency",
  "regulatory_compliance",
];

export const TASK_DEFINITIONS = {
  basic_completeness: {
    id: "basic_completeness",
    label: "Basic Completeness",
    difficulty: "Easy",
    difficultyLevel: 1,
    maxSteps: 30,
    checklistItems: 10,
    description: "Identify missing required sections from a baseline model card checklist.",
    modeCopy: "Best for quick structural audits and hackathon demos.",
    scoring: [
      { label: "Recall", value: "50%" },
      { label: "Precision", value: "30%" },
      { label: "Coverage", value: "20%" },
    ],
  },
  technical_consistency: {
    id: "technical_consistency",
    label: "Technical Consistency",
    difficulty: "Medium",
    difficultyLevel: 2,
    maxSteps: 45,
    checklistItems: 15,
    description: "Find internal inconsistencies, thin documentation, and unsupported claims.",
    modeCopy: "Best for spotting contradictions and evidence gaps across sections.",
    scoring: [
      { label: "Recall", value: "40%" },
      { label: "Precision", value: "30%" },
      { label: "Suggestion quality", value: "15%" },
      { label: "Severity accuracy", value: "15%" },
    ],
  },
  regulatory_compliance: {
    id: "regulatory_compliance",
    label: "Regulatory Compliance",
    difficulty: "Hard",
    difficultyLevel: 3,
    maxSteps: 60,
    checklistItems: 26,
    description: "Run a full EU AI Act and NIST AI RMF documentation review.",
    modeCopy: "Best for high-risk-system walkthroughs and policy-heavy demos.",
    scoring: [
      { label: "Recall", value: "35%" },
      { label: "Precision", value: "25%" },
      { label: "Severity accuracy", value: "15%" },
      { label: "Regulatory mapping", value: "15%" },
      { label: "Efficiency", value: "10%" },
    ],
  },
};

export function getTaskDefinition(taskId) {
  return TASK_DEFINITIONS[taskId] || TASK_DEFINITIONS.basic_completeness;
}
