import { renderChecklist } from "../components/checklist.js";
import { renderFindingCard } from "../components/finding-card.js";
import { renderFlagForm } from "../components/flag-form.js";
import { renderProgressBar } from "../components/progress-bar.js";
import { renderSectionViewer } from "../components/section-viewer.js";
import { createBadge, createButton, createElement, createEmptyState } from "../utils/dom.js";
import { formatPercent, formatScore, slugToLabel } from "../utils/format.js";
import { createIcon } from "../utils/icons.js";
import { getTaskDefinition } from "../utils/task-meta.js";

function formatSignedScore(value) {
  const numeric = Number.isFinite(value) ? value : 0;
  const prefix = numeric > 0 ? "+" : "";
  return `${prefix}${formatScore(numeric)}`;
}

function getScoreBreakdown(auditState) {
  const reward = auditState?.reward || {};
  const observation = auditState?.observation;
  const rawState = auditState?.rawState;

  const precision = reward.precision_score ?? 0;
  const recall = reward.recall_score ?? 0;

  const totalSections = observation?.available_sections?.length || rawState?.available_sections?.length || 0;
  const reviewedSections = observation?.sections_reviewed?.length || rawState?.sections_reviewed?.length || 0;
  const coverage = reward.coverage_score ?? (totalSections > 0 ? reviewedSections / totalSections : 0);

  const stepCount = observation?.step_count ?? rawState?.step_count ?? 0;
  const maxSteps = rawState?.max_steps || 0;
  const efficiencyBonus =
    reward.efficiency_bonus ?? (maxSteps > 0 ? Math.max(0, 1 - stepCount / maxSteps) : 0);

  const progressBonus = reward.progress_bonus ?? 0.1 * coverage + 0.3 * recall;
  const falsePositivePenalty = reward.false_positive_penalty ?? 0;

  const actionHistory = rawState?.action_history || [];
  const readActions = actionHistory.filter((actionType) => actionType === "read_section");
  const repeatedReads = Math.max(0, readActions.length - new Set(readActions).size);
  const repetitionPenalty = reward.repetition_penalty ?? -0.02 * repeatedReads;

  const computedTotal =
    0.35 * precision +
    0.35 * recall +
    0.15 * coverage +
    0.1 * efficiencyBonus +
    0.05 * progressBonus +
    falsePositivePenalty +
    repetitionPenalty;
  const total = reward.total ?? Math.max(0, Math.min(1, computedTotal));

  return {
    precision,
    recall,
    coverage,
    efficiencyBonus,
    progressBonus,
    falsePositivePenalty,
    repetitionPenalty,
    total,
  };
}

function renderScorePreview(auditState) {
  const score = getScoreBreakdown(auditState);

  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Live reward" }),
              createElement("h3", { className: "card-title", text: "Score preview" }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "stack",
        children: [
          ["Precision", score.precision],
          ["Recall", score.recall],
          ["Coverage", score.coverage],
          ["Efficiency bonus", score.efficiencyBonus, true],
          ["Progress bonus", score.progressBonus, true],
          ["False-positive penalty", score.falsePositivePenalty, true],
          ["Repetition penalty", score.repetitionPenalty, true],
          ["Total", score.total],
        ].map(([label, value, signed]) =>
          createElement("div", {
            className: "meta-pair",
            children: [
              createElement("span", { className: "meta-label", text: label }),
              createElement("span", {
                className: "meta-value text-mono",
                text: signed ? formatSignedScore(value) : formatScore(value),
              }),
            ],
          }),
        ),
      }),
    ],
  });
}

function renderTaskSummary(state) {
  const observation = state.audit.observation;
  const task = getTaskDefinition(state.audit.taskId);
  const modelMetadata = observation.model_card_metadata;
  const maxSteps = state.audit.rawState?.max_steps || task.maxSteps;

  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: task.difficulty }),
              createElement("h3", { className: "card-title", text: task.label }),
            ],
          }),
          state.audit.done
            ? createBadge("Completed", "status-pass")
            : createBadge(state.audit.mode === "auto" ? "Auto mode" : "Manual mode", "status-neutral"),
        ],
      }),
      createElement("div", {
        className: "stack",
        children: [
          createElement("div", {
            className: "meta-pair",
            children: [
              createElement("span", { className: "meta-label", text: "Model" }),
              createElement("span", { className: "meta-value", text: modelMetadata.model_name }),
            ],
          }),
          createElement("div", {
            className: "meta-pair",
            children: [
              createElement("span", { className: "meta-label", text: "Framework" }),
              createElement("span", { className: "meta-value", text: modelMetadata.framework }),
            ],
          }),
          createElement("div", {
            className: "meta-pair",
            children: [
              createElement("span", { className: "meta-label", text: "Progress" }),
              createElement("span", {
                className: "meta-value",
                text: `${observation.step_count} / ${maxSteps}`,
              }),
            ],
          }),
        ],
      }),
      renderProgressBar({
        current: observation.step_count,
        max: maxSteps,
      }),
    ],
  });
}

function renderActionControls(state, actions) {
  const observation = state.audit.observation;
  const availableSections = observation.available_sections;
  const suiteBusy = Boolean(state.audit.suite?.inProgress);

  const sectionSelect = createElement("select", {
    className: "select",
    attrs: {
      "aria-label": "Read model card section",
      disabled: suiteBusy,
    },
    events: {
      change: (event) => actions.setSelectedSection(event.target.value),
    },
    children: availableSections.map((section) =>
      createElement("option", {
        attrs: {
          value: section,
          selected: section === state.audit.selectedSection,
        },
        text: slugToLabel(section),
      }),
    ),
  });

  const autoEnabled = state.audit.mode === "auto";

  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Controls" }),
              createElement("h3", { className: "card-title", text: "Run the audit" }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "form-grid two-up",
        children: [
          createElement("div", {
            className: "field",
            children: [
              createElement("span", { className: "field-label", text: "Read section" }),
              sectionSelect,
            ],
          }),
          createElement("div", {
            className: "field",
            children: [
              createElement("span", { className: "field-label", text: "Auto speed" }),
              createElement("input", {
                className: "range-input",
                attrs: {
                  type: "range",
                  min: "500",
                  max: "2000",
                  step: "250",
                  value: String(state.audit.autoDelay),
                  disabled: suiteBusy,
                },
                events: {
                  input: (event) => actions.setAutoDelay(Number(event.target.value)),
                },
              }),
              createElement("div", {
                className: "small-copy",
                text: `${state.audit.autoDelay}ms between steps`,
              }),
            ],
          }),
        ],
      }),
      createElement("label", {
        className: "toggle",
        children: [
          createElement("input", {
            attrs: {
              type: "checkbox",
              checked: autoEnabled,
              disabled: suiteBusy,
            },
            events: {
              change: (event) => actions.toggleAutoMode(event.target.checked),
            },
          }),
          createElement("span", { text: "Run automatically" }),
        ],
      }),
      createElement("div", {
        className: "button-row",
        children: [
          createButton({
            label: "Read Section",
            className: "button primary",
            icon: createIcon("next"),
            onClick: () => actions.readSection(state.audit.selectedSection),
            attrs: { disabled: suiteBusy },
          }),
          createButton({
            label: state.audit.showFlagForm ? "Hide Form" : "Flag Issue",
            className: "button secondary",
            icon: createIcon("flag"),
            onClick: () => actions.toggleFlagForm(),
            attrs: { disabled: suiteBusy },
          }),
          createButton({
            label: "Submit Audit",
            className: "button ghost",
            icon: createIcon("report"),
            onClick: () => actions.submitAudit(),
            attrs: { disabled: suiteBusy },
          }),
        ],
      }),
    ],
  });
}

export function renderAuditView(state, actions) {
  const observation = state.audit.observation;
  const suite = state.audit.suite;
  const suiteBusy = Boolean(suite?.inProgress);
  const currentSuiteTask = suiteBusy ? Math.min((suite.currentIndex || 0) + 1, suite.totalTasks || 1) : 0;

  if (!observation) {
    return createElement("div", {
      className: "main-view",
      children: [
        createEmptyState(
          "No audit is running",
          "Start from the overview or the New Audit button to load a model card and begin stepping through the environment.",
          createButton({
            label: "Choose Task",
            className: "button primary",
            icon: createIcon("play"),
            onClick: () => actions.openTaskModal(),
          }),
        ),
      ],
    });
  }

  const task = getTaskDefinition(state.audit.taskId);

  return createElement("div", {
    className: "main-view",
    children: [
      state.audit.error
        ? createElement("div", {
            className: "alert error",
            children: [createElement("span", { text: state.audit.error })],
          })
        : null,
      state.audit.info?.message
        ? createElement("div", {
            className: "alert info",
            children: [createElement("span", { text: state.audit.info.message })],
          })
        : null,
      suiteBusy
        ? createElement("div", {
            className: "alert info",
            children: [
              createElement("span", {
                text: `Run-all in progress. Task ${currentSuiteTask} of ${suite.totalTasks}: ${getTaskDefinition(state.audit.taskId).label}.`,
              }),
            ],
          })
        : null,
      createElement("div", {
        className: "audit-grid",
        children: [
          createElement("div", {
            className: "audit-column",
            children: [
              renderTaskSummary(state),
              renderChecklist({
                title: `${task.label} checklist`,
                checklist: observation.checklist,
                availableSections: observation.available_sections,
                reviewedSections: observation.sections_reviewed,
              }),
            ],
          }),
          createElement("div", {
            className: "audit-column",
            children: [
              renderSectionViewer(state.audit.currentSectionName, state.audit.currentSectionContent),
              renderActionControls(state, actions),
              state.audit.showFlagForm
                ? renderFlagForm({
                    taskId: state.audit.taskId,
                    availableSections: observation.available_sections,
                    defaultSection: state.audit.currentSectionName || state.audit.selectedSection,
                    onCancel: () => actions.toggleFlagForm(false),
                    onSubmit: (payload) => actions.flagIssue(payload),
                  })
                : null,
            ],
          }),
          createElement("div", {
            className: "audit-column",
            children: [
              renderScorePreview(state.audit),
              createElement("section", {
                className: "card",
                children: [
                  createElement("div", {
                    className: "card-header",
                    children: [
                      createElement("div", {
                        children: [
                          createElement("div", { className: "card-eyebrow", text: "Findings" }),
                          createElement("h3", { className: "card-title", text: "Issues flagged" }),
                        ],
                      }),
                      createBadge(
                        `${observation.findings_so_far.length} total`,
                        "status-neutral",
                      ),
                    ],
                  }),
                  observation.findings_so_far.length
                    ? createElement("div", {
                        className: "finding-list",
                        children: observation.findings_so_far.map((finding, index) =>
                          renderFindingCard(finding, index),
                        ),
                      })
                    : createElement("p", {
                        className: "card-subtitle",
                        text: "No issues flagged yet. Read sections or use auto mode to populate the findings log.",
                      }),
                ],
              }),
              createElement("details", {
                className: "card",
                children: [
                  createElement("summary", {
                    className: "card-title",
                    text: "Developer state",
                  }),
                  createElement("pre", {
                    className: "details-json",
                    text: JSON.stringify(state.audit.rawState || {}, null, 2),
                  }),
                ],
              }),
            ],
          }),
        ],
      }),
    ],
  });
}
