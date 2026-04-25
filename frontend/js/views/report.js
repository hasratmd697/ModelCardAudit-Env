import { renderChecklist } from "../components/checklist.js";
import { renderFindingCard } from "../components/finding-card.js";
import { renderScoreDisplay } from "../components/score-display.js";
import { createBadge, createButton, createElement, createEmptyState } from "../utils/dom.js";
import { formatDateTime, formatPercent, formatScore } from "../utils/format.js";
import { createIcon } from "../utils/icons.js";

function renderReportHeader(report, actions) {
  const subtitle =
    report.type === "suite"
      ? `${report.taskLabel} - Completed ${formatDateTime(report.completedAt)}`
      : `${report.taskLabel} - Completed ${formatDateTime(report.completedAt)}`;

  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", {
                className: "card-eyebrow",
                text: report.type === "suite" ? "Audit Suite Report" : "Audit Report",
              }),
              createElement("h2", { className: "topbar-title", text: report.modelName }),
              createElement("p", {
                className: "card-subtitle",
                text: subtitle,
              }),
            ],
          }),
          createBadge("Completed", "status-pass"),
        ],
      }),
      createElement("div", {
        className: "button-row",
        children: [
          createButton({
            label: "Export JSON",
            className: "button primary",
            icon: createIcon("download"),
            onClick: actions.exportReport,
          }),
          createButton({
            label: "New Audit",
            className: "button secondary",
            icon: createIcon("refresh"),
            onClick: () => actions.openTaskModal(),
          }),
        ],
      }),
    ],
  });
}

function renderRunSnapshot(report) {
  const summaryLines =
    report.type === "suite"
      ? [
          ["Audits completed", `${report.taskReports.length} / ${report.taskReports.length}`],
          ["Total findings", String(report.findings.length)],
          ["Steps used", `${report.stepCount} / ${report.maxSteps}`],
        ]
      : [
          ["Findings", String(report.findings.length)],
          ["Sections reviewed", `${report.sectionsReviewed.length} / ${report.availableSections.length}`],
          ["Steps used", `${report.stepCount} / ${report.maxSteps}`],
        ];

  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Summary" }),
              createElement("h3", { className: "card-title", text: "Run snapshot" }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "stack",
        children: [
          createElement("div", {
            className: "metric-card",
            children: [
              createElement("div", { className: "metric-label", text: "Final score" }),
              createElement("div", {
                className: "metric-value",
                text: `${formatScore(report.finalScore)} / 1.00`,
              }),
              createElement("div", {
                className: "metric-caption",
                text: `${formatPercent(report.finalScore)} overall performance`,
              }),
            ],
          }),
          createElement("div", {
            className: "stack",
            children: summaryLines.map(([label, value]) =>
              createElement("div", {
                className: "meta-pair",
                children: [
                  createElement("span", { className: "meta-label", text: label }),
                  createElement("span", { className: "meta-value", text: value }),
                ],
              }),
            ),
          }),
        ],
      }),
    ],
  });
}

function renderFindingsSection(report, title) {
  return createElement("section", {
    className: "card report-findings",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Findings" }),
              createElement("h3", { className: "card-title", text: title }),
            ],
          }),
          createBadge(`${report.findings.length} total`, "status-neutral"),
        ],
      }),
      report.findings.length
        ? createElement("div", {
            className: "finding-list",
            children: report.findings.map((finding, index) => renderFindingCard(finding, index)),
          })
        : createElement("p", {
            className: "card-subtitle",
            text: "No issues were flagged during this run.",
          }),
    ],
  });
}

function renderSuiteBreakdown(report) {
  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Suite Breakdown" }),
              createElement("h3", { className: "card-title", text: "Per-task results" }),
            ],
          }),
          createBadge(`${report.taskReports.length} tasks`, "status-neutral"),
        ],
      }),
      createElement("div", {
        className: "table-shell",
        children: [
          createElement("table", {
            className: "data-table",
            children: [
              createElement("thead", {
                children: [
                  createElement("tr", {
                    children: ["Task", "Score", "Findings", "Steps"].map((label) =>
                      createElement("th", { text: label }),
                    ),
                  }),
                ],
              }),
              createElement("tbody", {
                children: report.taskReports.map((taskReport) =>
                  createElement("tr", {
                    children: [
                      createElement("td", { text: taskReport.taskLabel }),
                      createElement("td", { text: formatScore(taskReport.finalScore) }),
                      createElement("td", { text: String(taskReport.findings.length) }),
                      createElement("td", { text: `${taskReport.stepCount} / ${taskReport.maxSteps}` }),
                    ],
                  }),
                ),
              }),
            ],
          }),
        ],
      }),
    ],
  });
}

export function renderReportView(state, actions) {
  const report = state.report;

  if (!report) {
    return createElement("div", {
      className: "main-view",
      children: [
        createEmptyState(
          "No completed report yet",
          "Submit an audit to generate the final report view with KPIs, findings, and the export payload.",
          createButton({
            label: "Go To Audit",
            className: "button primary",
            icon: createIcon("audit"),
            onClick: () => actions.navigate("audit"),
          }),
        ),
      ],
    });
  }

  const sharedSectionsChecklist = renderChecklist({
    title:
      report.type === "suite"
        ? "Sections reviewed across all tasks"
        : "Sections reviewed",
    checklist: report.checklist || [],
    availableSections: report.availableSections,
    reviewedSections: report.sectionsReviewed,
  });

  return createElement("div", {
    className: "main-view report-layout",
    children: [
      renderReportHeader(report, actions),
      createElement("div", {
        className: "report-hero",
        children: [
          renderScoreDisplay({
            reward: report.reward,
            finalScore: report.finalScore,
            title: report.type === "suite" ? "Combined scorecard" : "Final scorecard",
          }),
          renderRunSnapshot(report),
        ],
      }),
      report.type === "suite"
        ? createElement("div", {
            className: "report-summary-grid",
            children: [renderSuiteBreakdown(report), sharedSectionsChecklist],
          })
        : createElement("div", {
            className: "report-summary-grid",
            children: [renderFindingsSection(report, "Flagged issues"), sharedSectionsChecklist],
          }),
      report.type === "suite" ? renderFindingsSection(report, "Combined findings") : null,
    ],
  });
}
