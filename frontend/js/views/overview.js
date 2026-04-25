import { renderTaskCard } from "../components/task-card.js";
import { createBadge, createButton, createElement } from "../utils/dom.js";
import { formatDateTime, formatScore } from "../utils/format.js";
import { createIcon } from "../utils/icons.js";
import { getTaskDefinition, TASK_ORDER } from "../utils/task-meta.js";

function renderRecentRuns(runs, actions) {
  if (!runs.length) {
    return createElement("div", {
      className: "card compact",
      children: [
        createElement("div", { className: "card-title", text: "Recent runs" }),
        createElement("p", {
          className: "card-subtitle",
          text: "Completed audits will appear here once you submit a report.",
        }),
      ],
    });
  }

  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "History" }),
              createElement("h3", { className: "card-title", text: "Recent runs" }),
            ],
          }),
          createButton({
            label: "Open Report",
            className: "button ghost",
            icon: createIcon("report"),
            onClick: () => actions.navigate("report"),
          }),
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
                    children: ["Task", "Model", "Score", "Steps", "Completed"].map((label) =>
                      createElement("th", { text: label }),
                    ),
                  }),
                ],
              }),
              createElement("tbody", {
                children: runs.map((run) =>
                  createElement("tr", {
                    children: [
                      createElement("td", { text: run.taskLabel }),
                      createElement("td", { text: run.modelName }),
                      createElement("td", { text: formatScore(run.finalScore) }),
                      createElement("td", { text: `${run.stepCount} / ${run.maxSteps}` }),
                      createElement("td", { text: formatDateTime(run.completedAt) }),
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

export function renderOverviewView(state, actions) {
  const tasks = TASK_ORDER
    .filter((taskId) => state.tasks.available.includes(taskId) || !state.tasks.available.length)
    .map((taskId) => getTaskDefinition(taskId));

  const hero = createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Audit Dashboard" }),
              createElement("h2", {
                className: "topbar-title",
                text: "ModelCardAudit Environment",
              }),
              createElement("p", {
                className: "card-subtitle",
                text: "Run task-specific model-card audits, inspect findings in real time, and export structured reports for review.",
              }),
            ],
          }),
          state.server.status === "online"
            ? createBadge("Connected", "status-pass")
            : state.server.status === "offline"
              ? createBadge("Offline", "status-error")
              : createBadge("Checking", "status-warn"),
        ],
      }),
      createElement("div", {
        className: "metric-grid",
        children: [
          createElement("div", {
            className: "metric-card",
            children: [
              createElement("div", { className: "metric-label", text: "Tasks available" }),
              createElement("div", { className: "metric-value", text: String(tasks.length) }),
              createElement("div", {
                className: "metric-caption",
                text: "Easy, medium, and regulatory audits are preloaded.",
              }),
            ],
          }),
          createElement("div", {
            className: "metric-card",
            children: [
              createElement("div", { className: "metric-label", text: "Completed runs" }),
              createElement("div", {
                className: "metric-value",
                text: String(state.recentRuns.length),
              }),
              createElement("div", {
                className: "metric-caption",
                text: "Stored locally in this browser for quick demo access.",
              }),
            ],
          }),
          createElement("div", {
            className: "metric-card",
            children: [
              createElement("div", { className: "metric-label", text: "Recommended mode" }),
              createElement("div", { className: "metric-value", text: "Auto" }),
              createElement("div", {
                className: "metric-caption",
                text: "Use auto-run to demonstrate the audit agent step by step.",
              }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "button-row",
        children: [
          createButton({
            label: "Start New Audit",
            className: "button primary",
            icon: createIcon("play"),
            onClick: () => actions.openTaskModal(),
          }),
          createButton({
            label: "Review Cards",
            className: "button secondary",
            icon: createIcon("card"),
            onClick: () => actions.navigate("cards"),
          }),
        ],
      }),
    ],
  });

  const taskGrid = state.tasks.loading
    ? createElement("div", {
        className: "task-grid",
        children: Array.from({ length: 3 }, () =>
          createElement("div", {
            className: "card",
            children: [
              createElement("div", { className: "skeleton", attrs: { style: "height: 20px;" } }),
              createElement("div", { className: "skeleton", attrs: { style: "height: 72px;" } }),
              createElement("div", { className: "skeleton", attrs: { style: "height: 160px;" } }),
            ],
          }),
        ),
      })
    : createElement("div", {
        className: "task-grid",
        children: tasks.map((task) => renderTaskCard(task, actions)),
      });

  return createElement("div", {
    className: "main-view",
    children: [
      state.tasks.error
        ? createElement("div", {
            className: "alert error",
            children: [createElement("span", { text: state.tasks.error })],
          })
        : null,
      hero,
      taskGrid,
      renderRecentRuns(state.recentRuns, actions),
    ],
  });
}
