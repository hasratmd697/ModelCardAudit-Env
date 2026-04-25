import { createButton, createElement } from "../utils/dom.js";
import { createIcon } from "../utils/icons.js";
import { getTaskDefinition } from "../utils/task-meta.js";

const routeLabels = {
  overview: "Portfolio Overview",
  audit: "Live Audit Workspace",
  report: "Audit Report",
  cards: "Model Card Viewer",
};

export function renderTopbar(state, actions) {
  const activeTask =
    state.route === "report" && state.report
      ? { label: state.report.taskLabel }
      : state.audit.taskId
        ? getTaskDefinition(state.audit.taskId)
        : null;
  const routeLabel = routeLabels[state.route] || "ModelCardAudit";

  return createElement("header", {
    className: "topbar",
    children: [
      createElement("div", {
        children: [
          createElement("div", { className: "topbar-title", text: "ModelCardAudit" }),
          createElement("div", {
            className: "topbar-copy",
            text: activeTask
              ? `${routeLabel} · ${activeTask.label}`
              : routeLabel,
          }),
        ],
      }),
      createElement("div", {
        className: "layout-cluster",
        children: [
          createElement("div", {
            className: "route-pill",
            children: [
              createIcon("server"),
              createElement("span", { text: state.server.message }),
            ],
          }),
          createButton({
            label: "Refresh",
            className: "button ghost",
            icon: createIcon("refresh"),
            onClick: () => actions.refreshDataForCurrentView(),
          }),
        ],
      }),
    ],
  });
}
