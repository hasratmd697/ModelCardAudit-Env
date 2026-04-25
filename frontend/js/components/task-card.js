import { createBadge, createButton, createElement } from "../utils/dom.js";
import { createIcon } from "../utils/icons.js";

function renderDifficulty(level) {
  return createElement("div", {
    className: "task-difficulty",
    attrs: { "aria-label": `${level} out of 3 difficulty` },
    children: [1, 2, 3].map((index) =>
      createElement("span", {
        className: index <= level ? "active" : "",
        attrs: { "aria-hidden": "true" },
      }),
    ),
  });
}

export function renderTaskCard(task, actions) {
  return createElement("article", {
    className: "card",
    children: [
      createElement("div", {
        className: "task-card-copy",
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
              createBadge(task.difficulty, "status-neutral"),
            ],
          }),
          createElement("p", {
            className: "card-subtitle",
            text: task.description,
          }),
        ],
      }),
      createElement("div", {
        className: "stack",
        children: [
          createElement("div", {
            className: "meta-pair",
            children: [
              createElement("span", { className: "meta-label", text: "Difficulty" }),
              renderDifficulty(task.difficultyLevel),
            ],
          }),
          createElement("div", {
            className: "meta-pair",
            children: [
              createElement("span", { className: "meta-label", text: "Checklist" }),
              createElement("span", {
                className: "meta-value",
                text: `${task.checklistItems} items`,
              }),
            ],
          }),
          createElement("div", {
            className: "meta-pair",
            children: [
              createElement("span", { className: "meta-label", text: "Max steps" }),
              createElement("span", { className: "meta-value", text: String(task.maxSteps) }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "stack",
        children: [
          createElement("div", { className: "card-title", text: "Scoring" }),
          createElement("div", {
            className: "scoring-list",
            children: task.scoring.map((item) =>
              createElement("div", {
                className: "scoring-line",
                children: [
                  createElement("span", { className: "meta-label", text: item.label }),
                  createElement("span", { className: "meta-value", text: item.value }),
                ],
              }),
            ),
          }),
        ],
      }),
      createButton({
        label: "Start Audit",
        className: "button primary",
        icon: createIcon("play"),
        onClick: () => actions.openTaskModal(task.id),
      }),
    ],
  });
}
