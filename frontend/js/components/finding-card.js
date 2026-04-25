import { createBadge, createElement } from "../utils/dom.js";
import { slugToLabel } from "../utils/format.js";

export function renderFindingCard(finding, index = 0) {
  const metaPrefix = finding.taskLabel ? `${finding.taskLabel} - ` : "";

  return createElement("article", {
    className: "finding-card",
    children: [
      createElement("div", {
        className: "finding-title-row",
        children: [
          createElement("div", {
            children: [
              createElement("div", {
                className: "card-title",
                text: `Finding #${index + 1}`,
              }),
              createElement("div", {
                className: "small-copy",
                text: `${metaPrefix}${slugToLabel(finding.section)} - ${slugToLabel(finding.type)}`,
              }),
            ],
          }),
          createElement("div", {
            className: "finding-meta",
            children: [createBadge(slugToLabel(finding.severity), `severity-${finding.severity}`)],
          }),
        ],
      }),
      createElement("p", {
        className: "finding-description",
        text: finding.description,
      }),
      finding.suggested_fix || finding.suggestion
        ? createElement("div", {
            className: "finding-extra",
            children: [
              createElement("strong", { text: "Suggestion: " }),
              createElement("span", {
                text: finding.suggested_fix || finding.suggestion,
              }),
            ],
          })
        : null,
      finding.regulation
        ? createElement("div", {
            className: "finding-extra",
            children: [
              createElement("strong", { text: "Regulation: " }),
              createElement("span", { text: finding.regulation }),
            ],
          })
        : null,
    ],
  });
}
