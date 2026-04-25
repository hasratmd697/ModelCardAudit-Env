import { createBadge, createElement } from "../utils/dom.js";
import { prettyChecklistStatus, slugToLabel } from "../utils/format.js";

function getStatus(section, availableSections, reviewedSections) {
  if (reviewedSections.includes(section)) {
    return "reviewed";
  }
  if (availableSections.includes(section)) {
    return "pending";
  }
  return "missing";
}

export function renderChecklist({ checklist, availableSections, reviewedSections, title = "Checklist" }) {
  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Review Coverage" }),
              createElement("h3", { className: "card-title", text: title }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "checklist-list",
        children: checklist.map((item) => {
          const status = getStatus(item.section, availableSections, reviewedSections);
          return createElement("div", {
            className: `checklist-item ${status}`,
            children: [
              createElement("div", {
                className: "checklist-copy",
                children: [
                  createElement("strong", { text: slugToLabel(item.section) }),
                  createElement("span", { text: item.requirement }),
                ],
              }),
              createBadge(prettyChecklistStatus(status), `status-${status}`),
            ],
          });
        }),
      }),
    ],
  });
}
