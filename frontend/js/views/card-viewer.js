import { renderSectionContent } from "../components/section-viewer.js";
import { createBadge, createButton, createElement, createEmptyState } from "../utils/dom.js";
import { slugToLabel } from "../utils/format.js";
import { createIcon } from "../utils/icons.js";

function getSectionInventory(modelCardSections, checklist) {
  const available = Object.keys(modelCardSections || {});
  const required = checklist.map((item) => item.section);
  const combined = new Set([...available, ...required]);
  return [...combined];
}

function setupScrollSpy(viewRoot) {
  const sections = [...viewRoot.querySelectorAll(".card-doc-section")];
  const tocLinks = [...viewRoot.querySelectorAll(".toc-link")];

  if (!sections.length || !tocLinks.length) {
    return;
  }

  const byId = new Map(tocLinks.map((link) => [link.dataset.target, link]));

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];

      if (!visible) {
        return;
      }

      tocLinks.forEach((link) => link.classList.remove("active"));
      byId.get(visible.target.id)?.classList.add("active");
    },
    {
      rootMargin: "-25% 0px -55% 0px",
      threshold: [0.2, 0.6, 0.9],
    },
  );

  sections.forEach((section) => observer.observe(section));
}

export function renderCardViewerView(state, actions) {
  const modelCard = state.audit.modelCard;

  if (!modelCard) {
    return createElement("div", {
      className: "main-view",
      children: [
        createEmptyState(
          "No model card loaded",
          "Start an audit first so the card viewer has a live model card to display.",
          createButton({
            label: "Start Audit",
            className: "button primary",
            icon: createIcon("play"),
            onClick: () => actions.openTaskModal(),
          }),
        ),
      ],
    });
  }

  const sections = getSectionInventory(modelCard.sections, state.audit.observation?.checklist || []);
  const root = createElement("div", {
    className: "main-view",
    children: [
      createElement("section", {
        className: "card",
        children: [
          createElement("div", {
            className: "card-header",
            children: [
              createElement("div", {
                children: [
                  createElement("div", { className: "card-eyebrow", text: "Model Card" }),
                  createElement("h2", {
                    className: "topbar-title",
                    text: modelCard.metadata.model_name,
                  }),
                  createElement("p", {
                    className: "card-subtitle",
                    text: `${modelCard.metadata.model_type} · ${modelCard.metadata.framework}`,
                  }),
                ],
              }),
              createButton({
                label: "Back To Audit",
                className: "button ghost",
                icon: createIcon("audit"),
                onClick: () => actions.navigate("audit"),
              }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "card-viewer-layout",
        children: [
          createElement("aside", {
            className: "card-viewer-toc card",
            children: [
              createElement("div", {
                className: "card-header",
                children: [
                  createElement("div", {
                    children: [
                      createElement("div", { className: "card-eyebrow", text: "Contents" }),
                      createElement("h3", { className: "card-title", text: "Sections" }),
                    ],
                  }),
                ],
              }),
              createElement("div", {
                className: "toc-list",
                children: sections.map((section, index) => {
                  const exists = Boolean(modelCard.sections[section]);
                  return createElement("button", {
                    className: `toc-link ${!exists ? "missing" : ""} ${index === 0 ? "active" : ""}`,
                    attrs: {
                      type: "button",
                    },
                    dataset: {
                      target: `section-${section}`,
                    },
                    events: {
                      click: () =>
                        root
                          .querySelector(`#section-${section}`)
                          ?.scrollIntoView({ behavior: "smooth", block: "start" }),
                    },
                    children: [
                      createElement("span", { text: slugToLabel(section) }),
                      exists
                        ? createBadge("Present", "status-pass")
                        : createBadge("Missing", "status-error"),
                    ],
                  });
                }),
              }),
            ],
          }),
          createElement("section", {
            className: "card card-doc",
            children: sections.map((section) => {
              const exists = Boolean(modelCard.sections[section]);
              return createElement("article", {
                className: "card-doc-section stack",
                attrs: {
                  id: `section-${section}`,
                },
                children: [
                  createElement("div", {
                    className: "card-doc-header",
                    children: [
                      createElement("h3", {
                        className: "card-title",
                        text: slugToLabel(section),
                      }),
                      exists
                        ? createBadge("Available", "status-pass")
                        : createBadge("Missing", "status-error"),
                    ],
                  }),
                  exists
                    ? renderSectionContent(modelCard.sections[section])
                    : createElement("div", {
                        className: "section-frame",
                        children: [
                          createElement("div", {
                            className: "empty-state",
                            children: [
                              createElement("div", {
                                children: [
                                  createElement("h3", {
                                    className: "card-title",
                                    text: "Section not present in current card",
                                  }),
                                  createElement("p", {
                                    text: "This section is required by the checklist but missing from the sampled model card.",
                                  }),
                                ],
                              }),
                            ],
                          }),
                        ],
                      }),
                ],
              });
            }),
          }),
        ],
      }),
    ],
  });

  queueMicrotask(() => setupScrollSpy(root));

  return root;
}
