import { createButton, createElement } from "../utils/dom.js";
import { createIcon } from "../utils/icons.js";

const navItems = [
  { route: "overview", label: "Overview", icon: "overview" },
  { route: "audit", label: "Audit", icon: "audit" },
  { route: "report", label: "Report", icon: "report" },
  { route: "cards", label: "Cards", icon: "card" },
];

export function renderSidebar(state, actions) {
  const { server } = state;

  const nav = createElement(
    "nav",
    {
      className: "sidebar-nav",
      children: navItems.map((item) =>
        createElement("button", {
          className: `sidebar-nav-link ${state.route === item.route ? "active" : ""}`,
          attrs: { type: "button" },
          events: {
            click: () => actions.navigate(item.route),
          },
          children: [
            createIcon(item.icon),
            createElement("span", {
              className: "sidebar-label",
              text: item.label,
            }),
          ],
        }),
      ),
    },
  );

  return createElement("aside", {
    className: "sidebar",
    children: [
      createElement("div", {
        className: "sidebar-brand",
        children: [
          createElement("div", {
            className: "brand-mark",
            children: [createIcon("shield", { size: 20 })],
          }),
          createElement("div", {
            className: "sidebar-meta",
            children: [
              createElement("div", { className: "brand-title", text: "ModelCardAudit" }),
              createElement("div", {
                className: "brand-subtitle",
                text: "Compliance workspace",
              }),
            ],
          }),
        ],
      }),
      nav,
      createElement("div", {
        className: "sidebar-footer stack",
        children: [
          createElement("div", {
            className: "card compact",
            children: [
              createElement("div", { className: "card-title", text: "Server status" }),
              createElement("div", {
                className: `status-indicator ${
                  server.status === "online"
                    ? "online"
                    : server.status === "offline"
                      ? "offline"
                      : "checking"
                }`,
                children: [
                  createElement("span", {
                    className: "dot",
                    attrs: { "aria-hidden": "true" },
                  }),
                  createElement("span", {
                    className: "sidebar-footer-copy",
                    text: server.message,
                  }),
                ],
              }),
            ],
          }),
          createButton({
            label: "New Audit",
            className: "button primary",
            icon: createIcon("play"),
            onClick: () => actions.openTaskModal(),
          }),
        ],
      }),
    ],
  });
}
