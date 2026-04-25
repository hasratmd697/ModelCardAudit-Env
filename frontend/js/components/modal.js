import { createButton, createElement } from "../utils/dom.js";
import { createIcon } from "../utils/icons.js";

export function renderModal({ title, subtitle, content, footerActions, onClose }) {
  const dialog = createElement("div", {
    className: "modal-dialog",
    attrs: {
      role: "dialog",
      "aria-modal": "true",
      "aria-labelledby": "modal-title",
    },
    children: [
      createElement("div", {
        className: "modal-header",
        children: [
          createElement("div", {
            children: [
              createElement("h2", {
                attrs: { id: "modal-title" },
                className: "card-title",
                text: title,
              }),
              subtitle
                ? createElement("p", { className: "card-subtitle", text: subtitle })
                : null,
            ],
          }),
          createButton({
            label: "Close",
            className: "button ghost",
            icon: createIcon("close"),
            onClick: onClose,
          }),
        ],
      }),
      createElement("div", {
        className: "modal-body",
        children: [content],
      }),
      footerActions
        ? createElement("div", {
            className: "modal-footer",
            children: footerActions,
          })
        : null,
    ],
  });

  const backdrop = createElement("div", {
    className: "modal-backdrop",
    events: {
      click: (event) => {
        if (event.target === backdrop) {
          onClose();
        }
      },
    },
    children: [dialog],
  });

  return backdrop;
}
