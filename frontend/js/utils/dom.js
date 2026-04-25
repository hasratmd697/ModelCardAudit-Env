export function createElement(tag, options = {}) {
  const {
    className,
    text,
    html,
    attrs,
    dataset,
    children,
    events,
  } = options;

  const element = document.createElement(tag);

  if (className) {
    element.className = className;
  }

  if (text !== undefined) {
    element.textContent = text;
  }

  if (html !== undefined) {
    element.innerHTML = html;
  }

  if (attrs) {
    Object.entries(attrs).forEach(([key, value]) => {
      if (value === false || value === null || value === undefined) {
        return;
      }
      if (value === true) {
        element.setAttribute(key, "");
        return;
      }
      element.setAttribute(key, String(value));
    });
  }

  if (dataset) {
    Object.entries(dataset).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        element.dataset[key] = String(value);
      }
    });
  }

  if (events) {
    Object.entries(events).forEach(([eventName, handler]) => {
      element.addEventListener(eventName, handler);
    });
  }

  appendChildren(element, children);
  return element;
}

export function appendChildren(parent, children = []) {
  const list = Array.isArray(children) ? children : [children];
  list
    .flat(Infinity)
    .filter(Boolean)
    .forEach((child) => {
      if (typeof child === "string") {
        parent.append(document.createTextNode(child));
      } else {
        parent.append(child);
      }
    });
  return parent;
}

export function clearElement(element) {
  element.replaceChildren();
  return element;
}

export function createFragment(children = []) {
  const fragment = document.createDocumentFragment();
  appendChildren(fragment, children);
  return fragment;
}

export function createButton({
  label,
  className = "button ghost",
  icon,
  onClick,
  attrs,
}) {
  return createElement("button", {
    className,
    attrs: {
      type: "button",
      ...attrs,
    },
    children: [icon, createElement("span", { text: label })],
    events: onClick ? { click: onClick } : undefined,
  });
}

export function createBadge(label, className) {
  return createElement("span", {
    className: `badge ${className}`.trim(),
    children: [
      createElement("span", { className: "badge-dot", attrs: { "aria-hidden": "true" } }),
      createElement("span", { text: label }),
    ],
  });
}

export function createEmptyState(title, description, actionNode) {
  return createElement("section", {
    className: "card empty-state",
    children: [
      createElement("div", {
        children: [
          createElement("h2", { className: "card-title", text: title }),
          createElement("p", { text: description }),
        ],
      }),
      actionNode || null,
    ],
  });
}
