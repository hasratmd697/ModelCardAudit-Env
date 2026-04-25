import { createElement } from "../utils/dom.js";
import { formatPercent } from "../utils/format.js";

export function renderProgressBar({ current, max }) {
  const ratio = max > 0 ? current / max : 0;
  const percentage = Math.round(ratio * 100);

  return createElement("section", {
    className: "card compact",
    children: [
      createElement("div", {
        className: "progress-meta",
        children: [
          createElement("span", { text: `Step ${current} / ${max}` }),
          createElement("span", { text: `${percentage}%` }),
        ],
      }),
      createElement("div", {
        className: "progress-track",
        children: [
          createElement("div", {
            className: "progress-fill",
            attrs: { style: `width: ${formatPercent(ratio)};` },
          }),
        ],
      }),
    ],
  });
}
