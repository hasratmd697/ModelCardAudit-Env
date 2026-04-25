import Chart from "chart.js/auto";

import { createElement } from "../utils/dom.js";
import { clamp01, formatPercent, formatScore } from "../utils/format.js";

function metric(label, value, caption) {
  return createElement("div", {
    className: "metric-card",
    children: [
      createElement("div", { className: "metric-label", text: label }),
      createElement("div", { className: "metric-value", text: formatScore(value) }),
      caption ? createElement("div", { className: "metric-caption", text: caption }) : null,
    ],
  });
}

function buildRadarChart(canvas, values) {
  if (!canvas.isConnected) {
    return;
  }

  // eslint-disable-next-line no-new
  new Chart(canvas, {
    type: "radar",
    data: {
      labels: ["Precision", "Recall", "Coverage", "Efficiency", "Overall"],
      datasets: [
        {
          label: "Audit Score",
          data: values,
          borderColor: "#2563EB",
          backgroundColor: "rgba(37, 99, 235, 0.12)",
          pointBackgroundColor: "#2563EB",
          pointBorderColor: "#FFFFFF",
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.25,
            showLabelBackdrop: false,
          },
          pointLabels: {
            color: "#5F6578",
            font: {
              family: "Inter",
              size: 12,
            },
          },
          grid: {
            color: "rgba(148, 163, 184, 0.24)",
          },
          angleLines: {
            color: "rgba(148, 163, 184, 0.24)",
          },
        },
      },
    },
  });
}

export function renderScoreDisplay({ reward, finalScore, title = "Audit score", chart = true }) {
  const effectiveTotal = finalScore ?? reward?.total ?? 0;
  const precision = reward?.precision_score ?? 0;
  const recall = reward?.recall_score ?? 0;
  const coverage = reward?.coverage_score ?? 0;
  const efficiency = clamp01(reward?.efficiency_bonus ?? 0);

  const canvas = createElement("canvas", {
    attrs: { width: "360", height: "280" },
  });

  if (chart) {
    queueMicrotask(() => buildRadarChart(canvas, [precision, recall, coverage, efficiency, effectiveTotal]));
  }

  return createElement("section", {
    className: "card",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Score Breakdown" }),
              createElement("h3", { className: "card-title", text: title }),
            ],
          }),
          createElement("div", {
            className: "badge status-neutral",
            children: [createElement("span", { text: `${formatPercent(effectiveTotal)}` })],
          }),
        ],
      }),
      createElement("div", {
        className: "metric-grid",
        children: [
          metric("Recall", recall, "Ground-truth issues found"),
          metric("Precision", precision, "Correctness of findings"),
          metric("Coverage", coverage, "Sections reviewed"),
          metric("Overall", effectiveTotal, "Final combined score"),
        ],
      }),
      chart
        ? createElement("div", {
            attrs: { style: "height: 300px;" },
            children: [canvas],
          })
        : null,
    ],
  });
}
