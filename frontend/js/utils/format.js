const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

const decimalFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function slugToLabel(value = "") {
  return value
    .split(/[_-]/g)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

export function sentenceCase(value = "") {
  return value
    .replace(/[_-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function formatPercent(value = 0) {
  return percentFormatter.format(clamp01(value));
}

export function formatScore(value = 0) {
  return decimalFormatter.format(Number.isFinite(value) ? value : 0);
}

export function formatDateTime(dateLike) {
  const date = dateLike instanceof Date ? dateLike : new Date(dateLike);
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function clamp01(value = 0) {
  return Math.max(0, Math.min(1, value));
}

export function prettyChecklistStatus(status) {
  if (status === "reviewed") return "Reviewed";
  if (status === "pending") return "Queued";
  if (status === "missing") return "Missing";
  return "Unknown";
}

export function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function uniqueBy(items, getKey) {
  const seen = new Set();
  return items.filter((item) => {
    const key = getKey(item);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}
