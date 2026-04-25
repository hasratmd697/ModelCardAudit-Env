const svgNs = "http://www.w3.org/2000/svg";

const icons = {
  shield: [
    ["path", { d: "M12 3l7 3v6c0 4.5-2.9 8.6-7 10-4.1-1.4-7-5.5-7-10V6l7-3z" }],
    ["path", { d: "M9 12l2 2 4-4" }],
  ],
  overview: [
    ["rect", { x: "3", y: "3", width: "8", height: "8", rx: "1.5" }],
    ["rect", { x: "13", y: "3", width: "8", height: "5", rx: "1.5" }],
    ["rect", { x: "13", y: "10", width: "8", height: "11", rx: "1.5" }],
    ["rect", { x: "3", y: "13", width: "8", height: "8", rx: "1.5" }],
  ],
  audit: [
    ["path", { d: "M4 6h16" }],
    ["path", { d: "M4 12h10" }],
    ["path", { d: "M4 18h7" }],
    ["circle", { cx: "18", cy: "12", r: "3" }],
  ],
  report: [
    ["path", { d: "M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z" }],
    ["path", { d: "M14 2v5h5" }],
    ["path", { d: "M9 13h6" }],
    ["path", { d: "M9 17h6" }],
    ["path", { d: "M9 9h2" }],
  ],
  card: [
    ["path", { d: "M4 5h16v14H4z" }],
    ["path", { d: "M8 9h8" }],
    ["path", { d: "M8 13h5" }],
    ["path", { d: "M8 17h7" }],
  ],
  play: [
    ["path", { d: "M8 5l11 7-11 7V5z" }],
  ],
  pause: [
    ["rect", { x: "6", y: "5", width: "4", height: "14", rx: "1" }],
    ["rect", { x: "14", y: "5", width: "4", height: "14", rx: "1" }],
  ],
  flag: [
    ["path", { d: "M5 3v18" }],
    ["path", { d: "M5 4h10l-1.5 4L15 12H5" }],
  ],
  refresh: [
    ["path", { d: "M20 4v6h-6" }],
    ["path", { d: "M4 20v-6h6" }],
    ["path", { d: "M20 10a8 8 0 0 0-14-4" }],
    ["path", { d: "M4 14a8 8 0 0 0 14 4" }],
  ],
  download: [
    ["path", { d: "M12 3v12" }],
    ["path", { d: "M8 11l4 4 4-4" }],
    ["path", { d: "M4 20h16" }],
  ],
  server: [
    ["rect", { x: "3", y: "4", width: "18", height: "6", rx: "1.5" }],
    ["rect", { x: "3", y: "14", width: "18", height: "6", rx: "1.5" }],
    ["path", { d: "M7 7h.01" }],
    ["path", { d: "M7 17h.01" }],
  ],
  spark: [
    ["path", { d: "M12 2l2.7 6.3L21 11l-6.3 2.7L12 20l-2.7-6.3L3 11l6.3-2.7L12 2z" }],
  ],
  close: [
    ["path", { d: "M6 6l12 12" }],
    ["path", { d: "M18 6L6 18" }],
  ],
  next: [
    ["path", { d: "M9 6l6 6-6 6" }],
  ],
  clock: [
    ["circle", { cx: "12", cy: "12", r: "9" }],
    ["path", { d: "M12 7v5l3 2" }],
  ],
};

export function createIcon(name, options = {}) {
  const { size = 18, className = "" } = options;
  const icon = icons[name] || icons.spark;
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.9");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("class", `icon ${className}`.trim());

  icon.forEach(([tag, attrs]) => {
    const node = document.createElementNS(svgNs, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    svg.append(node);
  });

  return svg;
}
