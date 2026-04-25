import { createElement } from "../utils/dom.js";
import { slugToLabel } from "../utils/format.js";

function isMarkdownTable(lines, index) {
  return (
    lines[index]?.includes("|") &&
    lines[index + 1] &&
    /^[\s|:-]+$/.test(lines[index + 1].trim())
  );
}

function parseTableRow(line) {
  return line
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean);
}

function buildTable(rows) {
  const [headerRow, , ...bodyRows] = rows;
  const table = createElement("div", {
    className: "table-shell",
    children: [
      createElement("table", {
        className: "data-table",
        children: [
          createElement("thead", {
            children: [
              createElement("tr", {
                children: parseTableRow(headerRow).map((cell) =>
                  createElement("th", { text: cell }),
                ),
              }),
            ],
          }),
          createElement("tbody", {
            children: bodyRows.map((row) =>
              createElement("tr", {
                children: parseTableRow(row).map((cell) =>
                  createElement("td", { text: cell }),
                ),
              }),
            ),
          }),
        ],
      }),
    ],
  });

  return table;
}

export function renderSectionContent(content) {
  if (!content) {
    return createElement("div", {
      className: "empty-state",
      children: [
        createElement("div", {
          children: [
            createElement("h3", {
              className: "card-title",
              text: "No section loaded yet",
            }),
            createElement("p", {
              text: "Choose a section from the dropdown or let auto mode walk through the model card.",
            }),
          ],
        }),
      ],
    });
  }

  const lines = String(content).split("\n");
  const nodes = [];
  let index = 0;

  while (index < lines.length) {
    const currentLine = lines[index]?.trim();

    if (!currentLine) {
      index += 1;
      continue;
    }

    if (isMarkdownTable(lines, index)) {
      const tableLines = [lines[index], lines[index + 1]];
      index += 2;

      while (index < lines.length && lines[index].includes("|")) {
        tableLines.push(lines[index]);
        index += 1;
      }

      nodes.push(buildTable(tableLines));
      continue;
    }

    if (currentLine.startsWith("- ")) {
      const listItems = [];
      while (index < lines.length && lines[index].trim().startsWith("- ")) {
        listItems.push(lines[index].trim().slice(2));
        index += 1;
      }
      nodes.push(
        createElement("ul", {
          children: listItems.map((item) => createElement("li", { text: item })),
        }),
      );
      continue;
    }

    const paragraphLines = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !isMarkdownTable(lines, index) &&
      !lines[index].trim().startsWith("- ")
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }

    nodes.push(
      createElement("p", {
        text: paragraphLines.join(" "),
      }),
    );
  }

  return createElement("div", {
    className: "section-content",
    children: nodes,
  });
}

export function renderSectionViewer(sectionName, content) {
  return createElement("section", {
    className: "stack loose",
    children: [
      createElement("div", {
        className: "card-header",
        children: [
          createElement("div", {
            children: [
              createElement("div", { className: "card-eyebrow", text: "Section Viewer" }),
              createElement("h3", {
                className: "card-title",
                text: sectionName ? slugToLabel(sectionName) : "No section selected",
              }),
            ],
          }),
        ],
      }),
      createElement("div", {
        className: "section-frame",
        children: [renderSectionContent(content)],
      }),
    ],
  });
}
