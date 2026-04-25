import { createButton, createElement } from "../utils/dom.js";
import { slugToLabel } from "../utils/format.js";

const issueTypes = ["missing", "inconsistent", "insufficient", "non_compliant"];
const severities = ["low", "medium", "high", "critical"];

function createSeverityRadio(option, selected) {
  const id = `severity-${option}`;
  return createElement("label", {
    className: `chip-radio ${selected === option ? "selected" : ""}`,
    attrs: { for: id },
    children: [
      createElement("input", {
        attrs: {
          id,
          type: "radio",
          name: "severity",
          value: option,
          checked: selected === option,
        },
      }),
      createElement("span", { text: slugToLabel(option) }),
    ],
  });
}

export function renderFlagForm({ taskId, availableSections, onSubmit, onCancel, defaultSection }) {
  const form = createElement("form", { className: "form-grid" });

  const sectionSelect = createElement("select", {
    className: "select",
    attrs: { name: "section_name" },
    children: availableSections.map((section) =>
      createElement("option", {
        attrs: {
          value: section,
          selected: section === defaultSection,
        },
        text: slugToLabel(section),
      }),
    ),
  });

  const issueTypeSelect = createElement("select", {
    className: "select",
    attrs: { name: "issue_type" },
    children: issueTypes.map((issueType) =>
      createElement("option", {
        attrs: { value: issueType },
        text: slugToLabel(issueType),
      }),
    ),
  });

  const descriptionInput = createElement("textarea", {
    className: "textarea",
    attrs: {
      name: "description",
      required: true,
      placeholder: "Describe the evidence for this issue...",
    },
  });

  const suggestionInput = createElement("textarea", {
    className: "textarea",
    attrs: {
      name: "suggestion",
      placeholder: "Optional remediation guidance...",
    },
  });

  const regulationField = createElement("div", {
    className: "field",
    attrs: { hidden: taskId !== "regulatory_compliance" },
    children: [
      createElement("label", {
        className: "field-label",
        attrs: { for: "regulation" },
        text: "Regulation",
      }),
      createElement("input", {
        className: "input",
        attrs: {
          id: "regulation",
          name: "regulation",
          placeholder: "EU AI Act Article 14",
        },
      }),
    ],
  });

  form.append(
    createElement("div", {
      className: "card compact",
      children: [
        createElement("div", {
          className: "card-header",
          children: [
            createElement("div", {
              children: [
                createElement("div", { className: "card-eyebrow", text: "Flag Issue" }),
                createElement("h3", { className: "card-title", text: "Create finding" }),
              ],
            }),
          ],
        }),
        createElement("div", {
          className: "form-grid two-up",
          children: [
            createElement("div", {
              className: "field",
              children: [
                createElement("label", {
                  className: "field-label",
                  attrs: { for: "flag-section" },
                  text: "Section",
                }),
                sectionSelect,
              ],
            }),
            createElement("div", {
              className: "field",
              children: [
                createElement("label", {
                  className: "field-label",
                  attrs: { for: "flag-issue-type" },
                  text: "Issue type",
                }),
                issueTypeSelect,
              ],
            }),
          ],
        }),
        createElement("div", {
          className: "field",
          children: [
            createElement("span", { className: "field-label", text: "Severity" }),
            createElement("div", {
              className: "radio-row",
              children: severities.map((severity, index) =>
                createSeverityRadio(severity, index === 1 ? "medium" : ""),
              ),
            }),
          ],
        }),
        createElement("div", {
          className: "field",
          children: [
            createElement("label", { className: "field-label", text: "Description" }),
            descriptionInput,
          ],
        }),
        createElement("div", {
          className: "field",
          children: [
            createElement("label", { className: "field-label", text: "Suggested fix" }),
            suggestionInput,
          ],
        }),
        regulationField,
        createElement("div", {
          className: "button-row",
          children: [
            createButton({
              label: "Save Finding",
              className: "button primary",
              onClick: () => {},
              attrs: { type: "submit" },
            }),
            createButton({
              label: "Cancel",
              className: "button ghost",
              onClick: onCancel,
            }),
          ],
        }),
      ],
    }),
  );

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    onSubmit({
      action_type: "flag_issue",
      section_name: formData.get("section_name"),
      issue_type: formData.get("issue_type"),
      severity: formData.get("severity") || "medium",
      description: formData.get("description"),
      suggestion: formData.get("suggestion") || undefined,
      regulation: formData.get("regulation") || undefined,
    });
  });

  return form;
}
