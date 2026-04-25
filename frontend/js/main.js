import "../css/variables.css";
import "../css/reset.css";
import "../css/layout.css";
import "../css/components.css";
import "../css/views.css";

import { api } from "./api.js";
import { renderModal } from "./components/modal.js";
import { renderSidebar } from "./components/sidebar.js";
import { renderTopbar } from "./components/topbar.js";
import { navigate, initRouter } from "./router.js";
import { store } from "./state.js";
import { renderAuditView } from "./views/audit.js";
import { renderCardViewerView } from "./views/card-viewer.js";
import { renderOverviewView } from "./views/overview.js";
import { renderReportView } from "./views/report.js";
import { createButton, createElement } from "./utils/dom.js";
import { getAutoAction } from "./utils/audit-logic.js";
import { downloadJson, formatDateTime } from "./utils/format.js";
import { createIcon } from "./utils/icons.js";
import { getTaskDefinition, TASK_ORDER } from "./utils/task-meta.js";

const appRoot = document.querySelector("#app");
let autoRunTimer = null;
const TASK_FILTER_ORDER = ["all", "easy", "medium", "hard"];

function getTaskFilterForTask(taskId) {
  const difficulty = getTaskDefinition(taskId).difficulty.toLowerCase();
  return TASK_FILTER_ORDER.includes(difficulty) ? difficulty : "all";
}

function getDefaultSelectedSection(observation) {
  if (!observation) {
    return "";
  }

  const unreviewed = observation.available_sections.find(
    (section) => !observation.sections_reviewed.includes(section),
  );

  return unreviewed || observation.available_sections[0] || "";
}

function buildModelCard(rawState, observation) {
  if (!rawState && !observation) {
    return null;
  }

  const metadata = rawState?.model_card_metadata || observation?.model_card_metadata;
  const sections = rawState?.model_card_sections || {};

  return {
    id: rawState?.model_card_id || metadata?.model_name || "unknown-model-card",
    metadata,
    sections,
  };
}

function buildReportPayload(auditState) {
  const observation = auditState.observation;
  const task = getTaskDefinition(auditState.taskId);

  return {
    type: "single",
    taskId: task.id,
    taskLabel: task.label,
    modelName: observation.model_card_metadata.model_name,
    completedAt: new Date().toISOString(),
    finalScore: auditState.info?.score ?? auditState.reward?.total ?? 0,
    reward: auditState.reward,
    findings: observation.findings_so_far,
    checklist: observation.checklist,
    sectionsReviewed: observation.sections_reviewed,
    availableSections: observation.available_sections,
    stepCount: observation.step_count,
    maxSteps: auditState.rawState?.max_steps || task.maxSteps,
    rawState: auditState.rawState,
  };
}

function averageMetric(values) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0) / values.length;
}

function collectUniqueStrings(values) {
  return [...new Set(values.filter(Boolean))];
}

function buildSuiteReport(taskReports) {
  const rewardFields = [
    "total",
    "precision_score",
    "recall_score",
    "coverage_score",
    "efficiency_bonus",
    "false_positive_penalty",
    "progress_bonus",
    "repetition_penalty",
  ];

  const reward = rewardFields.reduce((accumulator, field) => {
    accumulator[field] = averageMetric(taskReports.map((report) => report.reward?.[field] ?? 0));
    return accumulator;
  }, {});
  const availableSections = collectUniqueStrings(taskReports.flatMap((report) => report.availableSections));

  return {
    type: "suite",
    taskId: "full_audit_suite",
    taskLabel: "Full Audit Suite",
    modelName: taskReports[0]?.modelName || "Unknown model",
    completedAt: new Date().toISOString(),
    finalScore: averageMetric(taskReports.map((report) => report.finalScore)),
    reward,
    findings: taskReports.flatMap((report) =>
      report.findings.map((finding) => ({
        ...finding,
        taskId: report.taskId,
        taskLabel: report.taskLabel,
      })),
    ),
    checklist: availableSections.map((section) => ({
      id: section,
      section,
      requirement: "Review this section during at least one audit pass.",
    })),
    sectionsReviewed: collectUniqueStrings(taskReports.flatMap((report) => report.sectionsReviewed)),
    availableSections,
    stepCount: taskReports.reduce((total, report) => total + (report.stepCount || 0), 0),
    maxSteps: taskReports.reduce((total, report) => total + (report.maxSteps || 0), 0),
    rawState: {
      task_reports: taskReports.map((report) => ({
        task_id: report.taskId,
        task_label: report.taskLabel,
        final_score: report.finalScore,
        raw_state: report.rawState,
      })),
    },
    taskReports,
  };
}

function wait(delay) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, delay);
  });
}

function clearAutoRun() {
  if (autoRunTimer) {
    window.clearTimeout(autoRunTimer);
    autoRunTimer = null;
  }
}

function setDocumentTitle(state) {
  const routeTitles = {
    overview: "Overview",
    audit: "Live Audit",
    report: "Audit Report",
    cards: "Model Cards",
  };

  document.title = `ModelCardAudit Dashboard · ${routeTitles[state.route] || "Overview"}`;
}

async function refreshTasks() {
  store.setState((state) => ({
    ...state,
    tasks: {
      ...state.tasks,
      loading: true,
      error: null,
    },
  }));

  try {
    const data = await api.getTasks();
    store.setState((state) => ({
      ...state,
      tasks: {
        loading: false,
        available: data.tasks || [],
        error: null,
      },
    }));
  } catch (error) {
    store.setState((state) => ({
      ...state,
      tasks: {
        loading: false,
        available: [],
        error: error.message,
      },
    }));
  }
}

async function checkHealth() {
  try {
    const payload = await api.getHealth();
    const nextStatus = "online";
    const nextMessage = payload.message || "Backend connected";

    store.setState((state) => {
      if (state.server.status === nextStatus && state.server.message === nextMessage) {
        return state;
      }

      return {
        ...state,
        server: {
          status: nextStatus,
          message: nextMessage,
          lastCheckedAt: new Date().toISOString(),
        },
      };
    });
  } catch (error) {
    const nextStatus = "offline";
    const nextMessage = error.message || "Backend unavailable";

    store.setState((state) => {
      if (state.server.status === nextStatus && state.server.message === nextMessage) {
        return state;
      }

      return {
        ...state,
        server: {
          status: nextStatus,
          message: nextMessage,
          lastCheckedAt: new Date().toISOString(),
        },
      };
    });
  }
}

async function fetchRawAuditState() {
  try {
    return await api.getState();
  } catch {
    return null;
  }
}

function openTaskModal(preselectedTask) {
  const state = store.getState();
  const selectedTask = preselectedTask || state.audit.taskId || TASK_ORDER[0];
  store.setState({
    ...state,
    ui: {
      ...state.ui,
      modal: {
        type: "task-selector",
        selectedTask,
        taskFilter: preselectedTask ? getTaskFilterForTask(selectedTask) : "all",
        mode: state.audit.mode || "manual",
        autoDelay: state.audit.autoDelay || 1000,
        sourceType: state.audit.sourceType || "local",
        hfRepoId: state.audit.hfRepoId || "",
        hfRevision: state.audit.hfRevision || "",
        error: null,
      },
    },
  });
}

function closeModal() {
  const state = store.getState();
  store.setState({
    ...state,
    ui: {
      ...state.ui,
      modal: null,
    },
  });
}

function updateModal(patch) {
  const state = store.getState();
  if (!state.ui.modal) {
    return;
  }

  store.setState({
    ...state,
    ui: {
      ...state.ui,
      modal: {
        ...state.ui.modal,
        ...patch,
      },
    },
  });
}

function normalizeAuditConfig(config) {
  return {
    taskId: config.taskId,
    mode: config.mode || "manual",
    autoDelay: Number(config.autoDelay) || 1000,
    sourceType: config.sourceType === "huggingface" ? "huggingface" : "local",
    hfRepoId: (config.hfRepoId || "").trim(),
    hfRevision: (config.hfRevision || "").trim(),
  };
}

function validateAuditConfig(config) {
  if (config.sourceType === "huggingface" && !config.hfRepoId) {
    updateModal({
      error: "Enter a Hugging Face repo id in the form owner/model-name (for example: google/gemma-2-9b).",
    });
    return false;
  }

  return true;
}

async function initializeAuditRun(config, suite = null) {
  clearAutoRun();

  store.setState((state) => ({
    ...state,
    report: null,
    ui: {
      ...state.ui,
      modal: state.ui.modal
        ? {
            ...state.ui.modal,
            error: null,
          }
        : null,
    },
    audit: {
      ...state.audit,
      taskId: config.taskId,
      mode: config.mode,
      autoDelay: config.autoDelay,
      sourceType: config.sourceType,
      hfRepoId: config.hfRepoId,
      hfRevision: config.hfRevision,
      loading: true,
      stepping: false,
      error: null,
      observation: null,
      reward: null,
      info: null,
      done: false,
      rawState: null,
      modelCard: null,
      currentSectionName: null,
      currentSectionContent: null,
      selectedSection: "",
      showFlagForm: false,
      lastActionType: null,
      suite,
    },
  }));

  try {
    const observation = await api.resetAudit(
      config.taskId,
      config.sourceType === "huggingface"
        ? {
            hfRepoId: config.hfRepoId,
            hfRevision: config.hfRevision,
          }
        : {},
    );
    const rawState = await fetchRawAuditState();
    const selectedSection = getDefaultSelectedSection(observation);

    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        loading: false,
        taskId: config.taskId,
        mode: config.mode,
        autoDelay: config.autoDelay,
        sourceType: config.sourceType,
        hfRepoId: config.hfRepoId,
        hfRevision: config.hfRevision,
        observation,
        reward: null,
        info: null,
        rawState,
        modelCard: buildModelCard(rawState, observation),
        selectedSection,
        suite,
      },
      ui: {
        ...state.ui,
        modal: null,
      },
    }));

    navigate("audit");
    return true;
  } catch (error) {
    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        loading: false,
        stepping: false,
        error: error.message,
        suite: suite
          ? {
              ...suite,
              inProgress: false,
            }
          : null,
      },
      ui: {
        ...state.ui,
        modal: state.ui.modal
          ? {
              ...state.ui.modal,
              error: error.message,
            }
          : null,
      },
    }));
    return false;
  }
}

async function startAudit(config) {
  const nextConfig = normalizeAuditConfig(config);
  if (!validateAuditConfig(nextConfig)) {
    return;
  }

  const initialized = await initializeAuditRun(nextConfig, null);
  if (!initialized) {
    return;
  }

  if (nextConfig.mode === "auto") {
    scheduleAutoRun();
  }
}

async function startAuditSuite(config) {
  const nextConfig = normalizeAuditConfig({
    ...config,
    mode: "auto",
  });
  if (!validateAuditConfig(nextConfig)) {
    return;
  }

  const taskReports = [];
  const taskIds = [...TASK_ORDER];

  try {
    for (let index = 0; index < taskIds.length; index += 1) {
      const taskId = taskIds[index];
      const suiteState = {
        active: true,
        inProgress: true,
        taskIds,
        totalTasks: taskIds.length,
        currentIndex: index,
        taskReports: [...taskReports],
      };

      const initialized = await initializeAuditRun(
        {
          ...nextConfig,
          taskId,
        },
        suiteState,
      );

      if (!initialized) {
        return;
      }

      while (true) {
        const currentAudit = store.getState().audit;
        if (!currentAudit.observation || currentAudit.done) {
          break;
        }

        const action = getAutoAction(currentAudit.observation);
        if (!action) {
          break;
        }

        await wait(nextConfig.autoDelay);
        await performAction(action, {
          internal: true,
          skipReport: true,
          skipNavigateToReport: true,
          skipAutoSchedule: true,
          rethrow: true,
        });
      }

      const completedAudit = store.getState().audit;
      taskReports.push(buildReportPayload(completedAudit));
    }

    const suiteReport = buildSuiteReport(taskReports);
    store.setState((state) => ({
      ...state,
      report: suiteReport,
      recentRuns: [suiteReport, ...state.recentRuns].slice(0, 8),
      audit: {
        ...state.audit,
        loading: false,
        stepping: false,
        error: null,
        info: {
          message: "Full audit suite completed.",
        },
        suite: {
          active: true,
          inProgress: false,
          taskIds,
          totalTasks: taskIds.length,
          currentIndex: taskIds.length,
          taskReports,
        },
      },
    }));

    navigate("report");
  } catch (error) {
    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        loading: false,
        stepping: false,
        error: error.message,
        suite: state.audit.suite
          ? {
              ...state.audit.suite,
              inProgress: false,
              taskReports,
            }
          : null,
      },
    }));
  }
}

async function performAction(action, options = {}) {
  const beforeState = store.getState();

  if (
    !beforeState.audit.observation ||
    beforeState.audit.stepping ||
    beforeState.audit.done ||
    (beforeState.audit.suite?.inProgress && !options.internal)
  ) {
    return null;
  }

  clearAutoRun();

  store.setState((state) => ({
    ...state,
    audit: {
      ...state.audit,
      stepping: true,
      error: null,
      info: null,
      lastActionType: action.action_type,
      currentSectionName:
        action.action_type === "read_section"
          ? action.section_name
          : state.audit.currentSectionName,
    },
  }));

  try {
    const stepResult = await api.stepAudit(action);
    const rawState = await fetchRawAuditState();

    store.setState((state) => {
      const nextObservation = stepResult.observation;
      const nextAudit = {
        ...state.audit,
        stepping: false,
        observation: nextObservation,
        reward: stepResult.reward,
        info: stepResult.info,
        done: stepResult.done,
        rawState,
        modelCard: buildModelCard(rawState, nextObservation),
        currentSectionName:
          action.action_type === "read_section"
            ? action.section_name
            : state.audit.currentSectionName,
        currentSectionContent:
          action.action_type === "read_section"
            ? (nextObservation.current_section ?? null)
            : state.audit.currentSectionContent,
        selectedSection: getDefaultSelectedSection(nextObservation),
        showFlagForm:
          action.action_type === "flag_issue" ? false : state.audit.showFlagForm,
      };

      const nextState = {
        ...state,
        audit: nextAudit,
      };

      if (stepResult.done && !options.skipReport) {
        const report = buildReportPayload(nextAudit);
        nextState.report = report;
        nextState.recentRuns = [report, ...state.recentRuns].slice(0, 8);
      }

      return nextState;
    });

    const latestState = store.getState();

    if (latestState.audit.done && !options.skipNavigateToReport) {
      navigate("report");
      return stepResult;
    }

    if (latestState.audit.mode === "auto" && !options.skipAutoSchedule) {
      scheduleAutoRun();
    }

    return stepResult;
  } catch (error) {
    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        stepping: false,
        error: error.message,
      },
    }));
    if (options.rethrow) {
      throw error;
    }
    return null;
  }
}

function scheduleAutoRun() {
  const state = store.getState();

  if (
    state.audit.mode !== "auto" ||
    !state.audit.observation ||
    state.audit.done ||
    state.audit.stepping ||
    state.audit.suite?.inProgress
  ) {
    return;
  }

  const action = getAutoAction(state.audit.observation);
  if (!action) {
    return;
  }

  autoRunTimer = window.setTimeout(() => {
    performAction(action);
  }, state.audit.autoDelay);
}

function refreshDataForCurrentView() {
  checkHealth();

  const state = store.getState();
  if (state.route === "overview") {
    refreshTasks();
  }
  if (state.audit.taskId) {
    fetchRawAuditState().then((rawState) => {
      if (!rawState) {
        return;
      }
      store.setState((currentState) => ({
        ...currentState,
        audit: {
          ...currentState.audit,
          rawState,
          modelCard: buildModelCard(rawState, currentState.audit.observation),
        },
      }));
    });
  }
}

function exportReport() {
  const state = store.getState();
  if (!state.report) {
    return;
  }

  if (state.report.type === "suite") {
    downloadJson(
      `${state.report.modelName.replace(/\s+/g, "-").toLowerCase()}-full-audit-suite-report.json`,
      {
        generated_at: state.report.completedAt,
        task_id: state.report.taskId,
        task_label: state.report.taskLabel,
        model_name: state.report.modelName,
        final_score: state.report.finalScore,
        reward: state.report.reward,
        findings: state.report.findings,
        sections_reviewed: state.report.sectionsReviewed,
        available_sections: state.report.availableSections,
        step_count: state.report.stepCount,
        max_steps: state.report.maxSteps,
        task_reports: state.report.taskReports,
        raw_state: state.report.rawState,
      },
    );
    return;
  }

  downloadJson(
    `${state.report.modelName.replace(/\s+/g, "-").toLowerCase()}-${state.report.taskId}-report.json`,
    {
      generated_at: state.report.completedAt,
      task_id: state.report.taskId,
      task_label: state.report.taskLabel,
      model_name: state.report.modelName,
      final_score: state.report.finalScore,
      reward: state.report.reward,
      findings: state.report.findings,
      sections_reviewed: state.report.sectionsReviewed,
      available_sections: state.report.availableSections,
      checklist: state.report.checklist,
      raw_state: state.report.rawState,
    },
  );
}

const actions = {
  navigate,
  openTaskModal,
  closeModal,
  refreshDataForCurrentView,
  setSelectedSection(section) {
    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        selectedSection: section,
      },
    }));
  },
  setAutoDelay(delay) {
    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        autoDelay: delay,
      },
    }));

    if (store.getState().audit.mode === "auto") {
      clearAutoRun();
      scheduleAutoRun();
    }
  },
  toggleAutoMode(enabled) {
    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        mode: enabled ? "auto" : "manual",
      },
    }));

    clearAutoRun();
    if (enabled) {
      scheduleAutoRun();
    }
  },
  toggleFlagForm(force) {
    store.setState((state) => ({
      ...state,
      audit: {
        ...state.audit,
        showFlagForm:
          typeof force === "boolean" ? force : !state.audit.showFlagForm,
      },
    }));
  },
  readSection(sectionName) {
    if (!sectionName) {
      return;
    }
    performAction({
      action_type: "read_section",
      section_name: sectionName,
    });
  },
  flagIssue(payload) {
    performAction(payload);
  },
  submitAudit() {
    performAction({
      action_type: "submit_audit",
    });
  },
  exportReport,
};

function renderTaskModal(state) {
  const modal = state.ui.modal;
  if (!modal || modal.type !== "task-selector") {
    return null;
  }

  const visibleTaskIds =
    modal.taskFilter && modal.taskFilter !== "all"
      ? TASK_ORDER.filter((taskId) => getTaskFilterForTask(taskId) === modal.taskFilter)
      : TASK_ORDER;

  const content = createElement("div", {
    className: "stack loose",
    children: [
      createElement("div", {
        className: "field",
        children: [
          createElement("span", { className: "field-label", text: "Quick select" }),
          createElement("div", {
            className: "task-filter-row",
            children: TASK_FILTER_ORDER.map((filterKey) => {
              const selected = (modal.taskFilter || "all") === filterKey;
              const label = filterKey === "all"
                ? "All"
                : `${filterKey.charAt(0).toUpperCase()}${filterKey.slice(1)}`;

              return createElement("button", {
                className: `task-filter-button ${selected ? "selected" : ""}`.trim(),
                attrs: { type: "button" },
                events: {
                  click: () => {
                    const patch = { taskFilter: filterKey, error: null };

                    if (filterKey !== "all") {
                      const matchedTask = TASK_ORDER.find(
                        (taskId) => getTaskFilterForTask(taskId) === filterKey,
                      );
                      if (matchedTask) {
                        patch.selectedTask = matchedTask;
                      }
                    }

                    updateModal(patch);
                  },
                },
                children: [createElement("span", { text: label })],
              });
            }),
          }),
        ],
      }),
      createElement("div", {
        className: "stack",
        children: visibleTaskIds.map((taskId) => {
          const task = getTaskDefinition(taskId);
          const selected = modal.selectedTask === taskId;
          return createElement("button", {
            className: `checklist-item ${selected ? "reviewed" : "pending"}`,
            attrs: { type: "button" },
            events: {
              click: () =>
                updateModal({
                  selectedTask: taskId,
                  taskFilter: getTaskFilterForTask(taskId),
                  error: null,
                }),
            },
            children: [
              createElement("div", {
                className: "checklist-copy",
                children: [
                  createElement("strong", { text: `${task.label} (${task.difficulty})` }),
                  createElement("span", {
                    text: `${task.description} · ${task.checklistItems} checklist items · ${task.maxSteps} max steps`,
                  }),
                ],
              }),
              selected
                ? createElement("span", {
                    className: "badge status-pass",
                    text: "Selected",
                  })
                : null,
            ],
          });
        }),
      }),
      createElement("div", {
        className: "field",
        children: [
          createElement("span", { className: "field-label", text: "Source" }),
          createElement("div", {
            className: "radio-row",
            children: [
              { value: "local", label: "Local dataset" },
              { value: "huggingface", label: "Hugging Face repo" },
            ].map((source) =>
              createElement("label", {
                className: `chip-radio ${modal.sourceType === source.value ? "selected" : ""}`,
                children: [
                  createElement("input", {
                    attrs: {
                      type: "radio",
                      name: "modal-source",
                      checked: modal.sourceType === source.value,
                    },
                    events: {
                      change: () => updateModal({ sourceType: source.value, error: null }),
                    },
                  }),
                  createElement("span", { text: source.label }),
                ],
              }),
            ),
          }),
        ],
      }),
      modal.sourceType === "huggingface"
        ? createElement("div", {
            className: "form-grid two-up",
            children: [
              createElement("div", {
                className: "field",
                children: [
                  createElement("span", {
                    className: "field-label",
                    text: "Repo id",
                  }),
                  createElement("input", {
                    className: "input",
                    attrs: {
                      type: "text",
                      value: modal.hfRepoId || "",
                      placeholder: "owner/model-name",
                    },
                    events: {
                      input: (event) =>
                        updateModal({
                          hfRepoId: event.target.value,
                          error: null,
                        }),
                    },
                  }),
                  createElement("span", {
                    className: "field-hint",
                    text: "Example: google/gemma-2-9b",
                  }),
                ],
              }),
              createElement("div", {
                className: "field",
                children: [
                  createElement("span", {
                    className: "field-label",
                    text: "Revision (optional)",
                  }),
                  createElement("input", {
                    className: "input",
                    attrs: {
                      type: "text",
                      value: modal.hfRevision || "",
                      placeholder: "main",
                    },
                    events: {
                      input: (event) =>
                        updateModal({
                          hfRevision: event.target.value,
                          error: null,
                        }),
                    },
                  }),
                  createElement("span", {
                    className: "field-hint",
                    text: "If empty, the app tries main then master.",
                  }),
                ],
              }),
            ],
          })
        : null,
      createElement("div", {
        className: "form-grid two-up",
        children: [
          createElement("div", {
            className: "field",
            children: [
              createElement("span", { className: "field-label", text: "Mode" }),
              createElement("div", {
                className: "radio-row",
                children: ["manual", "auto"].map((mode) =>
                  createElement("label", {
                    className: `chip-radio ${modal.mode === mode ? "selected" : ""}`,
                    children: [
                      createElement("input", {
                        attrs: {
                          type: "radio",
                          name: "modal-mode",
                          checked: modal.mode === mode,
                        },
                        events: {
                          change: () => updateModal({ mode }),
                        },
                      }),
                      createElement("span", {
                        text: mode === "manual" ? "Manual" : "Auto demo",
                      }),
                    ],
                  }),
                ),
              }),
            ],
          }),
          createElement("div", {
            className: "field",
            children: [
              createElement("span", { className: "field-label", text: "Speed" }),
              createElement("input", {
                className: "range-input",
                attrs: {
                  type: "range",
                  min: "500",
                  max: "2000",
                  step: "250",
                  value: String(modal.autoDelay),
                },
                events: {
                  input: (event) => updateModal({ autoDelay: Number(event.target.value) }),
                },
              }),
              createElement("span", {
                className: "field-hint",
                text: `${modal.autoDelay}ms per step`,
              }),
            ],
          }),
        ],
      }),
      modal.error
        ? createElement("div", {
            className: "alert error",
            children: [createElement("span", { text: modal.error })],
          })
        : createElement("div", {
            className: "small-copy",
            text:
              modal.taskFilter === "all"
                ? "Run All 3 executes Easy, Medium, and Hard sequentially in auto mode and generates one combined final report."
                : modal.sourceType === "huggingface"
                ? "Hugging Face mode imports README.md into audit sections and runs the same audit workflow on that card."
                : "Auto mode follows the same deterministic audit logic as the baseline inference script, so you can demo the run hands-free.",
          }),
    ],
  });

  return renderModal({
    title: "Select Audit Task",
    subtitle: "Choose the audit objective and whether to step manually or watch the agent run automatically.",
    content,
    footerActions: [
      createButton({
        label: "Cancel",
        className: "button ghost",
        onClick: closeModal,
      }),
      modal.taskFilter === "all"
        ? createButton({
            label: "Run All 3",
            className: "button secondary",
            icon: createIcon("spark"),
            onClick: () =>
              startAuditSuite({
                mode: "auto",
                autoDelay: modal.autoDelay,
                sourceType: modal.sourceType,
                hfRepoId: modal.hfRepoId,
                hfRevision: modal.hfRevision,
              }),
          })
        : null,
      createButton({
        label: "Start Audit",
        className: "button primary",
        icon: createIcon("play"),
        onClick: () =>
          startAudit({
            taskId: modal.selectedTask,
            mode: modal.mode,
            autoDelay: modal.autoDelay,
            sourceType: modal.sourceType,
            hfRepoId: modal.hfRepoId,
            hfRevision: modal.hfRevision,
          }),
      }),
    ],
    onClose: closeModal,
  });
}

function renderCurrentView(state) {
  if (state.route === "audit") {
    return renderAuditView(state, actions);
  }

  if (state.route === "report") {
    return renderReportView(state, actions);
  }

  if (state.route === "cards") {
    return renderCardViewerView(state, actions);
  }

  return renderOverviewView(state, actions);
}

function renderApp(state) {
  setDocumentTitle(state);

  const shell = createElement("div", {
    className: "app-shell",
    children: [
      renderTopbar(state, actions),
      renderSidebar(state, actions),
      createElement("main", {
        className: "main-shell",
        children: [renderCurrentView(state)],
      }),
    ],
  });

  appRoot.replaceChildren(shell);

  const modal = renderTaskModal(state);
  if (modal) {
    appRoot.append(modal);
  }
}

function initKeyboardShortcuts() {
  window.addEventListener("keydown", (event) => {
    const activeTag = document.activeElement?.tagName;
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(activeTag);
    const modalOpen = Boolean(store.getState().ui.modal);

    if (typing) {
      return;
    }

    if (event.key === "Escape" && modalOpen) {
      closeModal();
      return;
    }

    const state = store.getState();

    if (state.route !== "audit" || !state.audit.observation) {
      return;
    }

    const key = event.key.toLowerCase();

    if (key === "n") {
      event.preventDefault();
      actions.readSection(state.audit.selectedSection || getDefaultSelectedSection(state.audit.observation));
    }

    if (key === "f") {
      event.preventDefault();
      actions.toggleFlagForm();
    }

    if (key === "s") {
      event.preventDefault();
      actions.submitAudit();
    }
  });
}

store.subscribe(renderApp);

initRouter((route) => {
  store.setState((state) => ({
    ...state,
    route,
  }));
});

initKeyboardShortcuts();
checkHealth();
refreshTasks();
window.setInterval(checkHealth, 10_000);

renderApp(store.getState());

window.addEventListener("beforeunload", () => {
  clearAutoRun();
});

const stateHint = store.getState();
if (stateHint.report && stateHint.route === "overview") {
  console.info(`Last report generated at ${formatDateTime(stateHint.report.completedAt)}`);
}
