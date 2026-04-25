function loadRecentRuns() {
  try {
    const serialized = window.localStorage.getItem("modelcardaudit-recent-runs");
    return serialized ? JSON.parse(serialized) : [];
  } catch {
    return [];
  }
}

function saveRecentRuns(runs) {
  try {
    window.localStorage.setItem("modelcardaudit-recent-runs", JSON.stringify(runs));
  } catch {
    // Ignore storage issues in private browsing or locked-down environments.
  }
}

const initialState = {
  route: "overview",
  server: {
    status: "checking",
    message: "Checking backend connectivity...",
    lastCheckedAt: null,
  },
  tasks: {
    loading: true,
    available: [],
    error: null,
  },
  audit: {
    taskId: null,
    mode: "manual",
    autoDelay: 1000,
    loading: false,
    stepping: false,
    error: null,
    observation: null,
    reward: null,
    info: null,
    done: false,
    rawState: null,
    modelCard: null,
    sourceType: "local",
    hfRepoId: "",
    hfRevision: "",
    currentSectionName: null,
    currentSectionContent: null,
    selectedSection: "",
    showFlagForm: false,
    lastActionType: null,
    suite: null,
  },
  report: null,
  recentRuns: loadRecentRuns(),
  ui: {
    modal: null,
  },
};

function createStore(baseState) {
  let state = baseState;
  const listeners = new Set();

  function notify() {
    listeners.forEach((listener) => listener(state));
  }

  return {
    getState() {
      return state;
    },
    setState(update) {
      state = typeof update === "function" ? update(state) : { ...state, ...update };
      saveRecentRuns(state.recentRuns);
      notify();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export const store = createStore(initialState);

export function getInitialState() {
  return initialState;
}
