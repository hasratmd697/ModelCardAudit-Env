const configuredBase = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

function buildUrl(path) {
  if (configuredBase) {
    return `${configuredBase}${path}`;
  }
  if (path === "/") {
    return "/api-root";
  }
  return path;
}

async function request(path, options = {}) {
  const response = await fetch(buildUrl(path), {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const errorPayload = await response.json();
      message = errorPayload.detail || errorPayload.message || message;
    } catch {
      // Ignore JSON parsing failure and keep the default message.
    }
    throw new Error(message);
  }

  return response.json();
}

export const api = {
  getHealth() {
    return request("/");
  },
  getTasks() {
    return request("/tasks");
  },
  resetAudit(taskId, options = {}) {
    const body = { task_id: taskId };
    if (options.hfRepoId) {
      body.hf_repo_id = options.hfRepoId;
    }
    if (options.hfRevision) {
      body.hf_revision = options.hfRevision;
    }

    return request("/reset", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
  stepAudit(action) {
    return request("/step", {
      method: "POST",
      body: JSON.stringify(action),
    });
  },
  getState() {
    return request("/state");
  },
};
