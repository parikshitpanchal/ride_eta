const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export function getStoredToken() {
  if (typeof window !== "undefined") {
    return localStorage.getItem("token");
  }
  return null;
}

export async function fetchApi(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const token = getStoredToken();

  const headers = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    if (response.status === 401) {
      if (typeof window !== "undefined") {
        localStorage.removeItem("token");
        window.location.href = "/login";
      }
      return;
    }
    throw new Error(errorData.detail || `API Error: ${response.statusText}`);
  }

  return response.json();
}

// Authentication API
export async function loginUser(username, password) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Invalid credentials");
  }

  return response.json();
}

export async function getCurrentUser() {
  return fetchApi("/auth/me");
}

// Predictions API
export async function getPredictionStats() {
  return fetchApi("/predictions/stats");
}

export async function getPredictions({ page = 1, pageSize = 50, delayStatus = "", search = "", sortBy = "id", sortOrder = "desc" }) {
  const params = new URLSearchParams({
    page,
    page_size: pageSize,
    sort_by: sortBy,
    sort_order: sortOrder,
  });
  if (delayStatus) params.append("delay_status", delayStatus);
  if (search) params.append("search", search);

  return fetchApi(`/predictions?${params.toString()}`);
}

// Model Metrics API
export async function getLatestMetrics() {
  return fetchApi("/model/metrics");
}

export async function getTrainingHistory() {
  return fetchApi("/model/history");
}

// Training API
export async function getTrainingStatus() {
  return fetchApi("/training/status");
}

export async function startTraining() {
  return fetchApi("/training/start", { method: "POST" });
}

export async function stopTraining() {
  return fetchApi("/training/stop", { method: "POST" });
}

export async function getTrainingRuns() {
  return fetchApi("/training/runs");
}

export async function getTrainingRunEpochs(runId) {
  return fetchApi(`/training/runs/${runId}/epochs`);
}

// Driver Analytics API
export async function getTopDrivers() {
  return fetchApi("/drivers/top-performers");
}

export async function getWorstDrivers() {
  return fetchApi("/drivers/worst-performers");
}

export async function getDrivers({ page = 1, pageSize = 50, search = "", sortBy = "delay_rate", sortOrder = "desc" }) {
  const params = new URLSearchParams({
    page,
    page_size: pageSize,
    sort_by: sortBy,
    sort_order: sortOrder,
  });
  if (search) params.append("search", search);

  return fetchApi(`/drivers?${params.toString()}`);
}

// Admin & Pipeline API
export async function getPipelineStatus() {
  return fetchApi("/admin/pipeline-status");
}

export async function getConfig() {
  return fetchApi("/admin/config");
}

export async function updateConfig(configData) {
  return fetchApi("/admin/config", {
    method: "PUT",
    body: JSON.stringify(configData),
  });
}

export async function uploadCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  const token = getStoredToken();

  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}/admin/upload-csv`, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "CSV upload failed");
  }

  return response.json();
}

export async function runFeatureEngineering() {
  return fetchApi("/admin/run-feature-engineering", { method: "POST" });
}

export async function runPrediction(file) {
  const formData = new FormData();
  formData.append("file", file);
  const token = getStoredToken();

  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}/admin/run-prediction`, {
    method: "POST",
    headers,
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Prediction execution failed");
  }

  return response.json();
}
