/**
 * DocForensics AI — API Client
 * Connects directly to the FastAPI backend with structured error handling
 */

import { ForensicReport, HealthResponse } from "../types/forensic";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: "GET",
      headers: { 
        "Accept": "application/json",
        "Bypass-Tunnel-Reminder": "true"
      },
      cache: "no-store",
    });
    if (!res.ok) {
      throw new ApiError(`Health check failed (${res.status})`, res.status);
    }
    return await res.json();
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || "Failed to reach backend service", 0);
  }
}

export async function analyzeDocument(
  file: File,
  pageNumber: number = 1
): Promise<ForensicReport> {
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API_BASE_URL}/api/analyze?page_number=${pageNumber}`;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Bypass-Tunnel-Reminder": "true"
      },
      body: formData,
    });

    if (!res.ok) {
      let errorMsg = `Analysis failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) errorMsg = errorData.detail;
      } catch (_) {}
      throw new ApiError(errorMsg, res.status);
    }

    const data: ForensicReport = await res.json();
    return data;
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || "Network error during analysis", 0);
  }
}

export async function getAnalysis(sessionId: string): Promise<ForensicReport> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/analysis/${sessionId}`);
    if (!res.ok) {
      throw new ApiError(`Analysis session ${sessionId} not found`, res.status);
    }
    return await res.json();
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || "Failed to fetch analysis session", 0);
  }
}

export function getAssetUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? "" : "/"}${path}`;
}
