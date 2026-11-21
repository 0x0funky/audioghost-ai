import type {
  Anchor,
  AuthStatus,
  Batch,
  ExportFormat,
  LicenseInfo,
  Peaks,
  SeparationSettings,
  Session,
  SessionList,
  Span,
  SystemStatus,
} from "./types";

// Central place for backend URLs. Override with NEXT_PUBLIC_API_BASE if the
// API runs somewhere other than http://localhost:8000. "same-origin" (the
// desktop app, where the backend serves this UI itself) makes URLs relative.
const configuredBase = process.env.NEXT_PUBLIC_API_BASE;
export const API_BASE =
  configuredBase === "same-origin" ? "" : (configuredBase || "http://localhost:8000").replace(/\/$/, "");

export const apiUrl = (path: string) => (path.startsWith("http") ? path : `${API_BASE}${path}`);

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let message = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail) && body.detail[0]?.msg) message = body.detail[0].msg;
  } catch {
    // not JSON
  }
  return new ApiError(message, res.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch {
    throw new ApiError("Can't reach the AudioGhost backend on port 8000. Is start.bat (or start.sh) running?", 0);
  }
  if (!res.ok) throw await errorFrom(res);
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export interface SeparateRequest extends SeparationSettings {
  /** `retake_of`: a track of the sound this is another take of. */
  prompts: { text: string; anchors: Anchor[]; retake_of?: string }[];
  /** Separate only this part of the file. */
  range?: Span | null;
}

export interface RebuildRequest extends Omit<SeparationSettings, "cleanup"> {
  /** Sound ids, the first removed first. */
  order: string[];
}

export interface ExportRequest {
  kind: "mix" | "tracks" | "video";
  format: ExportFormat;
  /** Linear gain per id of the take that plays, and "rest". */
  gains: Record<string, number>;
  /** Parts of the file that play other gains ("only here"). */
  segments?: { start: number; end: number; gains: Record<string, number> }[];
  tracks: string[];
}

export interface BatchRequest extends SeparationSettings {
  name: string;
  prompts: { text: string; gain: number; seed: number | null }[];
  rest_gain: number;
  kind: "mix" | "tracks" | "video";
  format: ExportFormat;
}

export const api = {
  async system(): Promise<SystemStatus> {
    try {
      const data = await request<Omit<SystemStatus, "reachable">>("/api/system/status");
      return { ...data, reachable: true };
    } catch {
      return { reachable: false, redis: false, worker: false, worker_starting: false, busy: false, device: null };
    }
  },
  auth: () => request<AuthStatus>("/api/auth/status"),
  license: () => request<LicenseInfo>("/api/models/license"),
  acceptLicense: () => request<{ accepted: boolean; version: string }>("/api/models/license/accept", { method: "POST" }),
  login: (token: string) =>
    request<{ success: boolean; username: string }>("/api/auth/login", json("POST", { token })),
  sessions: () => request<SessionList>("/api/sessions"),
  session: (id: string) => request<Session>(`/api/sessions/${id}`),
  deleteSession: (id: string) => request<{ deleted: string }>(`/api/sessions/${id}`, { method: "DELETE" }),
  separate: (id: string, body: SeparateRequest) =>
    request<{ job_id: string; track_ids: string[]; session: Session }>(`/api/sessions/${id}/separate`, json("POST", body)),
  removeTrack: (id: string, trackId: string) =>
    request<{ removed?: string; cancelling?: string }>(`/api/sessions/${id}/tracks/${trackId}`, { method: "DELETE" }),
  activateTake: (id: string, trackId: string) =>
    request<Session>(`/api/sessions/${id}/tracks/${trackId}/activate`, { method: "POST" }),
  rebuildRest: (id: string, body: RebuildRequest) =>
    request<{ job_id: string; session: Session }>(`/api/sessions/${id}/rest/rebuild`, json("POST", body)),
  cancelRebuild: (id: string) =>
    request<{ cancelling: boolean; session: Session }>(`/api/sessions/${id}/rest/job`, { method: "DELETE" }),
  activateRest: (id: string, rev: number) =>
    request<Session>(`/api/sessions/${id}/rest/${rev}/activate`, { method: "POST" }),
  deleteRest: (id: string, rev: number) => request<Session>(`/api/sessions/${id}/rest/${rev}`, { method: "DELETE" }),
  peaks: (url: string) => request<Peaks>(url),
  batches: () => request<{ batches: Batch[] }>("/api/batches"),
  batch: (id: string) => request<Batch>(`/api/batches/${id}`),
  createBatch: (body: BatchRequest) => request<Batch>("/api/batches", json("POST", body)),
  deleteBatch: (id: string, withSessions: boolean) =>
    request<{ deleted: string }>(`/api/batches/${id}?sessions=${withSessions}`, { method: "DELETE" }),
};

/** Upload with progress (fetch can't report upload progress). */
function upload<T>(path: string, file: File, onProgress: (fraction: number) => void): { promise: Promise<T>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<T>((resolve, reject) => {
    xhr.open("POST", apiUrl(path));
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as T);
      else {
        const detail = (xhr.response as { detail?: unknown } | null)?.detail;
        reject(new ApiError(typeof detail === "string" ? detail : `Upload failed (${xhr.status})`, xhr.status));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError("Can't reach the AudioGhost backend on port 8000. Is start.bat (or start.sh) running?", 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled", -1));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}

export const uploadSession = (file: File, onProgress: (fraction: number) => void) =>
  upload<Session>("/api/sessions", file, onProgress);

/** Add a file to a batch: it is uploaded, becomes a session and is queued. */
export const uploadBatchFile = (batchId: string, file: File, onProgress: (fraction: number) => void) =>
  upload<Batch>(`/api/batches/${batchId}/items`, file, onProgress);

function filenameFrom(disposition: string | null, fallback: string): string {
  if (!disposition) return fallback;
  const star = /filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/.exec(disposition);
  if (star) {
    try {
      return decodeURIComponent(star[1].trim());
    } catch {
      // fall through
    }
  }
  const plain = /filename\s*=\s*"?([^";]+)"?/.exec(disposition);
  return plain ? plain[1].trim() : fallback;
}

/** Fetch a file the server builds and save it under the server's name. */
async function download(path: string, init: RequestInit | undefined, fallback: string): Promise<string> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), init);
  } catch {
    throw new ApiError("Can't reach the AudioGhost backend on port 8000.", 0);
  }
  if (!res.ok) throw await errorFrom(res);
  const blob = await res.blob();
  const name = filenameFrom(res.headers.get("Content-Disposition"), fallback);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
  return name;
}

/** POST an export and save the file under the server's name. */
export const downloadExport = (sessionId: string, body: ExportRequest) =>
  download(`/api/sessions/${sessionId}/export`, json("POST", body), `audioghost-export.${body.format}`);

/** One file of a batch, mixed and exported as the batch's recipe says. */
export const downloadBatchFile = (batchId: string, sessionId: string) =>
  download(`/api/batches/${batchId}/items/${sessionId}/export`, undefined, "audioghost-export");

/** Every finished file of a batch, in one ZIP. */
export const downloadBatch = (batchId: string) =>
  download(`/api/batches/${batchId}/export`, { method: "POST" }, "audioghost-batch.zip");
