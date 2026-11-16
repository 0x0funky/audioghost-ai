// Central place for backend URLs. Override with NEXT_PUBLIC_API_BASE if the
// API runs somewhere other than http://localhost:8000.
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

export const apiUrl = (path: string) => `${API_BASE}${path}`;

export type DownloadFormat = "wav" | "flac" | "mp3";
export type StemId = "original" | "ghost" | "clean";

export const DOWNLOAD_FORMATS: { value: DownloadFormat; label: string; hint: string }[] = [
    { value: "wav", label: "WAV", hint: "24-bit, lossless" },
    { value: "flac", label: "FLAC", hint: "lossless, smaller" },
    { value: "mp3", label: "MP3", hint: "320 kbps" },
];

/** Working file streamed to the in-app players. */
export const stemUrl = (taskId: string, stem: StemId | "video") =>
    apiUrl(`/api/tasks/${taskId}/download/${stem}`);

/** Attachment with a readable name, e.g. "song [vocals].wav". */
export const stemDownloadUrl = (taskId: string, stem: StemId, format: DownloadFormat) =>
    `${stemUrl(taskId, stem)}?format=${format}`;

export const downloadAllUrl = (taskId: string, format: DownloadFormat) =>
    apiUrl(`/api/tasks/${taskId}/download-all?format=${format}`);

export const videoWithAudioUrl = (taskId: string, stem: StemId) =>
    apiUrl(`/api/tasks/${taskId}/download-video-with-audio/${stem}`);

/** Start a download; the server's Content-Disposition decides the file name. */
export function triggerDownload(url: string) {
    const link = document.createElement("a");
    link.href = url;
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    link.remove();
}

export const QUALITY_LABELS: Record<string, string> = {
    fast: "Fast",
    balanced: "Balanced",
    high: "High",
};

/** localStorage helpers that never throw (private mode, blocked storage...). */
export function loadPref<T>(key: string, fallback: T): T {
    try {
        const raw = window.localStorage.getItem(`audioghost:${key}`);
        return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
    } catch {
        return fallback;
    }
}

export function savePref(key: string, value: unknown) {
    try {
        window.localStorage.setItem(`audioghost:${key}`, JSON.stringify(value));
    } catch {
        /* ignore */
    }
}

export function loadString(key: string, fallback: string): string {
    try {
        return window.localStorage.getItem(`audioghost:${key}`) || fallback;
    } catch {
        return fallback;
    }
}

export function saveString(key: string, value: string) {
    try {
        window.localStorage.setItem(`audioghost:${key}`, value);
    } catch {
        /* ignore */
    }
}
