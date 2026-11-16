import type { SeparationSettings } from "./types";

const PREFIX = "audioghost.";

export function loadPref<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    const value = JSON.parse(raw);
    if (fallback && typeof fallback === "object" && !Array.isArray(fallback)) {
      return { ...fallback, ...value };
    }
    return value as T;
  } catch {
    return fallback;
  }
}

export function savePref(key: string, value: unknown) {
  try {
    window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // private mode / storage full: preferences just don't persist
  }
}

export const DEFAULT_SETTINGS: SeparationSettings = {
  model_size: "base",
  quality: "balanced",
  variant: "default",
  stereo: false,
  use_float32: false,
  chunk_duration: 25,
};

export type Theme = "dark" | "light";

export function currentTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

export function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(PREFIX + "theme", theme);
  } catch {
    // ignore
  }
  window.dispatchEvent(new Event("audioghost-theme"));
}

export function subscribeTheme(listener: () => void) {
  window.addEventListener("audioghost-theme", listener);
  return () => window.removeEventListener("audioghost-theme", listener);
}
