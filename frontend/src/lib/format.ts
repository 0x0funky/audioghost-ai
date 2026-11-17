/** 83.4 -> "1:23", with tenths -> "1:23.4"; hours when needed. */
export function formatTime(seconds: number, tenths = false): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const sec = tenths ? s.toFixed(1).padStart(4, "0") : String(Math.floor(s)).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

/** Slider position (0..100) -> dB. 0 = silent, 75 = 0 dB, 100 = +6 dB. */
export function sliderToDb(position: number): number {
  if (position <= 0) return -Infinity;
  if (position <= 75) return ((position - 75) * 40) / 75;
  return ((position - 75) * 6) / 25;
}

export function sliderToGain(position: number): number {
  const db = sliderToDb(position);
  return db === -Infinity ? 0 : 10 ** (db / 20);
}

export function formatDb(position: number): string {
  const db = sliderToDb(position);
  if (db === -Infinity) return "-inf dB";
  if (Math.abs(db) < 0.05) return "0.0 dB";
  return `${db > 0 ? "+" : "−"}${Math.abs(db).toFixed(1)} dB`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 1) return `${Math.max(0.1, seconds).toFixed(1)} s`;
  if (seconds < 90) return `${Math.round(seconds)} s`;
  return `${Math.round(seconds / 60)} min`;
}

export function relativeDay(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000);
  const now = new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (date.getTime() >= startOfToday) return `Today, ${time}`;
  if (date.getTime() >= startOfToday - 86_400_000) return `Yesterday, ${time}`;
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function expiresIn(epochSeconds: number | null): string | null {
  if (!epochSeconds) return null;
  const hours = (epochSeconds * 1000 - Date.now()) / 3_600_000;
  if (hours <= 0) return "Expiring";
  if (hours < 1) return `Expires in ${Math.max(1, Math.round(hours * 60))} min`;
  if (hours < 3) return `Expires in ${Math.round(hours)} h`;
  return null;
}
