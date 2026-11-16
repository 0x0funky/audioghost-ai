export type TrackStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export type ModelSize = "small" | "base" | "large";
export type Quality = "fast" | "balanced" | "high";
export type Variant = "default" | "tv";
export type AnchorKind = "+" | "-";
export type Anchor = [AnchorKind, number, number];
export type ExportFormat = "wav" | "flac" | "mp3";

export interface SeparationSettings {
  model_size: ModelSize;
  quality: Quality;
  variant: Variant;
  stereo: boolean;
  use_float32: boolean;
  chunk_duration: number;
}

export interface Track {
  id: string;
  prompt: string;
  anchors: Anchor[];
  status: TrackStatus;
  progress: number;
  message: string;
  settings: SeparationSettings & { seed?: number | null };
  file: string | null;
  elapsed: number | null;
  error: string | null;
  created_at: number;
  finished_at: number | null;
  audio_url: string | null;
  peaks_url: string | null;
  jobs_ahead: number | null;
}

export interface SessionSource {
  filename: string;
  file: string;
  is_video: boolean;
  duration: number;
  frames: number;
  sample_rate: number;
  channels: number;
  audio_url: string;
  peaks_url: string;
  video_url: string | null;
}

export interface RestInfo {
  rev: number;
  file: string;
  tracks: string[];
  audio_url: string;
  peaks_url: string;
}

export interface Session {
  id: string;
  created_at: number;
  updated_at: number;
  expires_at: number | null;
  source: SessionSource;
  tracks: Track[];
  rest: RestInfo | null;
}

export interface SessionSummary {
  id: string;
  filename: string;
  is_video: boolean;
  duration: number;
  created_at: number;
  updated_at: number;
  expires_at: number | null;
  tracks: { prompt: string; status: TrackStatus }[];
}

export interface DeviceInfo {
  device?: string;
  gpu?: string;
  vram_total_gb?: number;
  vram_used_gb?: number;
  model?: string;
}

export interface SystemStatus {
  reachable: boolean;
  version?: string;
  redis: boolean;
  worker: boolean;
  worker_starting: boolean;
  busy: boolean;
  device: DeviceInfo | null;
}

export interface AuthStatus {
  authenticated: boolean;
  username: string | null;
  downloaded_models: string[];
}

export interface Peaks {
  duration: number;
  peaks: number[];
}

/** A sound typed in the panel but not sent yet, with the regions marked for it. */
export interface PendingPrompt {
  key: string;
  text: string;
}

export interface Region {
  id: string;
  promptKey: string;
  kind: AnchorKind;
  start: number;
  end: number;
}

/** Per-track mixer state (UI only). */
export interface ChannelState {
  volume: number; // slider position 0..100 (75 = 0 dB)
  muted: boolean;
  solo: boolean;
}
