export type TrackStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export type ModelSize = "small" | "base" | "large";
export type Quality = "fast" | "balanced" | "high";
export type Variant = "default" | "tv";
export type AnchorKind = "+" | "-";
export type Anchor = [AnchorKind, number, number];
export type ExportFormat = "wav" | "flac" | "mp3";
/** A part of the file, in seconds. */
export type Span = [number, number];

export interface SeparationSettings {
  model_size: ModelSize;
  quality: Quality;
  variant: Variant;
  stereo: boolean;
  use_float32: boolean;
  chunk_duration: number;
  /** "Clean up leftovers": each separated sound clears its traces from everything else. */
  cleanup: boolean;
  /** The same seed with the same settings repeats a result. null: a new random one every time. */
  seed: number | null;
}

/** One run of the model for a sound: a take. Several takes can belong to one sound. */
export interface Track {
  id: string;
  /** The sound this is a take of (tracks with the same value are takes of one sound). */
  sound: string;
  /** The finished take that plays, and is exported, for its sound. */
  active: boolean;
  prompt: string;
  anchors: Anchor[];
  /** Only this part of the file was separated; null: all of it. */
  range: Span | null;
  seed: number | null;
  status: TrackStatus;
  progress: number;
  message: string;
  settings: Omit<SeparationSettings, "seed"> & { seed?: number | null };
  file: string | null;
  elapsed: number | null;
  error: string | null;
  created_at: number;
  finished_at: number | null;
  audio_url: string | null;
  peaks_url: string | null;
  /** The model's own "everything but this sound" (played when only this track is changed). */
  without_url?: string | null;
  jobs_ahead: number | null;
}

/** A sound with its takes, as one lane of the timeline. */
export interface Sound {
  id: string;
  prompt: string;
  /** Finished takes, oldest first. */
  takes: Track[];
  /** The take that plays; null while the first one is still being made. */
  take: Track | null;
  /** Takes that are queued or running. */
  pending: Track[];
  /** Takes that failed (shown until dismissed). */
  failed: Track[];
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

export interface RestVersion {
  rev: number;
  seed: number | null;
  created_at: number | null;
  audio_url: string;
  peaks_url: string;
}

export interface RestInfo extends RestVersion {
  file: string;
  /** Sound ids, in the order the model removed them. */
  tracks: string[];
  /** Earlier versions, kept to compare with. */
  alts: RestVersion[];
  /** Made for other sounds or parts than the takes that play now. */
  stale: boolean;
}

/** A rebuild of "everything else" that is queued, running or has failed. */
export interface RestJob {
  status: TrackStatus;
  progress: number;
  message: string;
  error: string | null;
}

export interface Session {
  id: string;
  created_at: number;
  updated_at: number;
  expires_at: number | null;
  source: SessionSource;
  tracks: Track[];
  rest: RestInfo | null;
  rest_job: RestJob | null;
}

export interface SessionSummary {
  id: string;
  filename: string;
  is_video: boolean;
  duration: number;
  created_at: number;
  updated_at: number;
  expires_at: number | null;
  size_bytes: number;
  tracks: { prompt: string; status: TrackStatus }[];
}

export interface SessionList {
  sessions: SessionSummary[];
  total_bytes: number;
  folder: string;
  retention_hours: number;
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
  /** Ready on this computer: "facebook/sam-audio-<size>[-tv]" or "audioghostai/sam-audio-<size>". */
  downloaded_models: string[];
  /** Meta's SAM License, accepted once before the first download of AudioGhost's copies. */
  license_accepted?: boolean;
}

export interface LicenseInfo {
  version: string;
  url: string;
  accepted: boolean;
  text: string;
}

/** What stops a separation from downloading its model: the license, or (for -tv) a Hugging Face account. */
export type ModelGate = "license" | "token" | null;

export interface Peaks {
  duration: number;
  peaks: number[];
}

/** A sound typed in the panel but not sent yet. */
export interface PendingPrompt {
  key: string;
  text: string;
  /** Another take of this sound (its id), instead of a new sound. */
  sound?: string;
}

/** A hint for a sound that is not sent yet: where it can, or can't, be heard. */
export interface Region {
  id: string;
  promptKey: string;
  kind: AnchorKind;
  start: number;
  end: number;
}

/** The part of the timeline that is selected. */
export interface Selection {
  start: number;
  end: number;
  /** The lane the selection was dragged on: "original", "rest" or a sound id. */
  lane: string | null;
}

/** Per-track mixer state (UI only). */
export interface ChannelState {
  volume: number; // slider position 0..100 (75 = as recorded)
  muted: boolean;
  solo: boolean;
  /** "Only here": the change above applies inside these parts; everywhere when empty. */
  ranges?: Span[];
}

export type Mixer = Record<string, ChannelState>;

export interface BatchRecipe {
  prompts: { text: string; gain: number; seed: number | null }[];
  rest_gain: number;
  settings: SeparationSettings;
  kind: "mix" | "tracks" | "video";
  format: ExportFormat;
}

export interface BatchItem {
  session: string;
  filename: string;
  status: "queued" | "running" | "done" | "failed" | "missing";
  progress: number;
  message: string;
  error: string | null;
  duration?: number;
  is_video?: boolean;
}

export interface Batch {
  id: string;
  name: string;
  created_at: number;
  updated_at: number;
  recipe: BatchRecipe;
  counts: Record<string, number>;
  files: number;
  items?: BatchItem[];
}
