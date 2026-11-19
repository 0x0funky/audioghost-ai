"use client";

import { AlertTriangle, Check, ChevronDown, Clock3, SlidersHorizontal, X } from "lucide-react";
import { useState, type RefObject } from "react";
import { formatDuration } from "@/lib/format";
import type { ModelGate, ModelSize, PendingPrompt, Quality, Region, SeparationSettings, Session, Track } from "@/lib/types";
import styles from "./SeparatePanel.module.css";
import { Button, Segmented, SectionLabel, Spinner, Switch } from "./ui";

const SUGGESTIONS = ["vocals", "speech", "drums", "bass", "guitar", "piano", "background music", "background noise"];

const MODELS: { value: ModelSize; label: string; hint: string; title: string }[] = [
  { value: "small", label: "Small", hint: "~4 GB", title: "Fastest, needs about 4 GB of GPU memory" },
  { value: "base", label: "Base", hint: "~5.5 GB", title: "Balanced, needs about 5.5 GB of GPU memory" },
  { value: "large", label: "Large", hint: "~9 GB", title: "Best, needs about 9 GB of GPU memory" },
];

const QUALITIES: { value: Quality; label: string; hint: string; title: string }[] = [
  { value: "fast", label: "Fast", hint: "8 steps", title: "About 2x faster than Balanced" },
  { value: "balanced", label: "Balanced", hint: "16 steps", title: "Meta's default" },
  { value: "high", label: "High", hint: "32 steps", title: "About 2x slower than Balanced" },
];

function statusBadge(track: Track): string | null {
  if (track.status === "running") return `${Math.round(track.progress * 100)}%`;
  if (track.status === "queued") return "queued";
  if (track.status === "failed") return "failed";
  return null;
}

export function SeparatePanel({
  session,
  colors,
  pending,
  activeKey,
  regions,
  settings,
  onSettings,
  onAddPrompt,
  onRemovePrompt,
  onActivate,
  onRemoveTrack,
  onSeparate,
  submitting,
  blocker,
  gate,
  onLicense,
  onConnect,
  inputRef,
  estimate,
}: {
  session: Session;
  colors: Record<string, string>;
  pending: PendingPrompt[];
  activeKey: string | null;
  regions: Region[];
  settings: SeparationSettings;
  onSettings: (settings: SeparationSettings) => void;
  onAddPrompt: (text: string) => void;
  onRemovePrompt: (key: string) => void;
  onActivate: (key: string) => void;
  onRemoveTrack: (track: Track) => void;
  /** ``typed``: text still in the input (not added with Enter yet). Resolves true once queued. */
  onSeparate: (typed?: string) => Promise<boolean>;
  submitting: boolean;
  blocker: string | null;
  gate: ModelGate;
  onLicense: () => void;
  onConnect: () => void;
  inputRef: RefObject<HTMLInputElement | null>;
  estimate: string | null;
}) {
  const [text, setText] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [confirm, setConfirm] = useState<string | null>(null);

  const used = new Set([...session.tracks.map((t) => t.prompt.toLowerCase()), ...pending.map((p) => p.text.toLowerCase())]);
  const suggestions = SUGGESTIONS.filter((s) => !used.has(s));
  const set = (patch: Partial<SeparationSettings>) => onSettings({ ...settings, ...patch });

  const add = () => {
    const value = text.trim().replace(/\s+/g, " ");
    if (!value) return;
    onAddPrompt(value);
    setText("");
  };

  const removeDone = (track: Track) => {
    if (confirm !== track.id) {
      setConfirm(track.id);
      setTimeout(() => setConfirm((current) => (current === track.id ? null : current)), 4000);
      return;
    }
    setConfirm(null);
    onRemoveTrack(track);
  };

  const advancedSummary = [
    settings.stereo ? "Stereo" : "Mono",
    settings.variant === "tv" ? "precision checkpoint" : "standard checkpoint",
    settings.use_float32 ? "full FP32" : null,
    `${settings.chunk_duration} s chunks`,
    settings.cleanup ? "clean up leftovers" : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const typed = text.trim().replace(/\s+/g, " ");
  const queued = pending.length + (typed ? 1 : 0);
  const label =
    queued === 0
      ? "Separate"
      : queued === 1
        ? `Separate ${pending[0]?.text ?? typed}`
        : `Separate ${queued} sounds`;
  const separate = async () => {
    // Keep the typed text if the request fails, so it needn't be typed again.
    if (await onSeparate(typed || undefined)) setText("");
  };
  const visibleTracks = session.tracks;

  return (
    <aside className={styles.panel} aria-label="Separate sounds">
      <div className={styles.head}>
        <h2 className={styles.title}>Separate sounds</h2>
        <span className={styles.muted}>Runs on this PC</span>
      </div>

      <div className={styles.body}>
        <div className={styles.group}>
          <label htmlFor="sound-input" className={styles.label}>
            What should be pulled out?
          </label>
          <div className={styles.tags}>
            {visibleTracks.map((track) => {
              const badge = statusBadge(track);
              const isDone = track.status === "done";
              return (
                <span key={track.id} className={styles.chip}>
                  <span className={styles.chipDot} style={{ background: colors[track.id] }} aria-hidden="true" />
                  <span className={styles.chipText}>{track.prompt}</span>
                  {badge && (
                    <span className={`${styles.chipBadge} ${track.status === "failed" ? styles.chipFailed : ""} tabular`}>
                      {badge}
                    </span>
                  )}
                  {isDone && confirm === track.id ? (
                    <button type="button" className={`${styles.chipRemove} ${styles.confirm}`} onClick={() => removeDone(track)}>
                      Remove?
                    </button>
                  ) : (
                    <button
                      type="button"
                      className={styles.chipRemove}
                      aria-label={
                        track.status === "running"
                          ? `Cancel ${track.prompt}`
                          : isDone
                            ? `Remove the ${track.prompt} track`
                            : `Remove ${track.prompt}`
                      }
                      onClick={() => (isDone ? removeDone(track) : onRemoveTrack(track))}
                    >
                      <X size={13} aria-hidden="true" />
                    </button>
                  )}
                </span>
              );
            })}
            {pending.map((prompt) => {
              const count = regions.filter((r) => r.promptKey === prompt.key).length;
              return (
                <span key={prompt.key} className={`${styles.chip} ${styles.pendingChip}`} data-active={prompt.key === activeKey}>
                  <button
                    type="button"
                    className={styles.chipMain}
                    aria-pressed={prompt.key === activeKey}
                    title="Regions you drag on the timeline are marked for this sound"
                    onClick={() => onActivate(prompt.key)}
                  >
                    <span className={styles.chipText}>{prompt.text}</span>
                    <span className={styles.chipBadge}>{count ? `${count} region${count > 1 ? "s" : ""}` : "new"}</span>
                  </button>
                  <button
                    type="button"
                    className={styles.chipRemove}
                    aria-label={`Remove ${prompt.text}`}
                    onClick={() => onRemovePrompt(prompt.key)}
                  >
                    <X size={13} aria-hidden="true" />
                  </button>
                </span>
              );
            })}
            <input
              id="sound-input"
              ref={inputRef}
              className={styles.input}
              type="text"
              placeholder={pending.length || visibleTracks.length ? "Add a sound" : "e.g. vocals, a dog barking"}
              value={text}
              maxLength={200}
              autoComplete="off"
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  add();
                } else if (event.key === "Backspace" && !text && pending.length) {
                  onRemovePrompt(pending[pending.length - 1].key);
                }
              }}
            />
          </div>
          {suggestions.length > 0 && (
            <div className={styles.suggestions} role="group" aria-label="Suggestions">
              {suggestions.slice(0, 6).map((s) => (
                <button key={s} type="button" className={styles.suggestion} onClick={() => onAddPrompt(s)}>
                  + {s}
                </button>
              ))}
            </div>
          )}
          <p className={styles.hint}>
            Every sound gets its own track. Mute a track to remove that sound, solo it to keep only that sound.
          </p>
        </div>

        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>Model</legend>
          <Segmented label="Model" options={MODELS} value={settings.model_size} onChange={(model_size) => set({ model_size })} />
        </fieldset>

        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>Quality</legend>
          <Segmented label="Quality" options={QUALITIES} value={settings.quality} onChange={(quality) => set({ quality })} />
        </fieldset>

        <div>
          <button
            type="button"
            className={styles.disclosure}
            aria-expanded={advanced}
            aria-controls="advanced-options"
            onClick={() => setAdvanced((value) => !value)}
          >
            <SlidersHorizontal size={16} aria-hidden="true" />
            <span className={styles.disclosureText}>
              <span className={styles.label}>Advanced</span>
              <span className={styles.muted}>{advancedSummary}</span>
            </span>
            <ChevronDown size={16} className={styles.chevron} aria-hidden="true" />
          </button>
          {advanced && (
            <div id="advanced-options" className={styles.advanced}>
              <Switch
                checked={settings.stereo}
                onChange={(stereo) => set({ stereo })}
                title="Keep stereo"
                hint="Separates left and right separately (about 2x the time). Mono files stay mono."
              />
              <Switch
                checked={settings.variant === "tv"}
                onChange={(on) => set({ variant: on ? "tv" : "default" })}
                title="Precision checkpoint (-tv)"
                hint="Meta's -tv checkpoints pick the right sound more reliably. Separate download."
              />
              <Switch
                checked={settings.use_float32}
                onChange={(use_float32) => set({ use_float32 })}
                title="Full FP32 transformer"
                hint="More GPU memory and slower; the default already keeps the sensitive math in FP32."
              />
              <Switch
                checked={settings.cleanup}
                onChange={(cleanup) => set({ cleanup })}
                title="Clean up leftovers"
                hint="Uses each separated sound to clear its traces from Everything else. A sound that overlaps it can come out a little thinner."
              />
              <div className={styles.range}>
                <label htmlFor="chunk-length" className={styles.rangeHead}>
                  <span>Max chunk length</span>
                  <span className={`${styles.rangeValue} tabular`}>{settings.chunk_duration} s</span>
                </label>
                <input
                  id="chunk-length"
                  type="range"
                  min={5}
                  max={60}
                  step={5}
                  value={settings.chunk_duration}
                  onChange={(event) => set({ chunk_duration: Number(event.target.value) })}
                />
                <span className={styles.hint}>
                  Long files are split into overlapping, crossfaded chunks. Shorter chunks use less GPU memory.
                </span>
              </div>
            </div>
          )}
        </div>

        {gate === "license" ? (
          <div className={styles.callout}>
            The {settings.model_size} model downloads the first time you separate, with no Hugging Face account. It comes
            with Meta&apos;s SAM License.
            <Button variant="primary" onClick={onLicense}>
              Review the license
            </Button>
          </div>
        ) : gate === "token" ? (
          <div className={styles.callout}>
            Meta&apos;s precision checkpoints need a Hugging Face account with access to them. The standard ones don&apos;t.
            <Button variant="primary" onClick={onConnect}>
              Connect Hugging Face
            </Button>
          </div>
        ) : (
          <div className={styles.submit}>
            <Button
              variant="primary"
              size="large"
              disabled={queued === 0 || submitting || !!blocker}
              onClick={separate}
            >
              {submitting ? "Starting..." : label}
            </Button>
            <p className={styles.estimate} aria-live="polite">
              {blocker ?? (queued === 0 ? "Type a sound above or pick a suggestion." : estimate ?? "Runs on this PC")}
            </p>
          </div>
        )}
      </div>

      {visibleTracks.length > 0 && (
        <div className={styles.activity}>
          <SectionLabel>Activity</SectionLabel>
          <ul className={styles.activityList}>
            {visibleTracks.slice(-6).map((track) => (
              <li key={track.id} className={styles.activityItem}>
                {track.status === "done" && <Check size={16} className={styles.ok} aria-hidden="true" />}
                {track.status === "failed" && <AlertTriangle size={16} className={styles.bad} aria-hidden="true" />}
                {track.status === "queued" && <Clock3 size={16} className={styles.wait} aria-hidden="true" />}
                {track.status === "running" && (
                  <span className={styles.run}>
                    <Spinner size={16} />
                  </span>
                )}
                <span className={styles.activityName}>{track.prompt}</span>
                <span className={`${styles.activityMeta} tabular`}>
                  {track.status === "done"
                    ? `Done in ${formatDuration(track.elapsed ?? 0)}`
                    : track.status === "running"
                      ? `${Math.round(track.progress * 100)}%`
                      : track.status === "queued"
                        ? "Waiting"
                        : "Failed"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  );
}
