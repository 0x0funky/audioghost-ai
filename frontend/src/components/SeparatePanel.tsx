"use client";

import { AlertTriangle, Check, ChevronDown, Clock3, Dices, MapPin, Plus, SlidersHorizontal, X } from "lucide-react";
import { useState, type RefObject } from "react";
import { formatDuration, formatSpan, formatTime, isEnglish } from "@/lib/format";
import type {
  AnchorKind,
  ModelGate,
  ModelSize,
  PendingPrompt,
  Quality,
  Region,
  Selection,
  SeparationSettings,
  Session,
  Sound,
} from "@/lib/types";
import styles from "./SeparatePanel.module.css";
import { Button, IconButton, Segmented, SectionLabel, Spinner, Switch } from "./ui";

const SUGGESTIONS = ["vocals", "speech", "drums", "bass", "guitar", "piano", "background music", "background noise"];
/** The shortest part of a file that can be separated on its own (the server's rule). */
export const MIN_PART_SECONDS = 1;
const MAX_SEED = 2147483647;

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

export function SeparatePanel({
  session,
  sounds,
  pending,
  hints,
  marking,
  selection,
  partOnly,
  onPartOnly,
  settings,
  onSettings,
  onAddPrompt,
  onRemovePrompt,
  onRemoveHint,
  onMark,
  onHintFromSelection,
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
  sounds: Sound[];
  pending: PendingPrompt[];
  hints: Region[];
  /** The sound whose hints are being marked on the timeline. */
  marking: { key: string; kind: AnchorKind } | null;
  selection: Selection | null;
  /** Separate only the selected part of the file. */
  partOnly: boolean;
  onPartOnly: (on: boolean) => void;
  settings: SeparationSettings;
  onSettings: (settings: SeparationSettings) => void;
  /** Returns the new sound's key, or null when it was already in the list. */
  onAddPrompt: (text: string) => string | null;
  onRemovePrompt: (key: string) => void;
  onRemoveHint: (id: string) => void;
  /** Start marking hints for a sound on the timeline; null stops. */
  onMark: (key: string | null) => void;
  onHintFromSelection: (key: string, kind: AnchorKind) => void;
  /** ``typed``: text still in the input (not added yet). Resolves true once queued. */
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

  const used = new Set([...sounds.map((s) => s.prompt.toLowerCase()), ...pending.map((p) => p.text.toLowerCase())]);
  const suggestions = SUGGESTIONS.filter((s) => !used.has(s));
  const set = (patch: Partial<SeparationSettings>) => onSettings({ ...settings, ...patch });

  const typed = text.trim().replace(/\s+/g, " ");
  const english = isEnglish(typed);
  const add = () => {
    if (!typed || !english) return null;
    const key = onAddPrompt(typed);
    setText("");
    return key;
  };

  const advancedSummary = [
    settings.stereo ? "Stereo" : "Mono",
    settings.variant === "tv" ? "precision checkpoint" : "standard checkpoint",
    settings.use_float32 ? "full FP32" : null,
    `${settings.chunk_duration} s chunks`,
    settings.cleanup ? "clean up leftovers" : null,
    settings.seed !== null ? `seed ${settings.seed}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const queued = pending.length + (typed ? 1 : 0);
  const first = pending[0];
  const label =
    queued === 0
      ? "Separate"
      : queued === 1
        ? first?.sound
          ? `Try ${first.text} again`
          : `Separate ${first?.text ?? typed}`
        : `Separate ${queued} sounds`;
  const separate = async () => {
    // Keep the typed text if the request fails, so it needn't be typed again.
    if (await onSeparate(typed || undefined)) setText("");
  };

  const duration = session.source.duration;
  const selected = selection ? selection.end - selection.start : 0;
  const canPart = selected >= MIN_PART_SECONDS;
  const long = duration >= 90;

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
          <div className={styles.inputRow} data-invalid={!english || undefined}>
            <input
              id="sound-input"
              ref={inputRef}
              className={styles.input}
              type="text"
              placeholder={pending.length || sounds.length ? "Another sound, in English" : "e.g. vocals, a dog barking"}
              value={text}
              maxLength={200}
              autoComplete="off"
              spellCheck={false}
              aria-invalid={!english}
              aria-describedby={!english ? "sound-input-error" : undefined}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                  event.preventDefault();
                  if (english && queued > 0 && !submitting && !blocker && !gate) void separate();
                } else if (event.key === "Enter") {
                  event.preventDefault();
                  add();
                } else if (event.key === "Backspace" && !text && pending.length) {
                  onRemovePrompt(pending[pending.length - 1].key);
                }
              }}
            />
            <IconButton label="Add this sound to the list" size="small" disabled={!typed || !english} onClick={add}>
              <Plus size={16} aria-hidden="true" />
            </IconButton>
          </div>
          {!english && (
            <p id="sound-input-error" className={styles.inputError} role="alert">
              <AlertTriangle size={13} aria-hidden="true" />
              Describe the sound in English, for example &quot;a dog barking&quot;. The model only understands English.
            </p>
          )}

          {pending.length > 0 && (
            <ul className={styles.pendingList}>
              {pending.map((prompt) => {
                const own = hints.filter((hint) => hint.promptKey === prompt.key);
                const isMarking = marking?.key === prompt.key;
                return (
                  <li key={prompt.key} className={styles.pendingItem} data-marking={isMarking || undefined}>
                    <div className={styles.pendingTop}>
                      <span className={styles.pendingName} title={prompt.text}>
                        {prompt.text}
                      </span>
                      <span className={styles.chipBadge}>{prompt.sound ? "new take" : "new"}</span>
                      <IconButton label={`Remove ${prompt.text}`} size="tiny" onClick={() => onRemovePrompt(prompt.key)}>
                        <X size={14} aria-hidden="true" />
                      </IconButton>
                    </div>
                    {own.length > 0 && (
                      <div className={styles.hintChips}>
                        {own.map((hint) => (
                          <span key={hint.id} className={styles.hintChip} data-kind={hint.kind}>
                            <span className="tabular">
                              {hint.kind === "+" ? "here" : "not here"}: {formatSpan(hint.start, hint.end)}
                            </span>
                            <button
                              type="button"
                              className={styles.hintRemove}
                              aria-label={`Remove the hint at ${formatSpan(hint.start, hint.end)}`}
                              onClick={() => onRemoveHint(hint.id)}
                            >
                              <X size={11} aria-hidden="true" />
                            </button>
                          </span>
                        ))}
                      </div>
                    )}
                    <div className={styles.pendingActions}>
                      <button
                        type="button"
                        className={styles.linkButton}
                        aria-pressed={isMarking}
                        title="Optional: show the model where this sound can be heard, or where it can't"
                        onClick={() => onMark(isMarking ? null : prompt.key)}
                      >
                        <MapPin size={13} aria-hidden="true" />
                        {isMarking ? "Done marking" : own.length ? "Mark more" : "Mark where it is (optional)"}
                      </button>
                      {selection && !isMarking && (
                        <>
                          <button type="button" className={styles.linkButton} onClick={() => onHintFromSelection(prompt.key, "+")}>
                            It is in the selection
                          </button>
                          <button type="button" className={styles.linkButton} onClick={() => onHintFromSelection(prompt.key, "-")}>
                            Not in the selection
                          </button>
                        </>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}

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
            Every sound gets its own track. <strong>Remove</strong> takes it out of what you hear, <strong>Only</strong> keeps
            just that sound, and the slider sets how much of it stays.
          </p>
        </div>

        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>Part of the file</legend>
          <Segmented
            label="Part of the file"
            value={partOnly && canPart ? "part" : "whole"}
            onChange={(value) => onPartOnly(value === "part")}
            options={[
              { value: "whole", label: "Whole file", hint: formatTime(duration) },
              {
                value: "part",
                label: "Selected part",
                hint: selection ? (canPart ? formatSpan(selection.start, selection.end) : "at least 1 s") : "nothing selected",
                disabled: !canPart,
                title: canPart ? undefined : "Drag on the timeline to select a part first",
              },
            ]}
          />
          <p className={styles.hint}>
            {partOnly && canPart
              ? "Only this part goes through the model. The rest of the file stays exactly as it is."
              : long
                ? "A long file: drag on the timeline to select 10 to 30 seconds and try your description there first."
                : "Drag on the timeline to select a part, to separate a sound there only."}
          </p>
        </fieldset>

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
                <label htmlFor="seed" className={styles.rangeHead}>
                  <span>Seed</span>
                  <span className={styles.rangeValue}>{settings.seed === null ? "Random every time" : "Fixed"}</span>
                </label>
                <div className={styles.seedRow}>
                  <input
                    id="seed"
                    className={styles.seedInput}
                    type="text"
                    inputMode="numeric"
                    placeholder="Random"
                    autoComplete="off"
                    value={settings.seed === null ? "" : String(settings.seed)}
                    onChange={(event) => {
                      const digits = event.target.value.replace(/\D/g, "").slice(0, 10);
                      set({ seed: digits ? Math.min(MAX_SEED, Number(digits)) : null });
                    }}
                  />
                  <IconButton label="Back to a random seed" size="small" disabled={settings.seed === null} onClick={() => set({ seed: null })}>
                    <Dices size={16} aria-hidden="true" />
                  </IconButton>
                </div>
                <span className={styles.hint}>
                  The model starts from random noise, so two runs differ. The same seed with the same settings repeats a
                  result. Each track&apos;s menu shows the seed it used.
                </span>
              </div>
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
              disabled={queued === 0 || submitting || !!blocker || !english}
              onClick={separate}
            >
              {submitting ? "Starting..." : label}
            </Button>
            <p className={styles.estimate} aria-live="polite">
              {blocker ??
                (queued === 0
                  ? "Type a sound above or pick a suggestion."
                  : partOnly && canPart
                    ? `Only ${formatDuration(selected)} of the file`
                    : estimate ?? "Runs on this PC")}
            </p>
          </div>
        )}
      </div>

      {session.tracks.length > 0 && (
        <div className={styles.activity}>
          <SectionLabel>Activity</SectionLabel>
          <ul className={styles.activityList}>
            {session.tracks.slice(-6).map((track) => (
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
