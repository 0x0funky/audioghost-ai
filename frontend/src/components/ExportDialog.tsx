"use client";

import { Download } from "lucide-react";
import { useState } from "react";
import { downloadExport } from "@/lib/api";
import type { ExportFormat, Session, Sound } from "@/lib/types";
import styles from "./ExportDialog.module.css";
import { Button, Dialog, ErrorText } from "./ui";

export type ExportKind = "mix" | "tracks" | "video";

const FORMATS: { value: ExportFormat; label: string; hint: string }[] = [
  { value: "wav", label: "WAV", hint: "24-bit, lossless" },
  { value: "flac", label: "FLAC", hint: "Lossless, smaller" },
  { value: "mp3", label: "MP3", hint: "320 kbps" },
];

/** Same rule as the server: "vocals", "no vocals", "everything else" or "mix". */
export function mixLabel(session: Session, takes: { id: string; prompt: string }[], gains: Record<string, number>): string {
  const done = takes;
  const audible = done.filter((t) => (gains[t.id] ?? 0) > 0);
  const restOn = !!session.rest && (gains.rest ?? 0) > 0;
  const keys = [...audible.map((t) => t.id), ...(restOn ? ["rest"] : [])];
  const unity = keys.every((k) => Math.abs((gains[k] ?? 1) - 1) < 1e-3);
  if (unity && audible.length === 1 && !restOn) return audible[0].prompt;
  if (unity && restOn && done.length - audible.length === 1) {
    return `no ${done.find((t) => !audible.includes(t))!.prompt}`;
  }
  if (unity && restOn && audible.length === 0) return "everything else";
  return "mix";
}

const stem = (name: string) => name.replace(/\.[^.]+$/, "");
const ext = (name: string) => (/\.[^.]+$/.exec(name)?.[0] ?? ".mp4").toLowerCase();

export function ExportDialog({
  session,
  sounds,
  gains,
  segments,
  summary,
  open,
  initialKind,
  onClose,
}: {
  session: Session;
  sounds: Sound[];
  /** Linear gain per id of the take that plays, and "rest": what plays outside the parts below. */
  gains: Record<string, number>;
  /** Parts of the file that play other gains ("only here" changes). */
  segments: { start: number; end: number; gains: Record<string, number> }[];
  summary: string;
  open: boolean;
  initialKind: ExportKind;
  onClose: () => void;
}) {
  // One track per sound: the take that plays.
  const done = sounds.filter((sound) => sound.take).map((sound) => ({ id: sound.take!.id, prompt: sound.prompt }));
  const [kind, setKind] = useState<ExportKind>(initialKind);
  const [format, setFormat] = useState<ExportFormat>("wav");
  const [picked, setPicked] = useState<Set<string>>(() => new Set([...done.map((t) => t.id), "rest"]));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const silent = [gains, ...segments.map((s) => s.gains)].every((g) => Object.values(g).every((value) => value <= 0));
  const label = segments.length ? "mix" : mixLabel(session, done, gains);
  const base = stem(session.source.filename);
  const fileName =
    kind === "tracks"
      ? `${base} [tracks].zip`
      : kind === "video"
        ? `${base} [${label}]${ext(session.source.filename)}`
        : `${base} [${label}].${format}`;
  const chosen = [...done.map((t) => t.id), ...(session.rest ? ["rest"] : [])].filter((k) => picked.has(k));
  const disabled =
    busy || (kind === "tracks" ? chosen.length === 0 : silent) || (kind === "video" && !session.source.is_video);

  const toggle = (key: string) =>
    setPicked((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await downloadExport(session.id, { kind, format, gains, segments, tracks: chosen });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Export"
      subtitle={session.source.filename}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={disabled} onClick={run}>
            <Download size={16} aria-hidden="true" />
            {busy ? "Preparing..." : "Export"}
          </Button>
        </>
      }
    >
      <fieldset className={styles.fieldset}>
        <legend className={styles.legend}>What to export</legend>
        <label className={styles.option} data-checked={kind === "mix"}>
          <input type="radio" name="export-kind" checked={kind === "mix"} onChange={() => setKind("mix")} />
          <span className={styles.optionText}>
            <span className={styles.optionTitle}>What you hear</span>
            <span className={styles.optionHint}>
              {silent ? "Everything is removed right now." : `One file with your current mix: ${summary.charAt(0).toLowerCase() + summary.slice(1)}.`}
            </span>
          </span>
        </label>
        <div className={styles.option} data-checked={kind === "tracks"}>
          <label className={styles.optionLabel}>
            <input type="radio" name="export-kind" checked={kind === "tracks"} onChange={() => setKind("tracks")} />
            <span className={styles.optionText}>
              <span className={styles.optionTitle}>Separate tracks</span>
              <span className={styles.optionHint}>One file per track at full level, in a ZIP</span>
            </span>
          </label>
          {kind === "tracks" && (
            <div className={styles.checks}>
              {done.map((track) => (
                <label key={track.id} className={styles.check}>
                  <input type="checkbox" checked={picked.has(track.id)} onChange={() => toggle(track.id)} />
                  {track.prompt}
                </label>
              ))}
              {session.rest && (
                <label className={styles.check}>
                  <input type="checkbox" checked={picked.has("rest")} onChange={() => toggle("rest")} />
                  everything else
                </label>
              )}
            </div>
          )}
        </div>
        <label className={styles.option} data-checked={kind === "video"} data-disabled={!session.source.is_video}>
          <input
            type="radio"
            name="export-kind"
            checked={kind === "video"}
            disabled={!session.source.is_video}
            onChange={() => setKind("video")}
          />
          <span className={styles.optionText}>
            <span className={styles.optionTitle}>Video with this mix</span>
            <span className={styles.optionHint}>
              {session.source.is_video
                ? "The original picture, untouched, with your mix as its soundtrack"
                : "Available when the source is a video file"}
            </span>
          </span>
        </label>
      </fieldset>

      {kind !== "video" && (
        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>Format</legend>
          <div className={styles.formats}>
            {FORMATS.map((f) => (
              <label key={f.value} className={styles.format} data-checked={format === f.value}>
                <input type="radio" name="export-format" checked={format === f.value} onChange={() => setFormat(f.value)} />
                <span className={styles.optionText}>
                  <span className={styles.optionTitle}>{f.label}</span>
                  <span className={styles.optionHint}>{f.hint}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      )}

      <div className={styles.fileName}>
        <span className={styles.legend}>File name</span>
        <code className={styles.code}>{fileName}</code>
      </div>
      {error && <ErrorText>{error}</ErrorText>}
    </Dialog>
  );
}
