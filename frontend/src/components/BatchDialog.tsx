"use client";

import { FileAudio, Film, Layers, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { api, uploadBatchFile } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import type { ExportFormat, SeparationSettings, Session, Sound } from "@/lib/types";
import styles from "./Batch.module.css";
import exportStyles from "./ExportDialog.module.css";
import { Button, Dialog, ErrorText, IconButton, ProgressBar } from "./ui";

export const BATCH_ACCEPT =
  "audio/*,video/*,.mp3,.wav,.flac,.m4a,.ogg,.opus,.aac,.mp4,.mov,.mkv,.webm,.avi,.m4v,.mpeg";

const MODEL_LABEL: Record<string, string> = { small: "Small", base: "Base", large: "Large" };
const QUALITY_LABEL: Record<string, string> = { fast: "Fast", balanced: "Balanced", high: "High" };
const FORMATS: { value: ExportFormat; label: string; hint: string }[] = [
  { value: "wav", label: "WAV", hint: "24-bit, lossless" },
  { value: "flac", label: "FLAC", hint: "Lossless, smaller" },
  { value: "mp3", label: "MP3", hint: "320 kbps" },
];

const level = (gain: number) => (gain <= 0 ? "removed" : Math.abs(gain - 1) < 1e-3 ? "as recorded" : `at ${Math.round(gain * 100)}%`);
const isVideoFile = (file: File) => file.type.startsWith("video/") || /\.(mp4|mov|mkv|webm|avi|m4v|mpeg)$/i.test(file.name);

/** Files picked or dropped, with the ones already in the list left out. */
export function addFiles(current: File[], added: FileList | File[]): File[] {
  const seen = new Set(current.map((f) => `${f.name}:${f.size}`));
  const fresh = [...added].filter((f) => !seen.has(`${f.name}:${f.size}`));
  return [...current, ...fresh];
}

/**
 * "Batch": the sounds of this session, with this mix, for more files. Creates
 * the batch, uploads the files one after another (each is queued as soon as it
 * is up) and opens the batch's page.
 */
export function BatchDialog({
  session,
  sounds,
  gains,
  settings,
  summary,
  onClose,
  onCreated,
}: {
  session: Session;
  sounds: Sound[];
  /** Linear gain per sound id and "rest", as heard (without "only here" parts). */
  gains: Record<string, number>;
  settings: SeparationSettings;
  summary: string;
  onClose: () => void;
  onCreated: (batchId: string) => void;
}) {
  const done = sounds.filter((sound) => sound.take);
  const [files, setFiles] = useState<File[]>([]);
  const [kind, setKind] = useState<"mix" | "tracks" | "video">("mix");
  const [format, setFormat] = useState<ExportFormat>("wav");
  const [name, setName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<{ index: number; fraction: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const abort = useRef<(() => void) | null>(null);
  const busy = progress !== null;
  const anyVideo = files.some(isVideoFile) || session.source.is_video;

  const start = async () => {
    setError(null);
    setProgress({ index: 0, fraction: 0 });
    let batchId: string | null = null;
    const failed: string[] = [];
    try {
      const batch = await api.createBatch({
        ...settings,
        name: name.trim(),
        prompts: done.map((sound) => ({ text: sound.prompt, gain: gains[sound.id] ?? 1, seed: null })),
        rest_gain: gains.rest ?? 1,
        kind,
        format,
      });
      batchId = batch.id;
      for (let index = 0; index < files.length; index++) {
        setProgress({ index, fraction: 0 });
        const job = uploadBatchFile(batch.id, files[index], (fraction) => setProgress({ index, fraction }));
        abort.current = job.abort;
        try {
          await job.promise;
        } catch (err) {
          if ((err as { status?: number })?.status === -1) throw err; // cancelled: stop here
          failed.push(`${files[index].name}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      if (failed.length === files.length) throw new Error(failed[0]);
      onCreated(batch.id);
    } catch (err) {
      if ((err as { status?: number })?.status === -1 && batchId) {
        onCreated(batchId); // what was uploaded before the cancel is already queued
        return;
      }
      setError(err instanceof Error ? err.message : String(err));
      setProgress(null);
    } finally {
      abort.current = null;
    }
  };

  return (
    <Dialog
      open
      onClose={() => {
        if (!busy) onClose();
      }}
      title="Batch"
      subtitle="The same sounds and mix, for more files"
      footer={
        busy ? (
          <Button onClick={() => abort.current?.()}>Stop uploading</Button>
        ) : (
          <>
            <Button onClick={onClose}>Cancel</Button>
            <Button variant="primary" disabled={!files.length} onClick={start}>
              <Layers size={16} aria-hidden="true" />
              {files.length ? `Start ${files.length} file${files.length > 1 ? "s" : ""}` : "Start"}
            </Button>
          </>
        )
      }
    >
      <section className={styles.recipe} aria-label="What happens to every file">
        <div className={styles.recipeTitle}>Every file gets</div>
        <ul className={styles.recipeList}>
          {done.map((sound) => (
            <li key={sound.id}>
              <strong>{sound.prompt}</strong> separated, {level(gains[sound.id] ?? 1)} in the result
            </li>
          ))}
          <li>
            <strong>everything else</strong> {level(gains.rest ?? 1)}
          </li>
        </ul>
        <div className={styles.recipeMeta}>
          {MODEL_LABEL[settings.model_size]} model · {QUALITY_LABEL[settings.quality]} quality
          {settings.cleanup ? " · clean up leftovers" : ""} · now: {summary.charAt(0).toLowerCase() + summary.slice(1)}
        </div>
        <div className={styles.recipeMeta}>
          Hints and changes made in one part only belong to this file: they are not carried over.
        </div>
      </section>

      <fieldset className={exportStyles.fieldset}>
        <legend className={exportStyles.legend}>Result per file</legend>
        <div className={exportStyles.formats}>
          {(
            [
              { value: "mix", label: "One file", hint: "The mix above" },
              { value: "tracks", label: "Tracks", hint: "A ZIP per file" },
              { value: "video", label: "Video", hint: "Picture + the mix" },
            ] as const
          ).map((option) => (
            <label
              key={option.value}
              className={exportStyles.format}
              data-checked={kind === option.value}
              data-disabled={option.value === "video" && !anyVideo}
            >
              <input
                type="radio"
                name="batch-kind"
                checked={kind === option.value}
                disabled={option.value === "video" && !anyVideo}
                onChange={() => setKind(option.value)}
              />
              <span className={exportStyles.optionText}>
                <span className={exportStyles.optionTitle}>{option.label}</span>
                <span className={exportStyles.optionHint}>{option.hint}</span>
              </span>
            </label>
          ))}
        </div>
        {kind === "video" && <p className={styles.note}>Audio files in the batch are exported as one audio file instead.</p>}
      </fieldset>

      {kind !== "video" && (
        <fieldset className={exportStyles.fieldset}>
          <legend className={exportStyles.legend}>Format</legend>
          <div className={exportStyles.formats}>
            {FORMATS.map((f) => (
              <label key={f.value} className={exportStyles.format} data-checked={format === f.value}>
                <input type="radio" name="batch-format" checked={format === f.value} onChange={() => setFormat(f.value)} />
                <span className={exportStyles.optionText}>
                  <span className={exportStyles.optionTitle}>{f.label}</span>
                  <span className={exportStyles.optionHint}>{f.hint}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      )}

      <div className={styles.field}>
        <label htmlFor="batch-name" className={exportStyles.legend}>
          Name (optional)
        </label>
        <input
          id="batch-name"
          className={styles.textInput}
          type="text"
          maxLength={120}
          placeholder="e.g. Interviews, March"
          value={name}
          disabled={busy}
          onChange={(event) => setName(event.target.value)}
        />
      </div>

      <div
        className={`${styles.drop} ${dragging ? styles.dragging : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!busy && event.dataTransfer.files?.length) setFiles((current) => addFiles(current, event.dataTransfer.files));
        }}
      >
        <Upload size={20} aria-hidden="true" />
        <span>Drop audio or video files here</span>
        <Button size="small" disabled={busy} onClick={() => input.current?.click()}>
          Choose files
        </Button>
        <input
          ref={input}
          type="file"
          multiple
          accept={BATCH_ACCEPT}
          className="visually-hidden"
          tabIndex={-1}
          aria-label="Choose audio or video files"
          onChange={(event) => {
            if (event.target.files?.length) setFiles((current) => addFiles(current, event.target.files!));
            event.target.value = "";
          }}
        />
      </div>

      {files.length > 0 && (
        <ul className={styles.files}>
          {files.map((file, index) => {
            const state = !progress ? null : index < progress.index ? "done" : index === progress.index ? "now" : "next";
            return (
              <li key={`${file.name}:${file.size}`} className={styles.file}>
                {isVideoFile(file) ? (
                  <Film size={16} className={styles.fileIcon} aria-hidden="true" />
                ) : (
                  <FileAudio size={16} className={styles.fileIcon} aria-hidden="true" />
                )}
                <span className={styles.fileName} title={file.name}>
                  {file.name}
                </span>
                <span className={`${styles.fileMeta} tabular`}>
                  {state === "done" ? "Queued" : state === "now" ? `${Math.round(progress!.fraction * 100)}%` : formatBytes(file.size)}
                </span>
                {!busy && (
                  <IconButton
                    label={`Leave out ${file.name}`}
                    size="tiny"
                    onClick={() => setFiles((current) => current.filter((_, i) => i !== index))}
                  >
                    <X size={14} aria-hidden="true" />
                  </IconButton>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {progress && (
        <div className={styles.uploading} aria-live="polite">
          <span>
            Uploading {progress.index + 1} of {files.length}. Each file starts separating as soon as it is up.
          </span>
          <ProgressBar value={(progress.index + progress.fraction) / files.length} label="Upload progress" />
        </div>
      )}
      {error && <ErrorText>{error}</ErrorText>}
    </Dialog>
  );
}
