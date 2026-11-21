"use client";

import { AlertTriangle, ArrowLeft, Check, Clock3, Download, FileAudio, Film, Plus, Trash2 } from "lucide-react";
import { useRef, useState } from "react";
import { api, downloadBatch, downloadBatchFile, uploadBatchFile } from "@/lib/api";
import { formatTime, relativeDay } from "@/lib/format";
import { useBatch } from "@/lib/hooks";
import type { Batch, BatchItem } from "@/lib/types";
import styles from "./Batch.module.css";
import { BATCH_ACCEPT } from "./BatchDialog";
import { Button, Dialog, ErrorText, ProgressBar, Spinner } from "./ui";

const MODEL_LABEL: Record<string, string> = { small: "Small", base: "Base", large: "Large" };
const QUALITY_LABEL: Record<string, string> = { fast: "Fast", balanced: "Balanced", high: "High" };
const KIND_LABEL: Record<string, string> = { mix: "one file each", tracks: "a ZIP of tracks each", video: "a video each" };

const level = (gain: number) => (gain <= 0 ? "removed" : Math.abs(gain - 1) < 1e-3 ? "kept" : `at ${Math.round(gain * 100)}%`);

export function recipeLine(batch: Batch): string {
  const recipe = batch.recipe;
  const sounds = recipe.prompts.map((p) => `${p.text} ${level(p.gain)}`);
  if (Math.abs(recipe.rest_gain - 1) > 1e-3) sounds.push(`everything else ${level(recipe.rest_gain)}`);
  return sounds.join(", ");
}

function StatusCell({ item }: { item: BatchItem }) {
  if (item.status === "done") {
    return (
      <span className={`${styles.state} ${styles.good}`}>
        <Check size={15} aria-hidden="true" /> Done
      </span>
    );
  }
  if (item.status === "failed" || item.status === "missing") {
    return (
      <span className={`${styles.state} ${styles.bad}`} title={item.error ?? undefined}>
        <AlertTriangle size={15} aria-hidden="true" /> {item.error || "Failed"}
      </span>
    );
  }
  if (item.status === "running") {
    return (
      <span className={styles.running}>
        <span className={styles.state}>
          <Spinner size={14} /> {item.message || "Separating"} · {Math.round(item.progress * 100)}%
        </span>
        <ProgressBar value={item.progress} label={`${item.filename} progress`} />
      </span>
    );
  }
  return (
    <span className={styles.state}>
      <Clock3 size={15} aria-hidden="true" /> Waiting
    </span>
  );
}

/** The page of one batch: every file with its progress, and the results to download. */
export function BatchView({ id, onHome, onOpenSession }: { id: string; onHome: () => void; onOpenSession: (id: string) => void }) {
  const { batch, error, refresh, setBatch } = useBatch(id);
  const [busy, setBusy] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ index: number; count: number; fraction: number } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [withFiles, setWithFiles] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  if (!batch) {
    return (
      <div className={styles.page}>
        <div className={styles.center} aria-live="polite">
          {error?.status === 404 ? (
            <>
              <p>This batch no longer exists.</p>
              <Button variant="primary" onClick={onHome}>
                Back to the start
              </Button>
            </>
          ) : error ? (
            <p>{error.message}</p>
          ) : (
            <Spinner size={22} />
          )}
        </div>
      </div>
    );
  }

  const items = batch.items ?? [];
  const doneCount = batch.counts.done ?? 0;
  const active = (batch.counts.queued ?? 0) + (batch.counts.running ?? 0);
  const failed = (batch.counts.failed ?? 0) + (batch.counts.missing ?? 0);
  const recipe = batch.recipe;

  const run = async (key: string, action: () => Promise<unknown>) => {
    setBusy(key);
    setProblem(null);
    try {
      await action();
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const add = async (files: File[]) => {
    setProblem(null);
    const failures: string[] = [];
    for (let index = 0; index < files.length; index++) {
      setUpload({ index, count: files.length, fraction: 0 });
      try {
        setBatch(await uploadBatchFile(id, files[index], (fraction) => setUpload({ index, count: files.length, fraction })).promise);
      } catch (err) {
        failures.push(`${files[index].name}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    setUpload(null);
    if (failures.length) setProblem(failures.join(" "));
    refresh();
  };

  return (
    <div className={styles.page}>
      <div className={styles.inner}>
        <button type="button" className={styles.back} onClick={onHome}>
          <ArrowLeft size={15} aria-hidden="true" /> Start
        </button>
        <header className={styles.header}>
          <div className={styles.headerText}>
            <h1 className={styles.title}>{batch.name || "Batch"}</h1>
            <p className={styles.lead}>
              {recipeLine(batch)}. {MODEL_LABEL[recipe.settings.model_size]} model, {QUALITY_LABEL[recipe.settings.quality]}{" "}
              quality; {KIND_LABEL[recipe.kind]}
              {recipe.kind !== "video" ? ` (${recipe.format.toUpperCase()})` : ""}.
            </p>
            <p className={styles.counts} aria-live="polite">
              {doneCount} of {items.length} done
              {active ? ` · ${active} to go` : ""}
              {failed ? ` · ${failed} failed` : ""} · started {relativeDay(batch.created_at)}
            </p>
          </div>
          <div className={styles.headerActions}>
            <Button
              variant="primary"
              disabled={!doneCount || busy !== null}
              onClick={() => run("all", () => downloadBatch(id))}
            >
              <Download size={16} aria-hidden="true" />
              {busy === "all" ? "Preparing..." : `Download ${doneCount === items.length ? "all" : `the ${doneCount} done`} (.zip)`}
            </Button>
            <Button disabled={upload !== null} onClick={() => input.current?.click()}>
              <Plus size={16} aria-hidden="true" />
              Add files
            </Button>
            <Button danger onClick={() => setDeleting(true)}>
              <Trash2 size={16} aria-hidden="true" />
              Delete
            </Button>
            <input
              ref={input}
              type="file"
              multiple
              accept={BATCH_ACCEPT}
              className="visually-hidden"
              tabIndex={-1}
              aria-label="Add audio or video files to this batch"
              onChange={(event) => {
                const files = [...(event.target.files ?? [])];
                event.target.value = "";
                if (files.length) void add(files);
              }}
            />
          </div>
        </header>

        {upload && (
          <div className={styles.uploading} aria-live="polite">
            <span>
              Uploading {upload.index + 1} of {upload.count}
            </span>
            <ProgressBar value={(upload.index + upload.fraction) / upload.count} label="Upload progress" />
          </div>
        )}
        {problem && <ErrorText>{problem}</ErrorText>}

        {items.length === 0 ? (
          <div className={styles.empty}>No files yet. Add the files this recipe should run on.</div>
        ) : (
          <ul className={styles.items}>
            {items.map((item) => (
              <li key={item.session} className={styles.item}>
                {item.is_video ? (
                  <Film size={18} className={styles.fileIcon} aria-hidden="true" />
                ) : (
                  <FileAudio size={18} className={styles.fileIcon} aria-hidden="true" />
                )}
                <span className={styles.itemName} title={item.filename}>
                  {item.filename}
                  {item.duration ? <span className={`${styles.itemMeta} tabular`}> · {formatTime(item.duration)}</span> : null}
                </span>
                <span className={styles.itemState}>
                  <StatusCell item={item} />
                </span>
                <span className={styles.itemActions}>
                  {item.status !== "missing" && (
                    <Button size="small" onClick={() => onOpenSession(item.session)} aria-label={`Open ${item.filename}`}>
                      Open
                    </Button>
                  )}
                  <Button
                    size="small"
                    disabled={item.status !== "done" || busy !== null}
                    onClick={() => run(item.session, () => downloadBatchFile(id, item.session))}
                    aria-label={`Download the result of ${item.filename}`}
                  >
                    <Download size={14} aria-hidden="true" />
                    {busy === item.session ? "Preparing..." : "Download"}
                  </Button>
                </span>
              </li>
            ))}
          </ul>
        )}
        <p className={styles.note}>
          Files are separated one after another, also while this page is closed. Open a file to listen, adjust its mix
          or try a sound again; its download here always follows this batch&apos;s recipe.
        </p>
      </div>

      <Dialog
        open={deleting}
        onClose={() => setDeleting(false)}
        title="Delete this batch?"
        footer={
          <>
            <Button onClick={() => setDeleting(false)}>Cancel</Button>
            <Button
              danger
              onClick={() => {
                setDeleting(false);
                void api
                  .deleteBatch(id, withFiles)
                  .then(onHome)
                  .catch((err) => setProblem(err instanceof Error ? err.message : String(err)));
              }}
            >
              Delete the batch
            </Button>
          </>
        }
      >
        <p className={styles.dialogText}>The list is deleted. Its files stay under Projects unless you delete them too.</p>
        <label className={styles.checkRow}>
          <input type="checkbox" checked={withFiles} onChange={(event) => setWithFiles(event.target.checked)} />
          Also delete its {items.length} project{items.length === 1 ? "" : "s"} and their audio
        </label>
      </Dialog>
    </div>
  );
}
