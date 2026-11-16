"use client";

import { AlertTriangle, Check, FileAudio, Film, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { uploadSession } from "@/lib/api";
import { api } from "@/lib/api";
import { expiresIn, formatTime, relativeDay } from "@/lib/format";
import { useRecentSessions } from "@/lib/hooks";
import type { AuthStatus, Session, SystemStatus } from "@/lib/types";
import styles from "./StartScreen.module.css";
import { Button, ErrorText, IconButton, ProgressBar, SectionLabel } from "./ui";

const ACCEPT =
  "audio/*,video/*,.mp3,.wav,.flac,.m4a,.ogg,.opus,.aac,.mp4,.mov,.mkv,.webm,.avi,.m4v,.mpeg";

const MODELS = [
  { size: "small", label: "Small", gb: "5.1 GB" },
  { size: "base", label: "Base", gb: "7.7 GB" },
  { size: "large", label: "Large", gb: "14.9 GB" },
];

function Setup({ system, auth, onConnect }: { system: SystemStatus | null; auth: AuthStatus | null; onConnect: () => void }) {
  const downloaded = new Set(auth?.downloaded_models ?? []);
  const device = system?.device;
  const tvCount = MODELS.filter((m) => downloaded.has(`facebook/sam-audio-${m.size}-tv`)).length;
  const workerReady = system?.reachable && system.redis && system.worker && !system.worker_starting;
  return (
    <section className={styles.card} aria-labelledby="setup-heading">
      <h2 id="setup-heading" className={styles.h2}>
        This computer
      </h2>
      <div className={styles.check}>
        {workerReady ? (
          <Check size={18} className={`${styles.checkIcon} ${styles.good}`} aria-hidden="true" />
        ) : (
          <AlertTriangle size={18} className={`${styles.checkIcon} ${styles.attention}`} aria-hidden="true" />
        )}
        <div>
          <div className={styles.checkTitle}>
            {workerReady ? device?.gpu ?? "Ready" : system?.worker_starting ? "Worker starting..." : "Worker not ready"}
          </div>
          <div className={styles.checkHint}>
            {workerReady && device?.vram_total_gb
              ? `${device.vram_total_gb} GB video memory`
              : workerReady
                ? device?.device === "cpu"
                  ? "Running on the CPU (slow)"
                  : ""
                : "Separation starts once the worker is up; see the status in the top bar."}
          </div>
        </div>
      </div>
      <div className={styles.check}>
        {auth?.authenticated ? (
          <Check size={18} className={`${styles.checkIcon} ${styles.good}`} aria-hidden="true" />
        ) : (
          <AlertTriangle size={18} className={`${styles.checkIcon} ${styles.attention}`} aria-hidden="true" />
        )}
        <div className={styles.grow}>
          <div className={styles.checkTitle}>
            {auth?.authenticated ? "Hugging Face connected" : "Hugging Face not connected"}
          </div>
          <div className={styles.checkHint}>
            {auth?.authenticated
              ? `Signed in as ${auth.username ?? "your account"}`
              : "The SAM-Audio models are gated: request access, then connect a token."}
          </div>
        </div>
        {!auth?.authenticated && (
          <Button size="small" variant="primary" onClick={onConnect}>
            Connect
          </Button>
        )}
      </div>
      <div className={styles.models}>
        <SectionLabel>Models</SectionLabel>
        {MODELS.map((model) => {
          const have = downloaded.has(`facebook/sam-audio-${model.size}`);
          return (
            <div key={model.size} className={styles.modelRow}>
              <span className={styles.modelName}>{model.label}</span>
              <span className={styles.small}>{model.gb}</span>
              <span className={`${styles.modelState} ${have ? styles.good : styles.muted}`}>
                {have ? (
                  <>
                    <Check size={14} aria-hidden="true" />
                    Downloaded
                  </>
                ) : (
                  "Downloads on first use"
                )}
              </span>
            </div>
          );
        })}
        <div className={styles.modelRow}>
          <span className={styles.stack}>
            <span className={styles.modelName}>Precision checkpoints</span>
            <span className={styles.small}>Optional, pick the right sound more reliably</span>
          </span>
          <span className={`${styles.modelState} ${tvCount ? styles.good : styles.muted}`}>
            {tvCount ? `${tvCount} of 3 downloaded` : "On first use"}
          </span>
        </div>
      </div>
    </section>
  );
}

export function StartScreen({
  system,
  auth,
  onConnect,
  onOpen,
}: {
  system: SystemStatus | null;
  auth: AuthStatus | null;
  onConnect: () => void;
  onOpen: (session: Session | string) => void;
}) {
  const { sessions, error: listError, refresh } = useRecentSessions();
  const [dragging, setDragging] = useState(false);
  const [upload, setUpload] = useState<{ name: string; progress: number; abort: () => void } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const start = (file: File) => {
    setError(null);
    const job = uploadSession(file, (progress) =>
      setUpload((current) => (current ? { ...current, progress } : current)),
    );
    setUpload({ name: file.name, progress: 0, abort: job.abort });
    job.promise
      .then((session) => onOpen(session))
      .catch((err) => {
        if (err?.status !== -1) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setUpload(null));
  };

  const remove = async (id: string) => {
    if (confirmDelete !== id) {
      setConfirmDelete(id);
      setTimeout(() => setConfirmDelete((current) => (current === id ? null : current)), 4000);
      return;
    }
    setConfirmDelete(null);
    try {
      await api.deleteSession(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
    void refresh();
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <div>
          <h1 className={styles.title}>Pull any sound out of audio or video</h1>
          <p className={styles.lead}>
            Describe what you hear (vocals, a dog barking, traffic) and AudioGhost puts each sound on its own track, on
            your own GPU. Mix the tracks and export exactly what you hear. Your files never leave this computer.
          </p>
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
            const file = event.dataTransfer.files?.[0];
            if (file && !upload) start(file);
          }}
        >
          <div className={styles.dropIcon}>
            <Upload size={26} aria-hidden="true" />
          </div>
          {upload ? (
            <div className={styles.uploading} aria-live="polite">
              <div className={styles.uploadRow}>
                <span>
                  {upload.progress < 1 ? "Uploading" : "Reading the audio of"} <strong>{upload.name}</strong>
                </span>
                <span className="tabular">{Math.round(upload.progress * 100)}%</span>
              </div>
              <ProgressBar value={upload.progress} label="Upload progress" />
              <div>
                <Button size="small" onClick={upload.abort}>
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <>
              <div className={styles.dropTitle}>Drop an audio or video file</div>
              <div className={styles.muted}>MP3, WAV, FLAC, M4A, OGG · MP4, MOV, MKV, WEBM · up to 2 GB</div>
              <Button variant="primary" onClick={() => input.current?.click()}>
                Choose a file
              </Button>
              <input
                ref={input}
                type="file"
                accept={ACCEPT}
                className="visually-hidden"
                tabIndex={-1}
                aria-label="Choose an audio or video file"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  event.target.value = "";
                  if (file) start(file);
                }}
              />
            </>
          )}
        </div>
        {error && <ErrorText>{error}</ErrorText>}

        <section className={styles.section} aria-labelledby="recent-heading">
          <div className={styles.sectionHead}>
            <h2 id="recent-heading" className={styles.h2}>
              Recent
            </h2>
            <span className={styles.small}>Kept for 24 hours after the last change</span>
          </div>
          {listError && !sessions ? (
            <div className={styles.empty}>Recent sessions appear here once the backend is running.</div>
          ) : sessions && sessions.length === 0 ? (
            <div className={styles.empty}>Nothing yet. Your sessions show up here, so a page reload never loses work.</div>
          ) : (
            <ul className={styles.list}>
              {(sessions ?? []).map((item) => {
                const done = item.tracks.filter((t) => t.status === "done").map((t) => t.prompt);
                const expiry = expiresIn(item.expires_at);
                return (
                  <li key={item.id} className={styles.row}>
                    {item.is_video ? (
                      <Film size={18} className={styles.rowIcon} aria-hidden="true" />
                    ) : (
                      <FileAudio size={18} className={styles.rowIcon} aria-hidden="true" />
                    )}
                    <span className={styles.rowName} title={item.filename}>
                      {item.filename}
                    </span>
                    <span className={styles.rowMeta}>
                      {done.length ? done.join(", ") : "no tracks yet"} · {formatTime(item.duration)}
                    </span>
                    <span className={styles.rowActions}>
                      <span className={styles.rowMeta}>{expiry ?? relativeDay(item.updated_at)}</span>
                      <Button size="small" onClick={() => onOpen(item.id)} aria-label={`Open ${item.filename}`}>
                        Open
                      </Button>
                      {confirmDelete === item.id ? (
                        <Button size="small" danger onClick={() => remove(item.id)}>
                          Delete?
                        </Button>
                      ) : (
                        <IconButton label={`Delete ${item.filename}`} size="small" onClick={() => remove(item.id)}>
                          <Trash2 size={15} aria-hidden="true" />
                        </IconButton>
                      )}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </main>

      <aside className={styles.side}>
        <Setup system={system} auth={auth} onConnect={onConnect} />
        <section className={styles.card} aria-labelledby="new-heading">
          <h2 id="new-heading" className={styles.h2}>
            New in v1.2
          </h2>
          <ul className={styles.bullets}>
            <li>Several sounds in one run, each on its own track</li>
            <li>A mixer: volume, mute and solo per track, plus everything else</li>
            <li>Export exactly what you hear, or the video with your mix</li>
            <li>Your work survives a page reload</li>
          </ul>
        </section>
      </aside>
    </div>
  );
}
