"use client";

import { Clock, FileAudio, Film, Moon, Sun } from "lucide-react";
import Image from "next/image";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { formatTime } from "@/lib/format";
import { applyTheme, currentTheme, subscribeTheme } from "@/lib/prefs";
import type { AuthStatus, Session, SystemStatus } from "@/lib/types";
import styles from "./AppHeader.module.css";
import { Button, IconButton } from "./ui";

/** The AudioGhost logo (the original app icon). */
export function Logo({ size = 36 }: { size?: number }) {
  return (
    <Image
      src="/audioghost_logo.png"
      width={size}
      height={size}
      alt=""
      priority
      style={{ borderRadius: Math.round(size / 4), display: "block" }}
    />
  );
}

function summarize(system: SystemStatus | null): { tone: "ok" | "warn" | "bad"; label: string; meta?: string } {
  if (!system) return { tone: "warn", label: "Checking..." };
  if (!system.reachable) return { tone: "bad", label: "Backend offline" };
  if (!system.redis) return { tone: "bad", label: "Redis not running" };
  if (!system.worker) return { tone: "warn", label: "Worker not running" };
  if (system.worker_starting) return { tone: "warn", label: "Worker starting..." };
  const device = system.device;
  const gpu = device?.gpu?.replace(/^NVIDIA (GeForce )?/, "") ?? "Ready";
  const model = system.busy ? "separating" : device?.model ? `${device.model} ready` : "ready";
  const meta =
    device?.vram_total_gb != null && device.vram_used_gb != null
      ? `${device.vram_used_gb} / ${device.vram_total_gb} GB`
      : undefined;
  return { tone: "ok", label: `${gpu} · ${model}`, meta };
}

function StatusChip({ system, auth }: { system: SystemStatus | null; auth: AuthStatus | null }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const summary = summarize(system);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const checks: { tone: "ok" | "warn" | "bad"; title: string; hint?: React.ReactNode }[] = [];
  if (system) {
    checks.push(
      system.reachable
        ? { tone: "ok", title: `Backend v${system.version ?? "?"} on port 8000` }
        : { tone: "bad", title: "Backend offline", hint: <>Start AudioGhost with <code>start.bat</code> (or <code>./start.sh</code>).</> },
    );
    if (system.reachable) {
      checks.push(
        system.redis
          ? { tone: "ok", title: "Redis (job queue)" }
          : { tone: "bad", title: "Redis not running", hint: <>Start it with <code>start.bat</code> or <code>docker-compose up -d</code>.</> },
      );
    }
    if (system.reachable && system.redis) {
      checks.push(
        !system.worker
          ? { tone: "warn", title: "Separation worker not running", hint: "Check the AudioGhost Worker window for errors." }
          : system.worker_starting
            ? { tone: "warn", title: "Worker starting", hint: "Loading its libraries, about 20 seconds. Jobs you start now wait in the queue." }
            : {
                tone: "ok",
                title: system.device?.gpu ?? "Worker ready",
                hint: system.device?.model ? `Model in memory: sam-audio-${system.device.model}` : "No model loaded yet",
              },
      );
    }
  }
  const anyModel = (auth?.downloaded_models ?? []).length > 0;
  checks.push(
    auth?.license_accepted || anyModel
      ? { tone: "ok", title: anyModel ? "SAM-Audio models on this computer" : "SAM-Audio license accepted",
          hint: "Models download on first use. No Hugging Face account needed." }
      : { tone: "warn", title: "SAM-Audio license not accepted yet", hint: "It is shown before the first model download." },
  );

  return (
    <div className={styles.statusWrap} ref={wrap}>
      <button
        type="button"
        className={styles.status}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={`System status: ${summary.label}${summary.meta ? `, ${summary.meta} GPU memory in use` : ""}`}
        onClick={() => setOpen((value) => !value)}
      >
        <span className={`${styles.dot} ${styles[summary.tone]}`} aria-hidden="true" />
        <span className={styles.statusLabel}>{summary.label}</span>
        {summary.meta && <span className={`${styles.statusMeta} tabular`}>{summary.meta}</span>}
      </button>
      {open && (
        <div className={styles.popover} role="dialog" aria-label="System status">
          {checks.map((check) => (
            <div key={check.title} className={styles.check}>
              <span className={`${styles.dot} ${styles[check.tone]}`} aria-hidden="true" />
              <div>
                <div className={styles.checkTitle}>{check.title}</div>
                {check.hint && <div className={styles.checkHint}>{check.hint}</div>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ThemeToggle() {
  const theme = useSyncExternalStore(subscribeTheme, currentTheme, () => "dark" as const);
  const next = theme === "dark" ? "light" : "dark";
  return (
    <IconButton label={`Switch to ${next} theme`} onClick={() => applyTheme(next)}>
      {theme === "dark" ? <Moon size={18} aria-hidden="true" /> : <Sun size={18} aria-hidden="true" />}
    </IconButton>
  );
}

export function AppHeader({
  session,
  system,
  auth,
  onHome,
}: {
  session?: Session | null;
  system: SystemStatus | null;
  auth: AuthStatus | null;
  onHome: () => void;
}) {
  const source = session?.source;
  return (
    <header className={styles.header}>
      <button type="button" className={styles.brand} onClick={onHome} aria-label="AudioGhost home">
        <Logo />
        <span className={styles.wordmark}>
          Audio<span className={styles.ghost}>Ghost</span>
        </span>
      </button>
      {source && (
        <div className={styles.fileChip}>
          {source.is_video ? (
            <Film size={16} className={styles.fileIcon} aria-hidden="true" />
          ) : (
            <FileAudio size={16} className={styles.fileIcon} aria-hidden="true" />
          )}
          <span className={styles.fileName} title={source.filename}>
            {source.filename}
          </span>
          <span className={styles.fileMeta}>
            {formatTime(source.duration)} · {source.channels > 1 ? "stereo" : "mono"} · 48 kHz
          </span>
          <Button size="small" onClick={onHome}>
            New file
          </Button>
        </div>
      )}
      <div className={styles.spacer} />
      {session && (
        <Button onClick={onHome}>
          <Clock size={16} aria-hidden="true" />
          History
        </Button>
      )}
      <StatusChip system={system} auth={auth} />
      <ThemeToggle />
    </header>
  );
}
