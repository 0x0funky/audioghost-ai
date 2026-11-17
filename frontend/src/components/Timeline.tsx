"use client";

import { AlertTriangle, Download, Headphones, Plus, RotateCcw, Trash2, X } from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { formatDb, formatDuration, formatTime } from "@/lib/format";
import { usePeaks } from "@/lib/hooks";
import type { MultiTrackPlayer } from "@/lib/player";
import type { ChannelState, PendingPrompt, Region, Session, Track } from "@/lib/types";
import styles from "./Timeline.module.css";
import { IconButton, ProgressBar, Spinner } from "./ui";
import { Waveform } from "./Waveform";

const TICK_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];
const MODEL_LABEL: Record<string, string> = { small: "Small", base: "Base", large: "Large" };
const QUALITY_LABEL: Record<string, string> = { fast: "Fast", balanced: "Balanced", high: "High" };

export interface LaneActions {
  onSolo: (key: string) => void;
  onMute: (key: string) => void;
  onVolume: (key: string, position: number) => void;
  onDownload: (key: string) => void;
  onRemove: (track: Track) => void;
  onRetry: (track: Track) => void;
  onHoldOriginal: (on: boolean) => void;
  onAddSound: () => void;
}

interface Props {
  session: Session;
  player: MultiTrackPlayer;
  colors: Record<string, string>;
  mixer: Record<string, ChannelState>;
  holdOriginal: boolean;
  actions: LaneActions;
  regions: Region[];
  pending: PendingPrompt[];
  activeKey: string | null;
  tool: "+" | "-";
  zoom: number;
  onAddRegion: (region: Omit<Region, "id">) => void;
  onRemoveRegion: (id: string) => void;
  onNotice: (message: string) => void;
}

const pct = (seconds: number, duration: number) => `${(Math.max(0, Math.min(duration, seconds)) / duration) * 100}%`;

function Playhead({
  player,
  duration,
  head,
  laneWidth,
  onFollow,
}: {
  player: MultiTrackPlayer;
  duration: number;
  head: number;
  laneWidth: number;
  onFollow: (x: number) => void;
}) {
  const snap = useSyncExternalStore(player.subscribe, player.getSnapshot, player.getSnapshot);
  const x = head + (Math.min(snap.time, duration) / duration) * laneWidth;
  useEffect(() => {
    if (snap.playing) onFollow(x);
  }, [x, snap.playing, onFollow]);
  return <div className={styles.playhead} style={{ transform: `translateX(${x}px)` }} aria-hidden="true" />;
}

function RegionOverlays({ regions, draft, duration }: { regions: Region[]; draft: Region | null; duration: number }) {
  return (
    <>
      {regions.map((region) => (
        <div
          key={region.id}
          className={`${styles.region} ${region.kind === "-" ? styles.regionOut : ""}`}
          style={{ left: pct(region.start, duration), width: pct(region.end - region.start, duration) }}
        />
      ))}
      {draft && (
        <div
          className={`${styles.region} ${styles.draft} ${draft.kind === "-" ? styles.regionOut : ""}`}
          style={{ left: pct(draft.start, duration), width: pct(draft.end - draft.start, duration) }}
        />
      )}
    </>
  );
}

function SoloMute({ name, channel, onSolo, onMute }: { name: string; channel: ChannelState; onSolo: () => void; onMute: () => void }) {
  return (
    <>
      <button
        type="button"
        className={`${styles.toggle} ${styles.solo}`}
        aria-pressed={channel.solo}
        aria-label={`Solo ${name}`}
        title={`Solo ${name} (hear only the soloed tracks)`}
        onClick={onSolo}
      >
        S
      </button>
      <button
        type="button"
        className={`${styles.toggle} ${styles.mute}`}
        aria-pressed={channel.muted}
        aria-label={`Mute ${name}`}
        title={`Mute ${name} (remove it from what you hear)`}
        onClick={onMute}
      >
        M
      </button>
    </>
  );
}

function Volume({ name, channel, onChange }: { name: string; channel: ChannelState; onChange: (position: number) => void }) {
  return (
    <div className={styles.volume}>
      <input
        type="range"
        min={0}
        max={100}
        step={1}
        value={channel.volume}
        aria-label={`${name} volume`}
        aria-valuetext={formatDb(channel.volume)}
        onChange={(event) => onChange(Number(event.target.value))}
        onDoubleClick={() => onChange(75)}
      />
      <span className={`${styles.db} tabular`}>{channel.muted ? "Muted" : formatDb(channel.volume)}</span>
    </div>
  );
}

function trackSub(track: Track, session: Session): string {
  const settings = track.settings;
  const base = `${MODEL_LABEL[settings.model_size] ?? settings.model_size} · ${QUALITY_LABEL[settings.quality] ?? settings.quality}`;
  if (track.status === "done") return `${base}${track.elapsed ? ` · done in ${formatDuration(track.elapsed)}` : ""}`;
  if (track.status === "running") {
    const percent = Math.round(track.progress * 100);
    return `${track.message || "Separating"}${percent > 0 ? ` · ${percent}%` : ""}`;
  }
  if (track.status === "queued") {
    if (track.message) return track.message; // e.g. downloading or loading the model
    if (track.jobs_ahead) return `Waiting · ${track.jobs_ahead} job${track.jobs_ahead > 1 ? "s" : ""} ahead`;
    const running = session.tracks.find((t) => t.status === "running");
    return running ? `Waiting for ${running.prompt}` : "Waiting for the worker";
  }
  return "Failed";
}

export function Timeline({
  session,
  player,
  colors,
  mixer,
  holdOriginal,
  actions,
  regions,
  pending,
  activeKey,
  tool,
  zoom,
  onAddRegion,
  onRemoveRegion,
  onNotice,
}: Props) {
  const duration = Math.max(0.1, session.source.duration);
  const scroller = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x0: number; t0: number } | null>(null);
  const [visible, setVisible] = useState(0);
  const [draft, setDraft] = useState<Region | null>(null);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setVisible(el.clientWidth));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const head = visible && visible < 720 ? 176 : 272;
  const laneWidth = Math.max(200, (visible || 1000) - head) * zoom;

  // Keep the playhead in view while playing when zoomed in.
  const follow = useCallback(
    (x: number) => {
      const el = scroller.current;
      if (!el || el.scrollWidth <= el.clientWidth) return;
      if (x > el.scrollLeft + el.clientWidth - 24 || x < el.scrollLeft + head) {
        el.scrollLeft = Math.max(0, x - head - 40);
      }
    },
    [head],
  );

  const sourcePeaks = usePeaks(session.source.peaks_url);
  const restPeaks = usePeaks(session.rest?.peaks_url ?? null);
  const hasDone = session.tracks.some((t) => t.status === "done");

  const timeAt = (clientX: number) => {
    const rect = canvas.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - rect.left - head) / laneWidth) * duration));
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !canvas.current) return;
    if ((event.target as HTMLElement).closest("[data-no-scrub]")) return;
    const rect = canvas.current.getBoundingClientRect();
    if (event.clientX - rect.left < head) return;
    canvas.current.setPointerCapture(event.pointerId);
    drag.current = { x0: event.clientX, t0: timeAt(event.clientX) };
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = drag.current;
    if (!start || Math.abs(event.clientX - start.x0) <= 4) return;
    const t = timeAt(event.clientX);
    setDraft({
      id: "draft",
      promptKey: activeKey ?? "",
      kind: tool,
      start: Math.min(start.t0, t),
      end: Math.max(start.t0, t),
    });
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = drag.current;
    drag.current = null;
    setDraft(null);
    if (!start) return;
    const t = timeAt(event.clientX);
    if (Math.abs(event.clientX - start.x0) <= 4) {
      player.seek(t);
      return;
    }
    const from = Math.min(start.t0, t);
    const to = Math.max(start.t0, t);
    if (to - from < 0.2) return;
    if (!activeKey) {
      onNotice("Add a sound in the panel first, then drag over the timeline where you can (or can't) hear it.");
      return;
    }
    onAddRegion({ promptKey: activeKey, kind: tool, start: from, end: to });
  };

  const secondsPerPixel = duration / laneWidth;
  const step = TICK_STEPS.find((s) => s / secondsPerPixel >= 80) ?? 3600;
  const ticks: number[] = [];
  for (let t = 0; t < duration; t += step) ticks.push(t);

  const promptText = (key: string) => pending.find((p) => p.key === key)?.text ?? "sound";
  const channel = (key: string): ChannelState => mixer[key] ?? { volume: 75, muted: false, solo: false };
  const holdKeys = (event: ReactKeyboardEvent, on: boolean) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) {
      event.preventDefault();
      actions.onHoldOriginal(on);
    }
  };

  const style = { "--head-width": `${head}px`, "--lane-width": `${laneWidth}px` } as CSSProperties;

  return (
    <div className={styles.scroller} ref={scroller}>
      <div
        className={styles.canvas}
        ref={canvas}
        style={{ ...style, width: head + laneWidth }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          drag.current = null;
          setDraft(null);
        }}
        role="group"
        aria-label="Timeline: click to move the playhead, drag to mark a region"
      >
        {/* Ruler */}
        <div className={styles.row}>
          <div className={`${styles.head} ${styles.rulerHead}`} data-no-scrub>
            Tracks
          </div>
          <div className={styles.ruler}>
            {ticks.map((t) => (
              <div key={t} className={`${styles.tick} tabular`} style={{ left: pct(t, duration) }}>
                {formatTime(t)}
              </div>
            ))}
            {regions.map((region) => (
              <div
                key={region.id}
                data-no-scrub
                className={`${styles.regionChip} ${region.kind === "-" ? styles.chipOut : styles.chipIn}`}
                style={{ left: pct(region.start, duration), width: `max(24px, ${pct(region.end - region.start, duration)})` }}
                title={`${promptText(region.promptKey)} ${region.kind === "+" ? "is here" : "is not here"}: ${formatTime(region.start, true)}-${formatTime(region.end, true)}`}
              >
                <span className={styles.chipLabel}>
                  {promptText(region.promptKey)} {region.kind === "+" ? "here" : "not here"}
                </span>
                <button
                  type="button"
                  className={styles.chipRemove}
                  aria-label={`Remove region ${formatTime(region.start, true)} to ${formatTime(region.end, true)}`}
                  onClick={() => onRemoveRegion(region.id)}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Original */}
        <div className={styles.row}>
          <div className={styles.head} data-no-scrub>
            <div className={styles.headTop}>
              <span className={styles.swatch} style={{ background: "var(--track-original)" }} />
              <span className={styles.name}>Original</span>
            </div>
            <div className={styles.sub}>The untouched source</div>
            {hasDone && (
              <button
                type="button"
                className={styles.holdButton}
                aria-pressed={holdOriginal}
                onPointerDown={(event) => {
                  event.currentTarget.setPointerCapture(event.pointerId);
                  actions.onHoldOriginal(true);
                }}
                onPointerUp={() => actions.onHoldOriginal(false)}
                onPointerCancel={() => actions.onHoldOriginal(false)}
                onKeyDown={(event) => holdKeys(event, true)}
                onKeyUp={(event) => holdKeys(event, false)}
                onBlur={() => actions.onHoldOriginal(false)}
              >
                <Headphones size={15} aria-hidden="true" />
                Hold to hear the original
              </button>
            )}
          </div>
          <div className={styles.lane} style={{ height: 88 }}>
            <RegionOverlays regions={regions} draft={draft} duration={duration} />
            <Waveform peaks={sourcePeaks} color="var(--track-original)" opacity={!hasDone || holdOriginal ? 0.9 : 0.4} />
          </div>
        </div>

        {/* Tracks */}
        {session.tracks.map((track) => {
          const color = colors[track.id] ?? "var(--track-1)";
          const ch = channel(track.id);
          return (
            <TrackRow
              key={track.id}
              track={track}
              session={session}
              color={color}
              channel={ch}
              actions={actions}
              regions={regions}
              draft={draft}
              duration={duration}
              sourcePeaks={sourcePeaks}
              dimmed={holdOriginal}
            />
          );
        })}

        {/* Everything else */}
        {session.rest && (
          <div className={styles.row}>
            <div className={styles.head} data-no-scrub>
              <div className={styles.headTop}>
                <span className={styles.swatch} style={{ background: "var(--track-rest)" }} />
                <span className={styles.name}>Everything else</span>
                <SoloMute
                  name="everything else"
                  channel={channel("rest")}
                  onSolo={() => actions.onSolo("rest")}
                  onMute={() => actions.onMute("rest")}
                />
                <IconButton label="Download everything else" size="small" onClick={() => actions.onDownload("rest")}>
                  <Download size={15} aria-hidden="true" />
                </IconButton>
              </div>
              <div className={styles.sub}>What is left after the tracks above</div>
              <Volume name="Everything else" channel={channel("rest")} onChange={(v) => actions.onVolume("rest", v)} />
            </div>
            <div className={`${styles.lane} ${holdOriginal ? styles.laneDim : ""}`} style={{ height: 76 }}>
              <RegionOverlays regions={regions} draft={draft} duration={duration} />
              <Waveform peaks={restPeaks} color="var(--track-rest)" opacity={channel("rest").muted ? 0.3 : 0.9} />
            </div>
          </div>
        )}

        {/* Add a sound */}
        <div className={styles.row}>
          <div className={`${styles.head} ${styles.addRow}`} data-no-scrub>
            <button type="button" className={`${styles.holdButton} ${styles.dashed}`} onClick={actions.onAddSound}>
              <Plus size={15} aria-hidden="true" />
              Add a sound
            </button>
          </div>
          <div className={styles.addHint}>
            Drag across the timeline to mark where a sound is clearly audible, or where it is not; click to jump there.
            Space plays and pauses.
          </div>
        </div>

        <Playhead player={player} duration={duration} head={head} laneWidth={laneWidth} onFollow={follow} />
      </div>
    </div>
  );
}

function TrackRow({
  track,
  session,
  color,
  channel,
  actions,
  regions,
  draft,
  duration,
  sourcePeaks,
  dimmed,
}: {
  track: Track;
  session: Session;
  color: string;
  channel: ChannelState;
  actions: LaneActions;
  regions: Region[];
  draft: Region | null;
  duration: number;
  sourcePeaks: number[] | null;
  dimmed: boolean;
}) {
  const peaks = usePeaks(track.status === "done" ? track.peaks_url : null);
  const active = track.status === "running" || track.status === "queued";
  return (
    <div className={styles.row}>
      <div className={styles.head} data-no-scrub>
        <div className={styles.headTop}>
          <span className={styles.swatch} style={{ background: color }} />
          <span className={styles.name} title={track.prompt}>
            {track.prompt}
          </span>
          {track.status === "done" && (
            <>
              <SoloMute
                name={track.prompt}
                channel={channel}
                onSolo={() => actions.onSolo(track.id)}
                onMute={() => actions.onMute(track.id)}
              />
              <IconButton label={`Download ${track.prompt}`} size="small" onClick={() => actions.onDownload(track.id)}>
                <Download size={15} aria-hidden="true" />
              </IconButton>
            </>
          )}
          {active && (
            <IconButton
              label={track.status === "running" ? `Cancel ${track.prompt}` : `Remove ${track.prompt} from the queue`}
              size="small"
              onClick={() => actions.onRemove(track)}
            >
              <X size={15} aria-hidden="true" />
            </IconButton>
          )}
        </div>
        {track.status === "failed" ? (
          <>
            <div className={styles.error} title={track.error ?? undefined}>
              <AlertTriangle size={12} aria-hidden="true" /> {track.error || "Separation failed"}
            </div>
            <div className={styles.actions}>
              <button type="button" className={styles.smallButton} onClick={() => actions.onRetry(track)}>
                <RotateCcw size={12} aria-hidden="true" /> Try again
              </button>
              <button type="button" className={styles.smallButton} onClick={() => actions.onRemove(track)}>
                <Trash2 size={12} aria-hidden="true" /> Remove
              </button>
            </div>
          </>
        ) : (
          <div className={styles.sub} aria-live={active ? "polite" : undefined}>
            {trackSub(track, session)}
          </div>
        )}
        {track.status === "done" && (
          <Volume name={track.prompt} channel={channel} onChange={(v) => actions.onVolume(track.id, v)} />
        )}
        {track.status === "running" && <ProgressBar value={track.progress} label={`${track.prompt} progress`} />}
      </div>
      <div className={`${styles.lane} ${dimmed ? styles.laneDim : ""}`} style={{ height: 84 }}>
        <RegionOverlays regions={regions} draft={draft} duration={duration} />
        {track.status === "done" ? (
          <Waveform peaks={peaks} color={color} opacity={channel.muted ? 0.3 : 0.9} />
        ) : (
          <>
            <Waveform peaks={sourcePeaks} color={color} opacity={0.12} />
            {track.status === "running" && (
              <div className={styles.progressFill} style={{ width: `${Math.round(track.progress * 100)}%` }} />
            )}
            {track.status === "failed" && <div className={styles.failedTint} />}
            {active && (
              <div className={styles.laneStatus}>
                {track.status === "running" ? <Spinner size={14} /> : null}
                {track.status === "running" ? `Separating ${track.prompt}` : "Queued"}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
