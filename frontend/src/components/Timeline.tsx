"use client";

import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Ellipsis,
  Expand,
  Headphones,
  MapPin,
  Plus,
  RefreshCw,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { formatDb, formatDuration, formatLevel, formatSpan, formatTime } from "@/lib/format";
import { usePeaks } from "@/lib/hooks";
import { DEFAULT_CHANNEL } from "@/lib/mix";
import type { MultiTrackPlayer } from "@/lib/player";
import type {
  AnchorKind,
  ChannelState,
  Mixer,
  PendingPrompt,
  Region,
  RestVersion,
  Selection,
  Session,
  Sound,
  Track,
} from "@/lib/types";
import styles from "./Timeline.module.css";
import { cx, IconButton, Menu, ProgressBar, Spinner, type MenuItem } from "./ui";
import ui from "./ui.module.css";
import { Waveform } from "./Waveform";

const TICK_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];
const MODEL_LABEL: Record<string, string> = { small: "Small", base: "Base", large: "Large" };
const QUALITY_LABEL: Record<string, string> = { fast: "Fast", balanced: "Balanced", high: "High" };
const DRAG_PIXELS = 4;
const MIN_SELECTION = 0.05;

export interface LaneActions {
  onSolo: (key: string) => void;
  onMute: (key: string) => void;
  onVolume: (key: string, position: number) => void;
  onRemoveRange: (key: string, index: number) => void;
  onDownload: (key: string) => void;
  /** Another take of this sound, with a new seed. */
  onRetry: (sound: Sound) => void;
  onRetryWithHints: (sound: Sound) => void;
  /** Another take over the whole file, for a sound whose take covers a part. */
  onWholeFile: (sound: Sound) => void;
  onTake: (sound: Sound, take: Track) => void;
  onRemoveTake: (take: Track) => void;
  onRemoveSound: (sound: Sound) => void;
  /** Cancel a queued or running take, or dismiss a failed one. */
  onDrop: (take: Track) => void;
  onUseSeed: (seed: number) => void;
  onRebuildRest: () => void;
  onCancelRebuild: () => void;
  onRestVersion: (rev: number) => void;
  onDeleteRestVersion: (rev: number) => void;
  onHoldOriginal: (on: boolean) => void;
  onAddSound: () => void;
}

export interface TimelineHandle {
  /** Zoom so that this part fills the view. */
  zoomToSpan: (start: number, end: number) => void;
  /** Zoom by a factor, keeping the playhead (or the middle of the view) in place. */
  zoomBy: (factor: number) => void;
}

interface Props {
  session: Session;
  sounds: Sound[];
  player: MultiTrackPlayer;
  colors: Record<string, string>;
  mixer: Mixer;
  holdOriginal: boolean;
  actions: LaneActions;
  hints: Region[];
  pending: PendingPrompt[];
  /** The sound whose hints are being marked: drags mark instead of selecting. */
  marking: { key: string; kind: AnchorKind } | null;
  selection: Selection | null;
  onSelection: (selection: Selection | null) => void;
  /** The selected lane: "original", "rest" or a sound id. */
  focus: string | null;
  onFocus: (key: string | null) => void;
  zoom: number;
  maxZoom: number;
  onZoom: (zoom: number) => void;
  onAddHint: (hint: Omit<Region, "id">) => void;
  onRemoveHint: (id: string) => void;
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

/** The playhead's grip in the ruler (the ruler stays in view when the tracks scroll). */
function PlayheadGrip({ player, duration, laneWidth }: { player: MultiTrackPlayer; duration: number; laneWidth: number }) {
  const snap = useSyncExternalStore(player.subscribe, player.getSnapshot, player.getSnapshot);
  const x = (Math.min(snap.time, duration) / duration) * laneWidth;
  return <span className={styles.playheadGrip} style={{ transform: `translateX(${x}px)` }} aria-hidden="true" />;
}

/** Previous / next among a sound's takes, or the versions of "everything else". */
function Stepper({
  noun,
  index,
  count,
  detail,
  onStep,
}: {
  noun: string;
  index: number;
  count: number;
  detail?: string;
  onStep: (index: number) => void;
}) {
  return (
    <span className={styles.stepper} title={detail}>
      <button
        type="button"
        className={styles.step}
        aria-label={`Previous ${noun}`}
        disabled={index <= 0}
        onClick={() => onStep(index - 1)}
      >
        <ChevronLeft size={14} aria-hidden="true" />
      </button>
      <span className={`${styles.stepLabel} tabular`} aria-live="polite">
        <span className="visually-hidden">{noun} </span>
        {index + 1}/{count}
      </span>
      <button
        type="button"
        className={styles.step}
        aria-label={`Next ${noun}`}
        disabled={index >= count - 1}
        onClick={() => onStep(index + 1)}
      >
        <ChevronRight size={14} aria-hidden="true" />
      </button>
    </span>
  );
}

function OnlyRemove({ name, channel, onSolo, onMute }: { name: string; channel: ChannelState; onSolo: () => void; onMute: () => void }) {
  return (
    <>
      <button
        type="button"
        className={`${styles.toggle} ${styles.solo}`}
        aria-pressed={channel.solo}
        aria-label={`Only ${name}`}
        title={`Hear only ${name} (S)`}
        onClick={onSolo}
      >
        <span className={styles.long}>Only</span>
        <span className={styles.short}>S</span>
      </button>
      <button
        type="button"
        className={`${styles.toggle} ${styles.mute}`}
        aria-pressed={channel.muted}
        aria-label={`Remove ${name}`}
        title={`Take ${name} out of what you hear (M)`}
        onClick={onMute}
      >
        <span className={styles.long}>Remove</span>
        <span className={styles.short}>M</span>
      </button>
    </>
  );
}

function Level({ name, channel, onChange }: { name: string; channel: ChannelState; onChange: (position: number) => void }) {
  const text = channel.muted ? "Removed" : formatLevel(channel.volume);
  return (
    <div className={styles.volume} title={`How much of ${name} you hear: ${channel.muted ? "removed" : formatDb(channel.volume)}. Double-click for 100%.`}>
      <input
        type="range"
        min={0}
        max={100}
        step={1}
        value={channel.volume}
        aria-label={`How much of ${name} you hear`}
        aria-valuetext={text}
        onChange={(event) => onChange(Number(event.target.value))}
        onDoubleClick={() => onChange(DEFAULT_CHANNEL.volume)}
      />
      <span className={`${styles.db} tabular`}>{text}</span>
    </div>
  );
}

/** The parts of a lane where its mix change applies ("only here"). */
function OnlyHere({
  name,
  channel,
  duration,
  onRemove,
}: {
  name: string;
  channel: ChannelState;
  duration: number;
  onRemove: (index: number) => void;
}) {
  const ranges = channel.ranges ?? [];
  if (!ranges.length) return null;
  const kind = channel.solo ? "solo" : channel.muted || channel.volume <= 0 ? "mute" : "level";
  const label = kind === "solo" ? "Only this" : kind === "mute" ? "Removed" : formatLevel(channel.volume);
  return (
    <>
      {ranges.map(([start, end], index) => (
        <div
          key={`${start}-${end}`}
          className={styles.only}
          data-kind={kind}
          style={{ left: pct(start, duration), width: pct(end - start, duration) }}
        >
          <span className={styles.onlyLabel}>{label}</span>
          <button
            type="button"
            data-no-drag
            className={styles.onlyRemove}
            aria-label={`${name}: stop changing ${formatSpan(start, end)} only`}
            title="Remove this part"
            onClick={() => onRemove(index)}
          >
            <X size={12} aria-hidden="true" />
          </button>
        </div>
      ))}
    </>
  );
}

const takeBrief = (track: Track) =>
  `${MODEL_LABEL[track.settings.model_size] ?? track.settings.model_size} · ${QUALITY_LABEL[track.settings.quality] ?? track.settings.quality}`;

function takeSub(track: Track): string {
  const settings = track.settings;
  const base = `${MODEL_LABEL[settings.model_size] ?? settings.model_size} · ${QUALITY_LABEL[settings.quality] ?? settings.quality}`;
  return `${base}${track.elapsed ? ` · ${formatDuration(track.elapsed)}` : ""}`;
}

function runningText(track: Track, session: Session): string {
  if (track.status === "running") {
    const percent = Math.round(track.progress * 100);
    return `${track.message || "Separating"}${percent > 0 ? ` · ${percent}%` : ""}`;
  }
  if (track.message) return track.message; // e.g. downloading or loading the model
  if (track.jobs_ahead) return `Waiting · ${track.jobs_ahead} job${track.jobs_ahead > 1 ? "s" : ""} ahead`;
  const running = session.tracks.find((t) => t.status === "running");
  return running ? `Waiting for ${running.prompt}` : "Waiting for the worker";
}

export const Timeline = forwardRef<TimelineHandle, Props>(function Timeline(
  {
    session,
    sounds,
    player,
    colors,
    mixer,
    holdOriginal,
    actions,
    hints,
    pending,
    marking,
    selection,
    onSelection,
    focus,
    onFocus,
    zoom,
    maxZoom,
    onZoom,
    onAddHint,
    onRemoveHint,
  },
  ref,
) {
  const duration = Math.max(0.1, session.source.duration);
  const scroller = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLDivElement>(null);
  const hoverLine = useRef<HTMLDivElement>(null);
  const hoverLabel = useRef<HTMLSpanElement>(null);
  const gesture = useRef<{
    type: "scrub" | "select" | "mark" | "resize";
    x0: number;
    t0: number;
    lane: string | null;
    moved: boolean;
  } | null>(null);
  /** After the next layout, keep this time at this x of the view (zooming around a point). */
  const anchor = useRef<{ time: number; x: number } | null>(null);
  const [visible, setVisible] = useState(0);
  const [draft, setDraft] = useState<Selection | null>(null);
  const [draftHint, setDraftHint] = useState<{ start: number; end: number } | null>(null);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setVisible(el.clientWidth));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const compact = visible > 0 && visible < 640;
  const head = compact ? 176 : 288;
  const base = Math.max(200, (visible || 1000) - head);
  const laneWidth = base * zoom;

  useLayoutEffect(() => {
    const el = scroller.current;
    const wanted = anchor.current;
    if (!el || !wanted) return;
    anchor.current = null;
    el.scrollLeft = Math.max(0, head + (wanted.time / duration) * laneWidth - wanted.x);
  }, [laneWidth, head, duration]);

  // Keep the playhead in view while playing when zoomed in.
  const follow = useCallback(
    (x: number) => {
      const el = scroller.current;
      if (!el || el.scrollWidth <= el.clientWidth || gesture.current) return;
      if (x > el.scrollLeft + el.clientWidth - 24 || x < el.scrollLeft + head) {
        el.scrollLeft = Math.max(0, x - head - 40);
      }
    },
    [head],
  );

  const timeAt = useCallback(
    (clientX: number) => {
      const rect = canvas.current!.getBoundingClientRect();
      return Math.max(0, Math.min(duration, ((clientX - rect.left - head) / laneWidth) * duration));
    },
    [duration, head, laneWidth],
  );

  const zoomAround = useCallback(
    (next: number, clientX?: number) => {
      const el = scroller.current;
      const clamped = Math.max(1, Math.min(maxZoom, next));
      if (!el || clamped === zoom) return;
      const rect = el.getBoundingClientRect();
      const playheadX = head + (player.currentTime() / duration) * laneWidth - el.scrollLeft;
      const x =
        clientX !== undefined
          ? clientX - rect.left
          : playheadX > head && playheadX < rect.width
            ? playheadX
            : head + (rect.width - head) / 2;
      const time = ((el.scrollLeft + x - head) / laneWidth) * duration;
      anchor.current = { time: Math.max(0, Math.min(duration, time)), x };
      onZoom(clamped);
    },
    [duration, head, laneWidth, maxZoom, onZoom, player, zoom],
  );

  useImperativeHandle(
    ref,
    () => ({
      zoomBy: (factor) => zoomAround(zoom * factor),
      zoomToSpan: (start, end) => {
        const length = Math.max(0.2, end - start);
        const next = Math.max(1, Math.min(maxZoom, (0.9 * duration) / length));
        anchor.current = { time: start, x: head + 0.05 * base };
        if (next === zoom && scroller.current) {
          scroller.current.scrollLeft = Math.max(0, (start / duration) * laneWidth - 0.05 * base);
          anchor.current = null;
        } else onZoom(next);
      },
    }),
    [base, duration, head, laneWidth, maxZoom, onZoom, zoom, zoomAround],
  );

  // Ctrl + wheel (and trackpad pinch) zooms around the pointer. Needs a non-passive listener.
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoomAround(zoom * Math.exp(-event.deltaY * 0.0022), event.clientX);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoom, zoomAround]);

  const moveHover = (clientX: number | null) => {
    const line = hoverLine.current;
    if (!line || !canvas.current) return;
    const rect = canvas.current.getBoundingClientRect();
    if (clientX === null || clientX - rect.left < head) {
      line.style.opacity = "0";
      if (hoverLabel.current) hoverLabel.current.style.opacity = "0";
      return;
    }
    const t = timeAt(clientX);
    const x = (t / duration) * laneWidth;
    line.style.opacity = "1";
    line.style.transform = `translateX(${head + x}px)`;
    if (hoverLabel.current) {
      hoverLabel.current.style.opacity = "1";
      hoverLabel.current.style.transform = `translateX(${x}px)`;
      hoverLabel.current.textContent = formatTime(t, true);
    }
  };

  const edgeScroll = (clientX: number) => {
    const el = scroller.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (clientX > rect.right - 28) el.scrollLeft += 14;
    else if (clientX < rect.left + head + 28) el.scrollLeft -= 14;
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !canvas.current) return;
    const target = event.target as HTMLElement;
    if (target.closest("[data-no-drag]")) return;
    const rect = canvas.current.getBoundingClientRect();
    if (event.clientX - rect.left < head) return;
    canvas.current.setPointerCapture(event.pointerId);
    moveHover(null);
    const t = timeAt(event.clientX);
    const lane = target.closest<HTMLElement>("[data-lane]")?.dataset.lane ?? null;
    const edge = target.closest<HTMLElement>("[data-edge]")?.dataset.edge;
    if (edge && selection) {
      // Dragging an edge of the selection: the other edge stays where it is.
      const fixed = edge === "start" ? selection.end : selection.start;
      gesture.current = { type: "resize", x0: event.clientX, t0: fixed, lane: selection.lane, moved: true };
    } else if (target.closest("[data-ruler]")) {
      gesture.current = { type: "scrub", x0: event.clientX, t0: t, lane: null, moved: false };
      player.seek(t);
    } else if (event.shiftKey && !marking) {
      // Shift-click: select from the playhead (or stretch the selection) to here.
      const from = selection ? (Math.abs(t - selection.start) < Math.abs(t - selection.end) ? selection.end : selection.start) : player.currentTime();
      gesture.current = { type: "select", x0: event.clientX, t0: from, lane: selection?.lane ?? lane, moved: true };
      setDraft({ start: Math.min(from, t), end: Math.max(from, t), lane: selection?.lane ?? lane });
    } else {
      gesture.current = { type: marking ? "mark" : "select", x0: event.clientX, t0: t, lane, moved: false };
    }
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const g = gesture.current;
    if (!g) {
      moveHover(event.clientX);
      return;
    }
    const t = timeAt(event.clientX);
    if (g.type === "scrub") {
      player.seek(t);
      return;
    }
    if (!g.moved && Math.abs(event.clientX - g.x0) <= DRAG_PIXELS) return;
    g.moved = true;
    edgeScroll(event.clientX);
    const span = { start: Math.min(g.t0, t), end: Math.max(g.t0, t) };
    if (g.type === "mark") setDraftHint(span);
    else setDraft({ ...span, lane: g.lane });
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const g = gesture.current;
    gesture.current = null;
    setDraft(null);
    setDraftHint(null);
    if (!g || g.type === "scrub") return;
    const t = timeAt(event.clientX);
    if (!g.moved) {
      // A click: move the playhead there. Outside marking it also lets go of the selection.
      player.seek(t);
      if (g.type === "select") {
        onSelection(null);
        if (g.lane) onFocus(g.lane);
      }
      return;
    }
    const start = Math.min(g.t0, t);
    const end = Math.max(g.t0, t);
    if (g.type === "mark") {
      if (marking && end - start >= 0.2) onAddHint({ promptKey: marking.key, kind: marking.kind, start, end });
      return;
    }
    if (end - start < MIN_SELECTION) {
      onSelection(null);
      return;
    }
    onSelection({ start, end, lane: g.lane });
    if (g.lane) onFocus(g.lane);
  };

  const secondsPerPixel = duration / laneWidth;
  const step = TICK_STEPS.find((s) => s / secondsPerPixel >= 80) ?? 3600;
  const ticks: number[] = [];
  for (let t = 0; t < duration; t += step) ticks.push(t);
  const points = Math.round(1600 * Math.min(4, Math.max(1, zoom)));

  const sourcePeaks = usePeaks(session.source.peaks_url);
  const restPeaks = usePeaks(session.rest?.peaks_url ?? null);
  const done = sounds.filter((sound) => sound.take);
  const hasDone = done.length > 0;
  const promptText = (key: string) => pending.find((p) => p.key === key)?.text ?? "sound";
  const channel = (key: string): ChannelState => mixer[key] ?? DEFAULT_CHANNEL;
  const holdKeys = (event: ReactKeyboardEvent, on: boolean) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) {
      event.preventDefault();
      actions.onHoldOriginal(on);
    }
  };

  const shown = draft ?? selection;
  const left = (seconds: number) => head + (Math.max(0, Math.min(duration, seconds)) / duration) * laneWidth;
  const style = { "--head-width": `${head}px`, "--lane-width": `${laneWidth}px` } as CSSProperties;
  const rest = session.rest;
  const restJob = session.rest_job;
  const versions: RestVersion[] = rest ? [...rest.alts, rest].sort((a, b) => a.rev - b.rev) : [];
  const versionIndex = rest ? versions.findIndex((v) => v.rev === rest.rev) : 0;
  const restBusy = restJob?.status === "queued" || restJob?.status === "running";
  const restMenu: (MenuItem | null)[] = [
    { label: "Download this track", icon: <Download size={15} aria-hidden="true" />, onSelect: () => actions.onDownload("rest") },
    ...(rest && rest.alts.length
      ? [
          null,
          {
            label: "Delete this version",
            hint: "The other versions stay",
            icon: <Trash2 size={15} aria-hidden="true" />,
            danger: true,
            onSelect: () => actions.onDeleteRestVersion(rest.rev),
          } as MenuItem,
        ]
      : []),
  ];

  return (
    <div className={styles.scroller} ref={scroller}>
      <div
        className={styles.canvas}
        ref={canvas}
        data-compact={compact || undefined}
        data-marking={marking ? marking.kind : undefined}
        style={{ ...style, width: head + laneWidth }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => moveHover(null)}
        onPointerCancel={() => {
          gesture.current = null;
          setDraft(null);
          setDraftHint(null);
        }}
        role="group"
        aria-label="Timeline: drag the ruler to move the playhead, drag on a track to select a part"
      >
        {/* Ruler */}
        <div className={`${styles.row} ${styles.rulerRow}`}>
          <div className={`${styles.head} ${styles.rulerHead}`} data-no-drag>
            Tracks
          </div>
          <div className={styles.ruler} data-ruler title="Drag to move the playhead">
            {ticks.map((t) => (
              <div key={t} className={`${styles.tick} tabular`} style={{ left: pct(t, duration) }}>
                {formatTime(t, step < 1)}
              </div>
            ))}
            {hints.map((hint) => (
              <div
                key={hint.id}
                data-no-drag
                className={`${styles.regionChip} ${hint.kind === "-" ? styles.chipOut : styles.chipIn}`}
                data-dim={marking && marking.key !== hint.promptKey ? true : undefined}
                style={{ left: pct(hint.start, duration), width: `max(24px, ${pct(hint.end - hint.start, duration)})` }}
                title={`${promptText(hint.promptKey)} ${hint.kind === "+" ? "is here" : "is not here"}: ${formatSpan(hint.start, hint.end)}`}
              >
                <span className={styles.chipLabel}>
                  {promptText(hint.promptKey)} {hint.kind === "+" ? "here" : "not here"}
                </span>
                <button
                  type="button"
                  className={styles.chipRemove}
                  aria-label={`Remove the hint at ${formatSpan(hint.start, hint.end)}`}
                  onClick={() => onRemoveHint(hint.id)}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              </div>
            ))}
            {shown && (
              <div
                className={styles.selectionBar}
                style={{ left: pct(shown.start, duration), width: pct(shown.end - shown.start, duration) }}
              >
                <span className={styles.selectionEdge} data-edge="start" title="Drag to change where the selection starts" />
                <span className={`${styles.selectionLabel} tabular`}>{formatDuration(shown.end - shown.start)}</span>
                <span className={styles.selectionEdge} data-edge="end" title="Drag to change where the selection ends" />
              </div>
            )}
            <span className={`${styles.hoverLabel} tabular`} ref={hoverLabel} aria-hidden="true" />
            <PlayheadGrip player={player} duration={duration} laneWidth={laneWidth} />
          </div>
        </div>

        {/* Original */}
        <div className={styles.row}>
          <div
            className={styles.head}
            data-no-drag
            data-focused={focus === "original" || undefined}
            onPointerDown={() => onFocus("original")}
          >
            <div className={styles.headTop}>
              <span className={styles.swatch} style={{ background: "var(--track-original)" }} />
              <span className={styles.name}>Original</span>
            </div>
            {hasDone ? (
              <button
                type="button"
                className={styles.holdButton}
                aria-pressed={holdOriginal}
                title="Hold to compare with the untouched recording"
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
                <span className={styles.long}>Hold to hear the original</span>
                <span className={styles.short}>Hold: original</span>
              </button>
            ) : (
              <div className={styles.sub}>The untouched source</div>
            )}
          </div>
          <div className={styles.lane} data-lane="original" style={{ height: 76 }}>
            <Waveform peaks={sourcePeaks} color="var(--track-original)" opacity={!hasDone || holdOriginal ? 0.9 : 0.4} points={points} />
          </div>
        </div>

        {/* Sounds */}
        {sounds.map((sound) => (
          <SoundRow
            key={sound.id}
            sound={sound}
            session={session}
            color={colors[sound.id] ?? "var(--track-1)"}
            channel={channel(sound.id)}
            actions={actions}
            duration={duration}
            sourcePeaks={sourcePeaks}
            dimmed={holdOriginal}
            focused={focus === sound.id}
            onFocus={() => onFocus(sound.id)}
            points={points}
            compact={compact}
          />
        ))}

        {/* Everything else */}
        {rest && (
          <div className={styles.row}>
            <div
              className={styles.head}
              data-no-drag
              data-focused={focus === "rest" || undefined}
              onPointerDown={() => onFocus("rest")}
            >
              <div className={styles.headTop}>
                <span className={styles.swatch} style={{ background: "var(--track-rest)" }} />
                <span className={styles.name}>Everything else</span>
                {versions.length > 1 && !restBusy && (
                  <Stepper
                    noun="version"
                    index={versionIndex}
                    count={versions.length}
                    detail="Versions of everything else, oldest first"
                    onStep={(index) => actions.onRestVersion(versions[index].rev)}
                  />
                )}
                <Menu label="More for everything else" className={cx(ui.iconButton, ui.iconTiny)} items={restMenu}>
                  <Ellipsis size={16} aria-hidden="true" />
                </Menu>
              </div>
              <div className={styles.status} aria-live={restBusy ? "polite" : undefined}>
                <OnlyRemove
                  name="everything else"
                  channel={channel("rest")}
                  onSolo={() => actions.onSolo("rest")}
                  onMute={() => actions.onMute("rest")}
                />
                {restJob && restBusy ? (
                  <>
                    <Spinner size={13} />
                    <span className={styles.statusText} title={restJob.message}>
                      {restJob.status === "queued"
                        ? "Waiting to rebuild"
                        : `${restJob.message || "Rebuilding"}${restJob.progress > 0 ? ` · ${Math.round(restJob.progress * 100)}%` : ""}`}
                    </span>
                    <IconButton label="Cancel the rebuild" size="tiny" onClick={actions.onCancelRebuild}>
                      <X size={14} aria-hidden="true" />
                    </IconButton>
                  </>
                ) : restJob ? (
                  <>
                    <span className={`${styles.statusText} ${styles.bad}`} title={restJob.error ?? undefined}>
                      <AlertTriangle size={12} aria-hidden="true" /> {restJob.error || "The rebuild failed"}
                    </span>
                    <IconButton label="Dismiss" size="tiny" onClick={actions.onCancelRebuild}>
                      <X size={14} aria-hidden="true" />
                    </IconButton>
                  </>
                ) : (
                  <>
                    <span
                      className={`${styles.statusText} ${rest.stale ? styles.attention : ""}`}
                      title={
                        rest.stale
                          ? "A track changed since this was made. Rebuild it with the button on the right."
                          : "What the model leaves after removing the tracks above, one after another"
                      }
                    >
                      {rest.stale ? "Out of date" : done.length > 1 ? `${done.length} sounds out` : "What is left"}
                    </span>
                    {done.length > 1 && (
                      <IconButton
                        label="Try everything else again (the model removes the sounds again; this version is kept)"
                        size="tiny"
                        onClick={actions.onRebuildRest}
                      >
                        <RefreshCw size={14} aria-hidden="true" />
                      </IconButton>
                    )}
                  </>
                )}
              </div>
              <Level name="everything else" channel={channel("rest")} onChange={(v) => actions.onVolume("rest", v)} />
            </div>
            <div className={`${styles.lane} ${holdOriginal ? styles.laneDim : ""}`} data-lane="rest" style={{ height: 104 }}>
              <Waveform
                peaks={restPeaks}
                color="var(--track-rest)"
                opacity={channel("rest").muted && !channel("rest").ranges?.length ? 0.3 : 0.9}
                points={points}
              />
              <OnlyHere
                name="everything else"
                channel={channel("rest")}
                duration={duration}
                onRemove={(index) => actions.onRemoveRange("rest", index)}
              />
              {restJob?.status === "running" && (
                <div className={styles.progressFill} style={{ width: `${Math.round(restJob.progress * 100)}%` }} />
              )}
            </div>
          </div>
        )}

        {/* Add a sound */}
        <div className={styles.row}>
          <div className={`${styles.head} ${styles.addRow}`} data-no-drag>
            <button type="button" className={`${styles.holdButton} ${styles.dashed}`} onClick={actions.onAddSound}>
              <Plus size={15} aria-hidden="true" />
              Add a sound
            </button>
          </div>
          <div className={styles.addHint}>
            Each sound you add gets its own track. Space plays and pauses; press ? for all shortcuts.
          </div>
        </div>

        {/* Hints of the sounds that are not sent yet */}
        {hints.map((hint) => (
          <div
            key={hint.id}
            className={`${styles.region} ${hint.kind === "-" ? styles.regionOut : ""}`}
            data-dim={marking && marking.key !== hint.promptKey ? true : undefined}
            style={{ left: left(hint.start), width: left(hint.end) - left(hint.start) }}
            aria-hidden="true"
          />
        ))}
        {draftHint && marking && (
          <div
            className={`${styles.region} ${styles.draft} ${marking.kind === "-" ? styles.regionOut : ""}`}
            style={{ left: left(draftHint.start), width: left(draftHint.end) - left(draftHint.start) }}
            aria-hidden="true"
          />
        )}

        {/* The selection */}
        {shown && (
          <div
            className={styles.selection}
            style={{ left: left(shown.start), width: left(shown.end) - left(shown.start) }}
            aria-hidden="true"
          />
        )}

        <div className={styles.hoverLine} ref={hoverLine} aria-hidden="true" />
        <Playhead player={player} duration={duration} head={head} laneWidth={laneWidth} onFollow={follow} />
      </div>
    </div>
  );
});

function SoundRow({
  sound,
  session,
  color,
  channel,
  actions,
  duration,
  sourcePeaks,
  dimmed,
  focused,
  onFocus,
  points,
  compact,
}: {
  sound: Sound;
  session: Session;
  color: string;
  channel: ChannelState;
  actions: LaneActions;
  duration: number;
  sourcePeaks: number[] | null;
  dimmed: boolean;
  focused: boolean;
  onFocus: () => void;
  points: number;
  /** A narrow head: the takes sit on the second line, in place of the take's description. */
  compact: boolean;
}) {
  const take = sound.take;
  const peaks = usePeaks(take ? take.peaks_url : null);
  const running = sound.pending[0] ?? null;
  const failed = sound.failed[sound.failed.length - 1] ?? null;
  const index = take ? sound.takes.findIndex((t) => t.id === take.id) : 0;
  const name = sound.prompt;

  const takeStepper =
    take && sound.takes.length > 1 ? (
      <Stepper
        noun="take"
        index={index}
        count={sound.takes.length}
        detail={`Takes of ${name}, oldest first${take.seed != null ? ` · this one: seed ${take.seed}` : ""}`}
        onStep={(next) => actions.onTake(sound, sound.takes[next])}
      />
    ) : null;

  const menu: (MenuItem | null)[] = take
    ? [
        {
          label: "Try again",
          hint: "A new take with another seed. This one is kept.",
          icon: <RefreshCw size={15} aria-hidden="true" />,
          onSelect: () => actions.onRetry(sound),
        },
        {
          label: "Try again with hints...",
          hint: "First mark where the sound is, or is not",
          icon: <MapPin size={15} aria-hidden="true" />,
          onSelect: () => actions.onRetryWithHints(sound),
        },
        ...(take.range
          ? [
              {
                label: "Run on the whole file",
                hint: `This take covers ${formatSpan(take.range[0], take.range[1])} only`,
                icon: <Expand size={15} aria-hidden="true" />,
                onSelect: () => actions.onWholeFile(sound),
              } as MenuItem,
            ]
          : []),
        null,
        ...(take.seed !== null && take.seed !== undefined
          ? [
              {
                label: "Use this take's seed",
                hint: `Seed ${take.seed}: the same seed and settings repeat a result`,
                icon: <Copy size={15} aria-hidden="true" />,
                onSelect: () => actions.onUseSeed(take.seed as number),
              } as MenuItem,
            ]
          : []),
        { label: "Download this track", icon: <Download size={15} aria-hidden="true" />, onSelect: () => actions.onDownload(sound.id) },
        null,
        ...(sound.takes.length > 1
          ? [
              {
                label: "Delete this take",
                hint: `Take ${index + 1} of ${sound.takes.length}`,
                icon: <Trash2 size={15} aria-hidden="true" />,
                danger: true,
                onSelect: () => actions.onRemoveTake(take),
              } as MenuItem,
            ]
          : []),
        {
          label: "Remove this sound",
          hint: "It goes back into everything else",
          icon: <Trash2 size={15} aria-hidden="true" />,
          danger: true,
          onSelect: () => actions.onRemoveSound(sound),
        },
      ]
    : [];

  return (
    <div className={styles.row}>
      <div className={styles.head} data-no-drag data-focused={focused || undefined} onPointerDown={onFocus}>
        <div className={styles.headTop}>
          <span className={styles.swatch} style={{ background: color }} />
          <span className={styles.name} title={name}>
            {name}
          </span>
          {take && (
            <>
              {!compact && takeStepper}
              <Menu label={`More for ${name}`} className={cx(ui.iconButton, ui.iconTiny)} items={menu}>
                <Ellipsis size={16} aria-hidden="true" />
              </Menu>
            </>
          )}
          {!take && running && (
            <IconButton
              label={running.status === "running" ? `Cancel ${name}` : `Remove ${name} from the queue`}
              size="tiny"
              onClick={() => actions.onDrop(running)}
            >
              <X size={15} aria-hidden="true" />
            </IconButton>
          )}
        </div>

        {take ? (
          <div className={styles.status} aria-live={running ? "polite" : undefined}>
            <OnlyRemove
              name={name}
              channel={channel}
              onSolo={() => actions.onSolo(sound.id)}
              onMute={() => actions.onMute(sound.id)}
            />
            {running ? (
              <>
                <Spinner size={13} />
                <span className={styles.statusText} title={runningText(running, session)}>
                  New take: {runningText(running, session)}
                </span>
                <IconButton label={`Cancel the new take of ${name}`} size="tiny" onClick={() => actions.onDrop(running)}>
                  <X size={14} aria-hidden="true" />
                </IconButton>
              </>
            ) : failed ? (
              <>
                <span className={`${styles.statusText} ${styles.bad}`} title={failed.error ?? undefined}>
                  <AlertTriangle size={12} aria-hidden="true" /> New take failed: {failed.error || "separation failed"}
                </span>
                <IconButton label="Dismiss" size="tiny" onClick={() => actions.onDrop(failed)}>
                  <X size={14} aria-hidden="true" />
                </IconButton>
              </>
            ) : (
              <>
                {compact && takeStepper}
                <span
                  className={styles.statusText}
                  title={`${takeSub(take)}${take.range ? ` · only ${formatSpan(take.range[0], take.range[1])}` : ""}${take.seed != null ? ` · seed ${take.seed}` : ""}`}
                >
                  {compact && takeStepper ? "" : take.range ? `${formatTime(take.range[0])} to ${formatTime(take.range[1])}` : takeBrief(take)}
                </span>
                <IconButton label={`Try ${name} again (a new take; this one is kept)`} size="tiny" onClick={() => actions.onRetry(sound)}>
                  <RefreshCw size={14} aria-hidden="true" />
                </IconButton>
              </>
            )}
          </div>
        ) : failed && !running ? (
          <>
            <div className={styles.error} title={failed.error ?? undefined}>
              <AlertTriangle size={12} aria-hidden="true" /> {failed.error || "Separation failed"}
            </div>
            <div className={styles.actions}>
              <button type="button" className={styles.smallButton} onClick={() => actions.onRetry(sound)}>
                <RotateCcw size={12} aria-hidden="true" /> Try again
              </button>
              <button type="button" className={styles.smallButton} onClick={() => actions.onDrop(failed)}>
                <Trash2 size={12} aria-hidden="true" /> Remove
              </button>
            </div>
          </>
        ) : (
          <div className={styles.sub} aria-live="polite">
            {running ? runningText(running, session) : ""}
          </div>
        )}

        {take && <Level name={name} channel={channel} onChange={(v) => actions.onVolume(sound.id, v)} />}
        {!take && running?.status === "running" && <ProgressBar value={running.progress} label={`${name} progress`} />}
      </div>

      <div className={`${styles.lane} ${dimmed ? styles.laneDim : ""}`} data-lane={sound.id} style={{ height: 104 }}>
        {take ? (
          <>
            <Waveform peaks={peaks} color={color} opacity={channel.muted && !channel.ranges?.length ? 0.3 : 0.9} points={points} />
            {take.range && (
              <>
                <div className={styles.outside} style={{ left: 0, width: pct(take.range[0], duration) }} />
                <div className={styles.outside} style={{ left: pct(take.range[1], duration), right: 0 }} />
              </>
            )}
            <OnlyHere name={name} channel={channel} duration={duration} onRemove={(i) => actions.onRemoveRange(sound.id, i)} />
            {running?.status === "running" && (
              <div className={styles.progressFill} style={{ width: `${Math.round(running.progress * 100)}%` }} />
            )}
          </>
        ) : (
          <>
            <Waveform peaks={sourcePeaks} color={color} opacity={0.12} points={points} />
            {running?.status === "running" && (
              <div className={styles.progressFill} style={{ width: `${Math.round(running.progress * 100)}%` }} />
            )}
            {failed && !running && <div className={styles.failedTint} />}
            {running && (
              <div className={styles.laneStatus}>
                {running.status === "running" ? <Spinner size={14} /> : null}
                {running.status === "running" ? `Separating ${name}` : "Queued"}
                {running.range ? ` (${formatSpan(running.range[0], running.range[1])})` : ""}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
