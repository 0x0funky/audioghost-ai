"use client";

import { Ban, Crosshair, Download, Pause, Play, Repeat, SkipBack, X, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { api, apiUrl, downloadExport } from "@/lib/api";
import { formatDb, formatDuration, formatTime, sliderToGain } from "@/lib/format";
import { MultiTrackPlayer } from "@/lib/player";
import { DEFAULT_SETTINGS, loadPref, savePref } from "@/lib/prefs";
import type {
  AuthStatus,
  ChannelState,
  PendingPrompt,
  Region,
  SeparationSettings,
  Session,
  SystemStatus,
  Track,
} from "@/lib/types";
import { ExportDialog, type ExportKind } from "./ExportDialog";
import { SeparatePanel } from "./SeparatePanel";
import { Timeline, type LaneActions } from "./Timeline";
import { Button, IconButton, SectionLabel } from "./ui";
import styles from "./Workspace.module.css";

const DEFAULT_CHANNEL: ChannelState = { volume: 75, muted: false, solo: false };
let nextId = 0;
const uid = (prefix: string) => `${prefix}${Date.now().toString(36)}${(nextId++).toString(36)}`;

function Transport({
  player,
  duration,
  loop,
  onLoop,
  tool,
  onTool,
  zoom,
  onZoom,
  onMark,
}: {
  player: MultiTrackPlayer;
  duration: number;
  loop: boolean;
  onLoop: (loop: boolean) => void;
  tool: "+" | "-";
  onTool: (tool: "+" | "-") => void;
  zoom: number;
  onZoom: (zoom: number) => void;
  onMark: () => void;
}) {
  const snap = useSyncExternalStore(player.subscribe, player.getSnapshot, player.getSnapshot);
  return (
    <div className={styles.transport}>
      <IconButton label="Back to the start" onClick={() => player.seek(0)}>
        <SkipBack size={18} aria-hidden="true" />
      </IconButton>
      <button type="button" className={styles.play} aria-label={snap.playing ? "Pause" : "Play"} onClick={() => player.toggle()}>
        {snap.playing ? <Pause size={18} fill="currentColor" aria-hidden="true" /> : <Play size={18} fill="currentColor" aria-hidden="true" />}
      </button>
      <div className={`${styles.time} tabular`}>
        <span className={styles.timeNow}>{formatTime(snap.time, true)}</span>
        <span className={styles.timeTotal}> / {formatTime(duration, true)}</span>
      </div>
      <input
        type="range"
        className={styles.seek}
        min={0}
        max={duration}
        step={0.1}
        value={Math.min(snap.time, duration)}
        aria-label="Playback position"
        aria-valuetext={formatTime(snap.time)}
        onChange={(event) => player.seek(Number(event.target.value))}
      />
      <div className={styles.divider} />
      <span className={styles.muted}>Mark a region</span>
      <div role="group" aria-label="Region tool" className={styles.toolGroup}>
        <button type="button" className={styles.tool} aria-pressed={tool === "+"} onClick={() => onTool("+")}>
          <Crosshair size={15} aria-hidden="true" />
          Sound is here
        </button>
        <button type="button" className={`${styles.tool} ${styles.toolOut}`} aria-pressed={tool === "-"} onClick={() => onTool("-")}>
          <Ban size={15} aria-hidden="true" />
          Not here
        </button>
      </div>
      <Button size="small" onClick={onMark} title="Marks 5 seconds around the playhead (keyboard friendly)">
        Mark at playhead
      </Button>
      <div className={styles.spacer} />
      <IconButton label="Loop playback" aria-pressed={loop} onClick={() => onLoop(!loop)}>
        <Repeat size={18} aria-hidden="true" />
      </IconButton>
      <IconButton label="Zoom out" disabled={zoom <= 1} onClick={() => onZoom(Math.max(1, zoom / 2))}>
        <ZoomOut size={18} aria-hidden="true" />
      </IconButton>
      <IconButton label="Zoom in" disabled={zoom >= 16} onClick={() => onZoom(Math.min(16, zoom * 2))}>
        <ZoomIn size={18} aria-hidden="true" />
      </IconButton>
    </div>
  );
}

export function Workspace({
  session,
  onSessionChange,
  onRefresh,
  system,
  auth,
  onConnect,
}: {
  session: Session;
  onSessionChange: (session: Session) => void;
  onRefresh: () => void;
  system: SystemStatus | null;
  auth: AuthStatus | null;
  onConnect: () => void;
}) {
  const source = session.source;
  const [player] = useState(() => new MultiTrackPlayer(source.duration));
  const [mixer, setMixer] = useState<Record<string, ChannelState>>(() => loadPref(`mix.${session.id}`, {}));
  const [holdOriginal, setHoldOriginal] = useState(false);
  const [pending, setPending] = useState<PendingPrompt[]>([]);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [tool, setTool] = useState<"+" | "-">("+");
  const [zoom, setZoom] = useState(1);
  const [loop, setLoop] = useState(false);
  const [settings, setSettingsState] = useState<SeparationSettings>(() => loadPref("settings.v12", DEFAULT_SETTINGS));
  const [exportKind, setExportKind] = useState<ExportKind | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const setSettings = (next: SeparationSettings) => {
    setSettingsState(next);
    savePref("settings.v12", next);
  };
  const updateMixer = (key: string, patch: Partial<ChannelState>) =>
    setMixer((current) => {
      const next = { ...current, [key]: { ...DEFAULT_CHANNEL, ...current[key], ...patch } };
      savePref(`mix.${session.id}`, next);
      return next;
    });

  const colors = useMemo(() => {
    const map: Record<string, string> = {};
    session.tracks.forEach((track, index) => {
      map[track.id] = `var(--track-${(index % 8) + 1})`;
    });
    return map;
  }, [session.tracks]);

  const done = session.tracks.filter((t) => t.status === "done");
  const channel = (key: string) => mixer[key] ?? DEFAULT_CHANNEL;
  const mixKeys = [...done.map((t) => t.id), ...(session.rest ? ["rest"] : [])];
  const anySolo = mixKeys.some((key) => channel(key).solo);
  const mixGains: Record<string, number> = {};
  for (const key of mixKeys) {
    const ch = channel(key);
    mixGains[key] = ch.muted || (anySolo && !ch.solo) ? 0 : sliderToGain(ch.volume);
  }

  // What the player should play.
  const sources = [
    { key: "original", url: apiUrl(source.audio_url) },
    ...done.filter((t) => t.audio_url).map((t) => ({ key: t.id, url: apiUrl(t.audio_url!) })),
    ...(session.rest ? [{ key: "rest", url: apiUrl(session.rest.audio_url) }] : []),
  ];
  // Effects key on these strings, so a re-render with the same sources/gains is a no-op.
  const sourcesKey = JSON.stringify(sources);
  const playGains: Record<string, number> =
    holdOriginal || done.length === 0 ? { original: 1 } : { ...mixGains, original: 0 };
  const gainsKey = JSON.stringify(playGains);

  useEffect(() => {
    player.setSources(JSON.parse(sourcesKey));
  }, [player, sourcesKey]);
  useEffect(() => {
    player.setGains(JSON.parse(gainsKey));
  }, [player, gainsKey]);
  useEffect(() => {
    player.setLoop(loop);
  }, [player, loop]);
  useEffect(() => {
    player.attachVideo(videoRef.current);
    return () => player.attachVideo(null);
  }, [player]);
  useEffect(() => () => player.reset(), [player]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space" || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, button, a, [contenteditable='true'], dialog")) return;
      event.preventDefault();
      player.toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 9000);
    return () => clearTimeout(timer);
  }, [notice]);

  const label = (key: string) => (key === "rest" ? "everything else" : done.find((t) => t.id === key)?.prompt ?? key);
  const summaryParts = (() => {
    if (done.length === 0) return null;
    if (anySolo) return [`only ${mixKeys.filter((k) => channel(k).solo).map(label).join(" and ")}`];
    return mixKeys.map((key) => (channel(key).muted ? `${label(key)} muted` : `${label(key)} ${formatDb(channel(key).volume)}`));
  })();
  const summary = summaryParts ? summaryParts.join(", ") : "the original (no tracks yet)";

  const addPrompt = (text: string) => {
    const lower = text.toLowerCase();
    if (pending.some((p) => p.text.toLowerCase() === lower)) return;
    const key = uid("p");
    setPending((current) => [...current, { key, text }]);
    setActiveKey(key);
  };
  const removePrompt = (key: string) => {
    const rest = pending.filter((p) => p.key !== key);
    setPending(rest);
    setRegions((current) => current.filter((r) => r.promptKey !== key));
    if (activeKey === key) setActiveKey(rest.length ? rest[rest.length - 1].key : null);
  };

  const addRegion = (region: Omit<Region, "id">) => setRegions((current) => [...current, { ...region, id: uid("r") }]);
  const markAtPlayhead = () => {
    if (!activeKey) {
      setNotice("Add a sound in the panel first; regions are marked for the selected new sound.");
      inputRef.current?.focus();
      return;
    }
    const t = player.currentTime();
    const start = Math.max(0, t - 2.5);
    addRegion({ promptKey: activeKey, kind: tool, start, end: Math.min(source.duration, start + 5) });
  };

  const separate = async (typed?: string) => {
    const prompts = [...pending];
    if (typed && !prompts.some((p) => p.text.toLowerCase() === typed.toLowerCase())) {
      prompts.push({ key: uid("p"), text: typed });
    }
    if (!prompts.length) return;
    setSubmitting(true);
    try {
      const res = await api.separate(session.id, {
        ...settings,
        prompts: prompts.map((p) => ({
          text: p.text,
          anchors: regions
            .filter((r) => r.promptKey === p.key)
            .map((r) => [r.kind, Math.round(r.start * 1000) / 1000, Math.round(r.end * 1000) / 1000] as [
              "+" | "-",
              number,
              number,
            ]),
        })),
      });
      onSessionChange(res.session);
      setPending([]);
      setRegions([]);
      setActiveKey(null);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const removeTrack = async (track: Track) => {
    try {
      await api.removeTrack(session.id, track.id);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err));
    }
    onRefresh();
  };

  const actions: LaneActions = {
    onSolo: (key) => updateMixer(key, { solo: !channel(key).solo }),
    onMute: (key) => updateMixer(key, { muted: !channel(key).muted }),
    onVolume: (key, volume) => updateMixer(key, { volume }),
    onDownload: (key) => {
      downloadExport(session.id, { kind: "mix", format: "wav", gains: { [key]: 1 }, tracks: [] }).catch((err) =>
        setNotice(err instanceof Error ? err.message : String(err)),
      );
    },
    onRemove: (track) => void removeTrack(track),
    onRetry: async (track) => {
      try {
        const res = await api.separate(session.id, {
          ...track.settings,
          prompts: [{ text: track.prompt, anchors: track.anchors }],
        });
        onSessionChange(res.session);
        await api.removeTrack(session.id, track.id);
        onRefresh();
      } catch (err) {
        setNotice(err instanceof Error ? err.message : String(err));
      }
    },
    onHoldOriginal: setHoldOriginal,
    onAddSound: () => inputRef.current?.focus(),
  };

  const blocker = !system
    ? null
    : !system.reachable
      ? "The backend is offline: start AudioGhost with start.bat (or start.sh)."
      : !system.redis
        ? "Redis is not running, so jobs can't be queued."
        : null;

  const estimate = (() => {
    const count = pending.length || 1;
    const similar = done.filter(
      (t) => t.elapsed && t.settings.model_size === settings.model_size && t.settings.quality === settings.quality,
    );
    const busy = session.tracks.some((t) => t.status === "running" || t.status === "queued");
    if (!similar.length) return busy ? "Starts after the current job" : null;
    const perSound = similar.reduce((sum, t) => sum + (t.elapsed ?? 0), 0) / similar.length;
    return `About ${formatDuration(perSound * count)}${busy ? " after the current job" : ""}`;
  })();

  return (
    <div className={styles.workspace}>
      <main className={styles.main}>
        {source.is_video && source.video_url && (
          <div className={styles.videoWrap}>
            <video
              ref={videoRef}
              className={styles.video}
              src={apiUrl(source.video_url)}
              muted
              playsInline
              preload="auto"
              aria-label="Video preview, plays along with your mix"
              onClick={() => player.toggle()}
            />
          </div>
        )}
        <Transport
          player={player}
          duration={source.duration}
          loop={loop}
          onLoop={setLoop}
          tool={tool}
          onTool={setTool}
          zoom={zoom}
          onZoom={setZoom}
          onMark={markAtPlayhead}
        />
        {notice && (
          <div className={styles.notice} role="status">
            <span>{notice}</span>
            <IconButton label="Dismiss" size="tiny" onClick={() => setNotice(null)}>
              <X size={14} aria-hidden="true" />
            </IconButton>
          </div>
        )}
        <Timeline
          session={session}
          player={player}
          colors={colors}
          mixer={mixer}
          holdOriginal={holdOriginal}
          actions={actions}
          regions={regions}
          pending={pending}
          activeKey={activeKey}
          tool={tool}
          zoom={zoom}
          onAddRegion={addRegion}
          onRemoveRegion={(id) => setRegions((current) => current.filter((r) => r.id !== id))}
          onNotice={setNotice}
        />
        <div className={styles.mixBar}>
          <div className={styles.mixText}>
            <SectionLabel as="span">What you hear</SectionLabel>
            <div className={styles.mixSummary}>{summary}</div>
          </div>
          <div className={styles.spacer} />
          <Button disabled={!done.length} onClick={() => setExportKind("tracks")}>
            <Download size={16} aria-hidden="true" />
            All tracks (.zip)
          </Button>
          <Button variant="strong" disabled={!done.length} onClick={() => setExportKind(source.is_video ? "video" : "mix")}>
            <Download size={16} aria-hidden="true" />
            {source.is_video ? "Export video with this mix" : "Export what you hear"}
          </Button>
        </div>
      </main>

      <SeparatePanel
        session={session}
        colors={colors}
        pending={pending}
        activeKey={activeKey}
        regions={regions}
        settings={settings}
        onSettings={setSettings}
        onAddPrompt={addPrompt}
        onRemovePrompt={removePrompt}
        onActivate={setActiveKey}
        onRemoveTrack={(track) => void removeTrack(track)}
        onSeparate={separate}
        submitting={submitting}
        blocker={blocker}
        needsAuth={!!auth && !auth.authenticated}
        onConnect={onConnect}
        inputRef={inputRef}
        estimate={estimate}
      />

      {exportKind && (
        <ExportDialog
          key={exportKind}
          session={session}
          gains={mixGains}
          summary={summary}
          open
          initialKind={exportKind}
          onClose={() => setExportKind(null)}
        />
      )}
    </div>
  );
}
