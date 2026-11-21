"use client";

import {
  Ban,
  Crosshair,
  Download,
  Keyboard,
  Layers,
  Maximize2,
  PanelRightClose,
  PanelRightOpen,
  Pause,
  Play,
  Redo2,
  Repeat,
  ScanLine,
  SkipBack,
  Undo2,
  Video,
  VideoOff,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { api, apiUrl, downloadExport } from "@/lib/api";
import { formatDuration, formatLevel, formatSpan, formatTime } from "@/lib/format";
import { useHistory } from "@/lib/history";
import {
  channelGains,
  DEFAULT_CHANNEL,
  groupSounds,
  isChanged,
  mergeSpans,
  mixSegments,
  playPlan,
  takeGains,
  withoutKey,
} from "@/lib/mix";
import { MultiTrackPlayer } from "@/lib/player";
import { DEFAULT_LAYOUT, DEFAULT_SETTINGS, loadPref, savePref, type LayoutPrefs } from "@/lib/prefs";
import type {
  Anchor,
  AnchorKind,
  AuthStatus,
  ChannelState,
  Mixer,
  ModelGate,
  PendingPrompt,
  Region,
  Selection,
  SeparationSettings,
  Session,
  Sound,
  Span,
  SystemStatus,
  Track,
} from "@/lib/types";
import { BatchDialog } from "./BatchDialog";
import { ExportDialog, type ExportKind } from "./ExportDialog";
import { MIN_PART_SECONDS, SeparatePanel } from "./SeparatePanel";
import { Timeline, type LaneActions, type TimelineHandle } from "./Timeline";
import { Button, Dialog, IconButton, Menu, SectionLabel, Splitter } from "./ui";
import { RebuildDialog, ShortcutsDialog } from "./WorkspaceDialogs";
import styles from "./Workspace.module.css";

let nextId = 0;
const uid = (prefix: string) => `${prefix}${Date.now().toString(36)}${(nextId++).toString(36)}`;

function Clock({ player, duration }: { player: MultiTrackPlayer; duration: number }) {
  const snap = useSyncExternalStore(player.subscribe, player.getSnapshot, player.getSnapshot);
  return (
    <div className={`${styles.time} tabular`} aria-label="Playback position">
      <span className={styles.timeNow}>{formatTime(snap.time, true)}</span>
      <span className={styles.timeTotal}> / {formatTime(duration, true)}</span>
    </div>
  );
}

function PlayButton({ player }: { player: MultiTrackPlayer }) {
  const snap = useSyncExternalStore(player.subscribe, player.getSnapshot, player.getSnapshot);
  return (
    <button
      type="button"
      className={styles.play}
      aria-label={snap.playing ? "Pause" : "Play"}
      title={snap.playing ? "Pause (Space)" : "Play (Space)"}
      onClick={() => player.toggle()}
    >
      {snap.playing ? <Pause size={20} aria-hidden="true" /> : <Play size={20} aria-hidden="true" />}
    </button>
  );
}

/**
 * What a separation with these settings still needs before its model can download.
 * Nothing once the model is on this computer (Meta's checkpoint or AudioGhost's copy).
 */
function modelGate(auth: AuthStatus | null, settings: SeparationSettings): ModelGate {
  if (!auth) return null;
  const ready = auth.downloaded_models ?? [];
  if (settings.variant === "tv") {
    return ready.includes(`facebook/sam-audio-${settings.model_size}-tv`) || auth.authenticated ? null : "token";
  }
  const have = ready.some((repo) => repo.endsWith(`/sam-audio-${settings.model_size}`));
  return have || auth.license_accepted ? null : "license";
}

/** A channel that says nothing any more has no "only here" parts either. */
function tidy(ch: ChannelState): ChannelState {
  if (isChanged(ch) && ch.ranges?.length) return ch;
  return { volume: ch.volume, muted: ch.muted, solo: ch.solo };
}

/** "Remove a sound here": the tracks to pick from, for a selection made on no track in particular. */
function RemoveHereMenu({ options, onPick }: { options: { key: string; name: string }[]; onPick: (key: string) => void }) {
  return (
    <Menu
      label="Remove a sound in the selected part"
      className={styles.menuButton}
      align="start"
      items={options.map((option) => ({ label: `Remove ${option.name} here`, onSelect: () => onPick(option.key) }))}
    >
      Remove a sound here...
    </Menu>
  );
}

/** The mix in plain words, for the bar at the bottom. */
function describeMix(keys: string[], mixer: Mixer, label: (key: string) => string): string {
  const changed = keys.filter((key) => isChanged(mixer[key] ?? DEFAULT_CHANNEL));
  if (!changed.length) return "The original, with nothing changed";
  const parts = changed.map((key) => {
    const ch = mixer[key];
    const count = ch.ranges?.length ?? 0;
    const where = count ? ` in ${count} part${count > 1 ? "s" : ""}` : "";
    if (ch.solo) return `only ${label(key)}${where}`;
    if (ch.muted || ch.volume <= 0) return `no ${label(key)}${where}`;
    return `${label(key)} at ${formatLevel(ch.volume)}${where}`;
  });
  const text = parts.join(", ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

interface Compose {
  pending: PendingPrompt[];
  hints: Region[];
}

export function Workspace({
  session,
  onSessionChange,
  onRefresh,
  system,
  auth,
  onConnect,
  onLicense,
  onOpenBatch,
}: {
  session: Session;
  onSessionChange: (session: Session) => void;
  onRefresh: () => void;
  system: SystemStatus | null;
  auth: AuthStatus | null;
  onConnect: () => void;
  onLicense: () => void;
  onOpenBatch: (id: string) => void;
}) {
  const source = session.source;
  const duration = source.duration;
  const [player] = useState(() => new MultiTrackPlayer(duration));
  const history = useHistory();

  // What can be undone: the mix, and the sounds being described (with their hints).
  const [mixer, setMixerState] = useState<Mixer>(() => loadPref(`mix.${session.id}`, {}));
  const [compose, setComposeState] = useState<Compose>({ pending: [], hints: [] });
  const mixerRef = useRef(mixer);
  const composeRef = useRef(compose);
  const applyMixer = useCallback(
    (next: Mixer) => {
      mixerRef.current = next;
      setMixerState(next);
      savePref(`mix.${session.id}`, next);
    },
    [session.id],
  );
  const applyCompose = useCallback((next: Compose) => {
    composeRef.current = next;
    setComposeState(next);
  }, []);
  const commitMixer = useCallback(
    (change: (current: Mixer) => Mixer, key?: string) => {
      const before = mixerRef.current;
      const after = change(before);
      if (after === before) return;
      applyMixer(after);
      history.push({ tag: "mix", key, undo: () => applyMixer(before), redo: () => applyMixer(after) });
    },
    [applyMixer, history],
  );
  const commitCompose = useCallback(
    (change: (current: Compose) => Compose) => {
      const before = composeRef.current;
      const after = change(before);
      if (after === before) return;
      applyCompose(after);
      history.push({ tag: "compose", undo: () => applyCompose(before), redo: () => applyCompose(after) });
    },
    [applyCompose, history],
  );
  const { pending, hints } = compose;

  const [holdOriginal, setHoldOriginal] = useState(false);
  const [markingFor, setMarking] = useState<{ key: string; kind: AnchorKind } | null>(null);
  // Marking ends by itself when its sound is sent, removed or undone.
  const marking = markingFor && pending.some((p) => p.key === markingFor.key) ? markingFor : null;
  const [selection, setSelection] = useState<Selection | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [partOnly, setPartOnly] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [loop, setLoop] = useState(false);
  const [settings, setSettingsState] = useState<SeparationSettings>(() => loadPref("settings.v12", DEFAULT_SETTINGS));
  const [layout, setLayoutState] = useState<LayoutPrefs>(() => loadPref("layout.v1", DEFAULT_LAYOUT));
  const [videoAspect, setVideoAspect] = useState<number | null>(null);
  const [stage, setStage] = useState({ width: 0, height: 0 });
  const [exportKind, setExportKind] = useState<ExportKind | null>(null);
  const [batchOpen, setBatchOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [rebuildOpen, setRebuildOpen] = useState(false);
  const [confirm, setConfirm] = useState<{ title: string; body: string; action: string; run: () => void } | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoPane = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const timeline = useRef<TimelineHandle>(null);
  const inPoint = useRef<number | null>(null);

  const setSettings = (next: SeparationSettings) => {
    setSettingsState(next);
    savePref("settings.v12", next);
  };
  const setLayout = (patch: Partial<LayoutPrefs>) =>
    setLayoutState((current) => {
      const next = { ...current, ...patch };
      savePref("layout.v1", next);
      return next;
    });
  const fail = (err: unknown) => setNotice(err instanceof Error ? err.message : String(err));

  const sounds = useMemo(() => groupSounds(session.tracks), [session.tracks]);
  const done = sounds.filter((sound) => sound.take);
  const colors = useMemo(() => {
    const map: Record<string, string> = {};
    sounds.forEach((sound, index) => {
      map[sound.id] = `var(--track-${(index % 8) + 1})`;
    });
    return map;
  }, [sounds]);
  const channel = (key: string) => mixer[key] ?? DEFAULT_CHANNEL;
  const mixKeys = useMemo(
    () => [...sounds.filter((s) => s.take).map((s) => s.id), ...(session.rest ? ["rest"] : [])],
    [sounds, session.rest],
  );
  const label = (key: string) =>
    key === "rest" ? "everything else" : key === "original" ? "the original" : sounds.find((s) => s.id === key)?.prompt ?? key;
  const maxZoom = Math.max(16, Math.ceil(duration / 2));

  // Every source stays loaded and in sync; the gains pick which ones you hear (see playPlan).
  const sources = [
    { key: "original", url: apiUrl(source.audio_url) },
    ...done.map((s) => ({ key: s.id, url: apiUrl(s.take!.audio_url!) })),
    ...done.filter((s) => s.take!.without_url).map((s) => ({ key: withoutKey(s.id), url: apiUrl(s.take!.without_url!) })),
    ...(session.rest ? [{ key: "rest", url: apiUrl(session.rest.audio_url) }] : []),
  ];
  // Effects key on these strings, so a re-render with the same sources/gains is a no-op.
  const sourcesKey = JSON.stringify(sources);
  const segments = useMemo(() => mixSegments(mixKeys, mixer, duration), [mixKeys, mixer, duration]);
  const hasRest = session.rest !== null;
  const playKey = JSON.stringify(
    segments.map((segment) => ({
      start: segment.start,
      end: segment.end,
      gains: playPlan(sounds, hasRest, segment.gains, holdOriginal),
    })),
  );

  useEffect(() => {
    player.setSources(JSON.parse(sourcesKey));
  }, [player, sourcesKey]);
  useEffect(() => {
    player.setMix(JSON.parse(playKey));
  }, [player, playKey]);
  useEffect(() => {
    player.setLoop(loop);
  }, [player, loop]);
  useEffect(() => {
    player.setBounds(selection ? [selection.start, selection.end] : null);
  }, [player, selection]);
  useEffect(() => {
    player.attachVideo(videoRef.current);
    return () => player.attachVideo(null);
  }, [player, layout.videoShown]);
  useEffect(() => () => player.reset(), [player]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setStage({ width: el.clientWidth, height: el.clientHeight }));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 9000);
    return () => clearTimeout(timer);
  }, [notice]);

  // ------------------------------------------------------------------ the mix

  const setChannel = (key: string, patch: Partial<ChannelState>, historyKey?: string) =>
    commitMixer((current) => ({ ...current, [key]: tidy({ ...DEFAULT_CHANNEL, ...current[key], ...patch }) }), historyKey);

  /** Limit a track's change to the selection (or make the change there, if it has none yet). */
  const changeHere = (key: string, patch: Partial<ChannelState>) => {
    if (!selection) return;
    const span: Span = [selection.start, selection.end];
    commitMixer((current) => {
      const ch = { ...DEFAULT_CHANNEL, ...current[key] };
      const ranged = !!ch.ranges?.length;
      const base = isChanged(ch) ? ch : { ...ch, ...patch };
      return { ...current, [key]: { ...base, ranges: mergeSpans([...(ranged ? ch.ranges! : []), span]) } };
    });
    setFocus(key);
  };

  const removeRange = (key: string, index: number) =>
    commitMixer((current) => {
      const ch = current[key];
      if (!ch?.ranges) return current;
      const ranges = ch.ranges.filter((_, i) => i !== index);
      // Without parts left there is nothing "only here" about it: the track plays as recorded again.
      return { ...current, [key]: ranges.length ? { ...ch, ranges } : { ...DEFAULT_CHANNEL } };
    });

  // ----------------------------------------------- sounds being described

  const addPrompt = (text: string, sound?: string): string | null => {
    const lower = text.toLowerCase();
    const existing = composeRef.current.pending.find((p) => p.text.toLowerCase() === lower && p.sound === sound);
    if (existing) return null;
    const key = uid("p");
    commitCompose((current) => ({ ...current, pending: [...current.pending, { key, text, sound }] }));
    return key;
  };
  const removePrompt = (key: string) => {
    commitCompose((current) => ({
      pending: current.pending.filter((p) => p.key !== key),
      hints: current.hints.filter((h) => h.promptKey !== key),
    }));
    if (marking?.key === key) setMarking(null);
  };
  const addHint = (hint: Omit<Region, "id">) =>
    commitCompose((current) => ({ ...current, hints: [...current.hints, { ...hint, id: uid("r") }] }));
  const removeHint = (id: string) =>
    commitCompose((current) => ({ ...current, hints: current.hints.filter((h) => h.id !== id) }));

  /** Mark hints for a sound on the timeline. A selection made beforehand becomes its first hint. */
  const startMarking = (key: string | null) => {
    if (!key) {
      setMarking(null);
      return;
    }
    if (selection) {
      addHint({ promptKey: key, kind: "+", start: selection.start, end: selection.end });
      setSelection(null);
    }
    setMarking({ key, kind: "+" });
  };
  const markAtPlayhead = () => {
    if (!marking) return;
    const start = Math.max(0, player.currentTime() - 2.5);
    addHint({ promptKey: marking.key, kind: marking.kind, start, end: Math.min(duration, start + 5) });
  };

  const anchorsOf = (key: string): Anchor[] =>
    hints
      .filter((h) => h.promptKey === key)
      .map((h) => [h.kind, Math.round(h.start * 1000) / 1000, Math.round(h.end * 1000) / 1000]);
  const trackOf = (soundId: string) => session.tracks.find((t) => (t.sound ?? t.id) === soundId)?.id;

  const separate = async (typed?: string): Promise<boolean> => {
    const prompts = [...pending];
    if (typed && !prompts.some((p) => p.text.toLowerCase() === typed.toLowerCase())) {
      prompts.push({ key: uid("p"), text: typed });
    }
    if (!prompts.length) return false;
    const part = partOnly && selection && selection.end - selection.start >= MIN_PART_SECONDS;
    setSubmitting(true);
    try {
      const res = await api.separate(session.id, {
        ...settings,
        prompts: prompts.map((p) => ({
          text: p.text,
          anchors: anchorsOf(p.key),
          retake_of: p.sound ? trackOf(p.sound) : undefined,
        })),
        range: part ? [selection.start, selection.end] : null,
      });
      onSessionChange(res.session);
      applyCompose({ pending: [], hints: [] });
      history.drop("compose");
      setMarking(null);
      setPartOnly(false);
      return true;
    } catch (err) {
      fail(err);
      return false;
    } finally {
      setSubmitting(false);
    }
  };

  // ------------------------------------------------------------ takes

  /** Another take of a sound: the same description, the settings in the panel and another seed. */
  const retake = async (sound: Sound, range?: Span | null) => {
    const from = sound.take ?? sound.failed[sound.failed.length - 1];
    if (!from) return;
    try {
      const res = await api.separate(session.id, {
        ...settings,
        seed: settings.seed !== null && settings.seed !== from.seed ? settings.seed : null,
        prompts: [{ text: sound.prompt, anchors: from.anchors, retake_of: from.id }],
        range: range === undefined ? from.range : range,
      });
      onSessionChange(res.session);
      if (!sound.take) {
        for (const failed of sound.failed) await api.removeTrack(session.id, failed.id).catch(() => undefined);
        onRefresh();
      }
    } catch (err) {
      fail(err);
    }
  };

  const retakeWithHints = (sound: Sound) => {
    const take = sound.take;
    if (!take) return;
    const key = uid("p");
    commitCompose((current) => ({
      pending: [...current.pending.filter((p) => p.sound !== sound.id), { key, text: sound.prompt, sound: sound.id }],
      hints: [
        ...current.hints,
        ...take.anchors.map(([kind, start, end]) => ({ id: uid("r"), promptKey: key, kind, start, end })),
      ],
    }));
    setSelection(null);
    setMarking({ key, kind: "+" });
    if (!layout.sideShown) setLayout({ sideShown: true });
  };

  const activate = useCallback(
    async (takeId: string) => {
      try {
        onSessionChange(await api.activateTake(session.id, takeId));
      } catch (err) {
        setNotice(err instanceof Error ? err.message : String(err));
      }
    },
    [onSessionChange, session.id],
  );
  const chooseTake = (sound: Sound, take: Track) => {
    const previous = sound.take?.id;
    if (!previous || previous === take.id) return;
    void activate(take.id);
    history.push({ tag: "take", undo: () => void activate(previous), redo: () => void activate(take.id) });
  };

  const drop = async (ids: string[]) => {
    for (const id of ids) {
      try {
        await api.removeTrack(session.id, id);
      } catch (err) {
        fail(err);
      }
    }
    history.drop("take");
    onRefresh();
  };

  const rest = session.rest;
  const chooseRest = async (rev: number) => {
    try {
      onSessionChange(await api.activateRest(session.id, rev));
    } catch (err) {
      fail(err);
    }
  };
  const deleteRestVersion = async (rev: number) => {
    if (!rest) return;
    try {
      if (rev === rest.rev) {
        const others = rest.alts.map((alt) => alt.rev).sort((a, b) => a - b);
        const next = [...others].reverse().find((r) => r < rev) ?? others.find((r) => r > rev);
        if (next === undefined) return;
        await api.activateRest(session.id, next);
      }
      onSessionChange(await api.deleteRest(session.id, rev));
    } catch (err) {
      fail(err);
    }
  };

  const actions: LaneActions = {
    onSolo: (key) => setChannel(key, { solo: !channel(key).solo }),
    onMute: (key) => setChannel(key, { muted: !channel(key).muted }),
    onVolume: (key, volume) => setChannel(key, { volume }, `volume:${key}`),
    onRemoveRange: removeRange,
    onDownload: (key) => {
      const id = key === "rest" ? "rest" : sounds.find((s) => s.id === key)?.take?.id;
      if (!id) return;
      downloadExport(session.id, { kind: "mix", format: "wav", gains: { [id]: 1 }, tracks: [] }).catch(fail);
    },
    onRetry: (sound) => void retake(sound),
    onRetryWithHints: retakeWithHints,
    onWholeFile: (sound) => void retake(sound, null),
    onTake: chooseTake,
    onRemoveTake: (take) =>
      setConfirm({
        title: `Delete this take of ${take.prompt}?`,
        body: "Its audio is deleted. The other takes of this sound stay.",
        action: "Delete the take",
        run: () => void drop([take.id]),
      }),
    onRemoveSound: (sound) =>
      setConfirm({
        title: `Remove ${sound.prompt}?`,
        body:
          sound.takes.length > 1
            ? `Its ${sound.takes.length} takes are deleted, and the sound goes back into everything else.`
            : "Its track is deleted, and the sound goes back into everything else.",
        action: "Remove the sound",
        run: () =>
          void drop(session.tracks.filter((t) => (t.sound ?? t.id) === sound.id).map((t) => t.id)),
      }),
    onDrop: (take) => void drop([take.id]),
    onUseSeed: (seed) => {
      setSettings({ ...settings, seed });
      setNotice(`Seed ${seed} is set under Advanced: the next separation with the same settings repeats that take.`);
    },
    onRebuildRest: () => setRebuildOpen(true),
    onCancelRebuild: () => {
      api
        .cancelRebuild(session.id)
        .then((res) => onSessionChange(res.session))
        .catch(fail);
    },
    onRestVersion: (rev) => void chooseRest(rev),
    onDeleteRestVersion: (rev) => void deleteRestVersion(rev),
    onHoldOriginal: setHoldOriginal,
    onAddSound: () => {
      if (!layout.sideShown) setLayout({ sideShown: true });
      setTimeout(() => inputRef.current?.focus(), 0);
    },
  };

  // --------------------------------------------------------- keyboard

  // The handler below is installed once: it reads the latest state through these.
  const keys = useRef({ focus, selection, marking, loop, mixKeys, sounds, hasRest });
  const setChannelRef = useRef(setChannel);
  useEffect(() => {
    keys.current = { focus, selection, marking, loop, mixKeys, sounds, hasRest };
    setChannelRef.current = setChannel;
  });
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (document.querySelector("dialog[open]")) return;
      if (target?.closest("input, textarea, select, [contenteditable='true'], [role='menu']")) {
        if (event.key === "Escape" && target.closest("input[type='text']")) target.blur();
        return;
      }
      const state = keys.current;
      const mod = event.ctrlKey || event.metaKey;
      const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
      if (mod && key === "z") {
        event.preventDefault();
        if (event.shiftKey) history.redo();
        else history.undo();
        return;
      }
      if (mod && key === "y") {
        event.preventDefault();
        history.redo();
        return;
      }
      if (mod || event.altKey) return;
      const onControl = !!target?.closest("button, a, [role='separator']");
      if (onControl && (key === " " || key === "Enter" || key.startsWith("Arrow"))) return;
      const lanes = ["original", ...state.sounds.map((s) => s.id), ...(state.hasRest ? ["rest"] : [])];
      const mixable = state.focus && state.mixKeys.includes(state.focus) ? state.focus : null;
      const now = player.currentTime();
      let handled = true;
      switch (key) {
        case " ":
          if (event.repeat) break;
          player.toggle();
          break;
        case "Home":
          player.seek(0);
          break;
        case "End":
          player.seek(duration);
          break;
        case "ArrowLeft":
          player.seek(now - (event.shiftKey ? 5 : 1));
          break;
        case "ArrowRight":
          player.seek(now + (event.shiftKey ? 5 : 1));
          break;
        case "ArrowUp":
        case "ArrowDown": {
          const index = state.focus ? lanes.indexOf(state.focus) : -1;
          const next = key === "ArrowDown" ? Math.min(lanes.length - 1, index + 1) : Math.max(0, index - 1);
          setFocus(lanes[next] ?? null);
          break;
        }
        case "l":
          setLoop(!state.loop);
          break;
        case "i":
          inPoint.current = now;
          if (state.selection && state.selection.end > now + 0.05) setSelection({ ...state.selection, start: now });
          break;
        case "o": {
          const start = state.selection ? state.selection.start : inPoint.current ?? 0;
          if (now > start + 0.05) setSelection({ start, end: now, lane: state.selection?.lane ?? state.focus });
          break;
        }
        case "m":
          if (mixable) setChannelRef.current(mixable, { muted: !(mixerRef.current[mixable]?.muted ?? false) });
          break;
        case "s":
          if (mixable) setChannelRef.current(mixable, { solo: !(mixerRef.current[mixable]?.solo ?? false) });
          break;
        case "Escape":
          if (state.marking) setMarking(null);
          else if (state.selection) setSelection(null);
          else handled = false;
          break;
        case "+":
        case "=":
          timeline.current?.zoomBy(2);
          break;
        case "-":
          timeline.current?.zoomBy(0.5);
          break;
        case "0":
          setZoom(1);
          break;
        case "?":
          setShortcutsOpen(true);
          break;
        default:
          handled = false;
      }
      if (handled) event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player, duration, history]);

  // ----------------------------------------------------------- export

  // What plays outside every "only here" part, and the parts that differ from it.
  const baseGains = channelGains(mixKeys, mixer, -1);
  const baseKey = JSON.stringify(baseGains);
  const exportGains = takeGains(baseGains, sounds);
  const exportSegments = segments
    .filter((segment) => JSON.stringify(segment.gains) !== baseKey)
    .map((segment) => ({ start: segment.start, end: segment.end, gains: takeGains(segment.gains, sounds) }));
  const summary = done.length ? describeMix(mixKeys, mixer, label) : "The original (no tracks yet)";

  const blocker = !system
    ? null
    : !system.reachable
      ? "The backend is offline: start AudioGhost with start.bat (or start.sh)."
      : !system.redis
        ? "Redis is not running, so jobs can't be queued."
        : null;

  const estimate = (() => {
    const count = pending.length || 1;
    const similar = done
      .map((s) => s.take!)
      .filter((t) => t.elapsed && !t.range && t.settings.model_size === settings.model_size && t.settings.quality === settings.quality);
    const busy = session.tracks.some((t) => t.status === "running" || t.status === "queued");
    if (!similar.length) return busy ? "Starts after the current job" : null;
    const perSound = similar.reduce((sum, t) => sum + (t.elapsed ?? 0), 0) / similar.length;
    return `About ${formatDuration(perSound * count)}${busy ? " after the current job" : ""}`;
  })();

  // ----------------------------------------------------------- layout

  const hasVideo = source.is_video && !!source.video_url;
  const beside = hasVideo && videoAspect !== null && videoAspect < 0.9; // a portrait video sits beside the tracks
  const showVideo = hasVideo && layout.videoShown;
  const videoMax = beside ? Math.max(160, stage.width * 0.55) : Math.max(120, stage.height * 0.7);
  // Until it is resized by hand, the video takes the room its shape can use.
  const videoAuto = beside
    ? Math.max(220, Math.min(stage.height * (videoAspect ?? 0.5625), stage.width * 0.3))
    : Math.max(160, Math.min(stage.width / (videoAspect ?? 16 / 9), stage.height * 0.42));
  const videoSize = Math.round(Math.min((beside ? layout.videoWidth : layout.videoHeight) ?? videoAuto, videoMax));
  const fullscreen = () => {
    void videoPane.current?.requestFullscreen?.().catch(() => undefined);
  };

  // ----------------------------------------------- the bar above the tracks

  const activePrompt = marking ? pending.find((p) => p.key === marking.key) : pending[pending.length - 1];
  const focusedKey = focus && mixKeys.includes(focus) ? focus : null;
  const focusedChannel = focusedKey ? channel(focusedKey) : null;
  const selectionLength = selection ? selection.end - selection.start : 0;

  const contextBar = marking ? (
    <div className={`${styles.context} ${styles.contextMarking}`} role="region" aria-label="Marking hints">
      <div className={styles.contextText}>
        <strong>Drag on the timeline</strong> over the parts where you can {marking.kind === "+" ? "" : "not "}hear{" "}
        <strong>{activePrompt?.text ?? "the sound"}</strong>.
        <span className={styles.contextTip}>
          Hints help with short, separate sounds (a bark, a door). For speech or music that runs through, they can make
          the result worse.
        </span>
      </div>
      <div role="group" aria-label="What the marks mean" className={styles.toolGroup}>
        <button type="button" className={styles.tool} aria-pressed={marking.kind === "+"} onClick={() => setMarking({ ...marking, kind: "+" })}>
          <Crosshair size={15} aria-hidden="true" />
          It is here
        </button>
        <button
          type="button"
          className={`${styles.tool} ${styles.toolOut}`}
          aria-pressed={marking.kind === "-"}
          onClick={() => setMarking({ ...marking, kind: "-" })}
        >
          <Ban size={15} aria-hidden="true" />
          Not here
        </button>
      </div>
      <Button size="small" onClick={markAtPlayhead} title="Marks 5 seconds around the playhead">
        Mark at playhead
      </Button>
      <Button size="small" variant="primary" onClick={() => setMarking(null)}>
        Done
      </Button>
    </div>
  ) : selection ? (
    <div className={styles.context} role="region" aria-label="Selection">
      <div className={`${styles.contextText} tabular`}>
        <strong>{formatSpan(selection.start, selection.end)}</strong>
        <span className={styles.contextMuted}> · {formatDuration(selectionLength)} selected</span>
      </div>
      <Button size="small" onClick={() => void player.play()} title="Space plays the selection">
        <Play size={14} aria-hidden="true" />
        Play
      </Button>
      <Button size="small" onClick={() => timeline.current?.zoomToSpan(selection.start, selection.end)}>
        <ScanLine size={14} aria-hidden="true" />
        Zoom to it
      </Button>
      {(activePrompt || focusedKey || done.length > 0) && <div className={styles.divider} />}
      {activePrompt ? (
        <>
          <Button size="small" onClick={() => addHint({ promptKey: activePrompt.key, kind: "+", start: selection.start, end: selection.end })}>
            <Crosshair size={14} aria-hidden="true" />
            {activePrompt.text} is here
          </Button>
          <Button size="small" onClick={() => addHint({ promptKey: activePrompt.key, kind: "-", start: selection.start, end: selection.end })}>
            <Ban size={14} aria-hidden="true" />
            Not here
          </Button>
        </>
      ) : focusedKey && focusedChannel ? (
        isChanged(focusedChannel) ? (
          <Button size="small" onClick={() => changeHere(focusedKey, {})}>
            {focusedChannel.ranges?.length ? `Also here: ${label(focusedKey)}` : `Change ${label(focusedKey)} here only`}
          </Button>
        ) : (
          <>
            <Button size="small" onClick={() => changeHere(focusedKey, { muted: true })}>
              Remove {label(focusedKey)} here
            </Button>
            <Button size="small" onClick={() => changeHere(focusedKey, { solo: true })}>
              Only {label(focusedKey)} here
            </Button>
          </>
        )
      ) : done.length > 0 ? (
        <RemoveHereMenu
          options={mixKeys.map((key) => ({ key, name: label(key) }))}
          onPick={(key) => changeHere(key, { muted: true })}
        />
      ) : null}
      <div className={styles.spacer} />
      <IconButton label="Clear the selection (Esc)" size="small" onClick={() => setSelection(null)}>
        <X size={15} aria-hidden="true" />
      </IconButton>
    </div>
  ) : (
    // Always there, so the tracks don't jump when a selection appears under the pointer.
    <div className={`${styles.context} ${styles.contextIdle}`}>
      <div className={styles.contextText}>
        <span className={styles.contextMuted}>
          Drag on a track to select a part: play it, loop it, or change the mix there only. Drag the ruler to move the
          playhead.
        </span>
      </div>
    </div>
  );

  return (
    <div className={styles.workspace} style={{ "--side-width": `${layout.sideWidth}px` } as React.CSSProperties}>
      <main className={styles.main}>
        <div className={styles.stage} ref={stageRef} data-video={showVideo ? (beside ? "beside" : "above") : undefined}>
          {showVideo && (
            <>
              <div
                className={styles.videoPane}
                ref={videoPane}
                style={beside ? { width: videoSize } : { height: videoSize }}
              >
                <video
                  ref={videoRef}
                  className={styles.video}
                  src={apiUrl(source.video_url!)}
                  muted
                  playsInline
                  preload="auto"
                  aria-label="Video preview, plays along with your mix"
                  onLoadedMetadata={(event) => {
                    const el = event.currentTarget;
                    if (el.videoWidth && el.videoHeight) setVideoAspect(el.videoWidth / el.videoHeight);
                  }}
                  onClick={() => player.toggle()}
                />
                <div className={styles.videoTools}>
                  <IconButton label="Full screen" size="small" onClick={fullscreen}>
                    <Maximize2 size={15} aria-hidden="true" />
                  </IconButton>
                  <IconButton label="Hide the video" size="small" onClick={() => setLayout({ videoShown: false })}>
                    <VideoOff size={15} aria-hidden="true" />
                  </IconButton>
                </div>
              </div>
              <Splitter
                orientation={beside ? "vertical" : "horizontal"}
                label="Resize the video"
                value={videoSize}
                min={beside ? 140 : 100}
                max={videoMax}
                onChange={(value) => setLayout(beside ? { videoWidth: value } : { videoHeight: value })}
                onReset={() => setLayout(beside ? { videoWidth: null } : { videoHeight: null })}
              />
            </>
          )}

          <div className={styles.editor}>
            <div className={styles.transport}>
              <IconButton label="Back to the start (Home)" onClick={() => player.seek(selection ? selection.start : 0)}>
                <SkipBack size={18} aria-hidden="true" />
              </IconButton>
              <PlayButton player={player} />
              <Clock player={player} duration={duration} />
              <div className={styles.spacer} />
              <IconButton label="Undo (Ctrl+Z)" disabled={!history.canUndo} onClick={() => history.undo()}>
                <Undo2 size={18} aria-hidden="true" />
              </IconButton>
              <IconButton label="Redo (Ctrl+Shift+Z)" disabled={!history.canRedo} onClick={() => history.redo()}>
                <Redo2 size={18} aria-hidden="true" />
              </IconButton>
              <div className={styles.divider} />
              <IconButton
                label={selection ? "Loop the selection (L)" : "Loop playback (L)"}
                aria-pressed={loop}
                onClick={() => setLoop(!loop)}
              >
                <Repeat size={18} aria-hidden="true" />
              </IconButton>
              <IconButton label="Zoom out (-)" disabled={zoom <= 1} onClick={() => timeline.current?.zoomBy(0.5)}>
                <ZoomOut size={18} aria-hidden="true" />
              </IconButton>
              <IconButton label="Zoom in (+)" disabled={zoom >= maxZoom} onClick={() => timeline.current?.zoomBy(2)}>
                <ZoomIn size={18} aria-hidden="true" />
              </IconButton>
              <div className={styles.divider} />
              {hasVideo && !layout.videoShown && (
                <IconButton label="Show the video" onClick={() => setLayout({ videoShown: true })}>
                  <Video size={18} aria-hidden="true" />
                </IconButton>
              )}
              <IconButton label="Keyboard shortcuts (?)" onClick={() => setShortcutsOpen(true)}>
                <Keyboard size={18} aria-hidden="true" />
              </IconButton>
              <IconButton
                label={layout.sideShown ? "Hide the panel" : "Show the panel"}
                className={styles.sideToggle}
                onClick={() => setLayout({ sideShown: !layout.sideShown })}
              >
                {layout.sideShown ? <PanelRightClose size={18} aria-hidden="true" /> : <PanelRightOpen size={18} aria-hidden="true" />}
              </IconButton>
            </div>
            {contextBar}
            {notice && (
              <div className={styles.notice} role="status">
                <span>{notice}</span>
                <IconButton label="Dismiss" size="tiny" onClick={() => setNotice(null)}>
                  <X size={14} aria-hidden="true" />
                </IconButton>
              </div>
            )}
            <div className={styles.tracksArea}>
              <Timeline
                ref={timeline}
                session={session}
                sounds={sounds}
                player={player}
                colors={colors}
                mixer={mixer}
                holdOriginal={holdOriginal}
                actions={actions}
                hints={hints}
                pending={pending}
                marking={marking}
                selection={selection}
                onSelection={setSelection}
                focus={focus}
                onFocus={setFocus}
                zoom={zoom}
                maxZoom={maxZoom}
                onZoom={setZoom}
                onAddHint={addHint}
                onRemoveHint={removeHint}
              />
            </div>
          </div>
        </div>

        <div className={styles.mixBar}>
          <div className={styles.mixText}>
            <SectionLabel as="span">What you hear</SectionLabel>
            <div className={styles.mixSummary}>{summary}</div>
          </div>
          <div className={styles.spacer} />
          <Button
            disabled={!done.length}
            onClick={() => setBatchOpen(true)}
            title="Separate the same sounds from more files, with this mix"
          >
            <Layers size={16} aria-hidden="true" />
            Batch...
          </Button>
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

      {layout.sideShown && (
        <>
          <div className={styles.sideSplit}>
            <Splitter
              orientation="vertical"
              label="Resize the panel"
              value={layout.sideWidth}
              min={300}
              max={560}
              direction={-1}
              onChange={(value) => setLayout({ sideWidth: value })}
              onReset={() => setLayout({ sideWidth: DEFAULT_LAYOUT.sideWidth })}
            />
          </div>
          <div className={styles.side}>
            <SeparatePanel
              session={session}
              sounds={sounds}
              pending={pending}
              hints={hints}
              marking={marking}
              selection={selection}
              partOnly={partOnly}
              onPartOnly={setPartOnly}
              settings={settings}
              onSettings={setSettings}
              onAddPrompt={addPrompt}
              onRemovePrompt={removePrompt}
              onRemoveHint={removeHint}
              onMark={startMarking}
              onHintFromSelection={(key, kind) => {
                if (selection) addHint({ promptKey: key, kind, start: selection.start, end: selection.end });
              }}
              onSeparate={separate}
              submitting={submitting}
              blocker={blocker}
              gate={modelGate(auth, settings)}
              onLicense={onLicense}
              onConnect={onConnect}
              inputRef={inputRef}
              estimate={estimate}
            />
          </div>
        </>
      )}

      {exportKind && (
        <ExportDialog
          key={exportKind}
          session={session}
          sounds={sounds}
          gains={exportGains}
          segments={exportSegments}
          summary={summary}
          open
          initialKind={exportKind}
          onClose={() => setExportKind(null)}
        />
      )}
      {batchOpen && (
        <BatchDialog
          session={session}
          sounds={sounds}
          gains={channelGains(mixKeys, mixer, null)}
          settings={settings}
          summary={summary}
          onClose={() => setBatchOpen(false)}
          onCreated={onOpenBatch}
        />
      )}
      {rebuildOpen && (
        <RebuildDialog
          sounds={done}
          order={rest?.tracks ?? []}
          settings={settings}
          onClose={() => setRebuildOpen(false)}
          onRebuild={async (order) => {
            const res = await api.rebuildRest(session.id, {
              model_size: settings.model_size,
              quality: settings.quality,
              variant: settings.variant,
              stereo: settings.stereo,
              use_float32: settings.use_float32,
              chunk_duration: settings.chunk_duration,
              seed: settings.seed,
              order,
            });
            onSessionChange(res.session);
          }}
        />
      )}
      <ShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      <Dialog
        open={!!confirm}
        onClose={() => setConfirm(null)}
        title={confirm?.title ?? ""}
        footer={
          <>
            <Button onClick={() => setConfirm(null)}>Cancel</Button>
            <Button
              danger
              onClick={() => {
                confirm?.run();
                setConfirm(null);
              }}
            >
              {confirm?.action}
            </Button>
          </>
        }
      >
        <p className={styles.dialogText}>{confirm?.body}</p>
      </Dialog>
    </div>
  );
}
