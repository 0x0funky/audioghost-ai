/**
 * Plays the original and every track of a session in sync.
 *
 * Each source is a streaming <audio> element routed through a WebAudio
 * GainNode, so a mix is just a set of gains (muting, solo and "hold to hear
 * the original" switch instantly, and gains above 1 boost). The original's
 * element is the clock: the others (and a muted <video>) are nudged back when
 * they drift more than a few frames.
 */

export interface PlayerSnapshot {
  playing: boolean;
  time: number;
}

interface Channel {
  key: string;
  url: string;
  el: HTMLAudioElement;
  node: MediaElementAudioSourceNode | null;
  gain: GainNode | null;
  target: number;
}

const AUDIO_DRIFT = 0.05;
const VIDEO_DRIFT = 0.1;

export class MultiTrackPlayer {
  readonly duration: number;
  private ctx: AudioContext | null = null;
  private channels = new Map<string, Channel>();
  private video: HTMLVideoElement | null = null;
  private listeners = new Set<() => void>();
  private snapshot: PlayerSnapshot = { playing: false, time: 0 };
  private raf = 0;
  private lastEmit = 0;
  private loop = false;
  /** Last gains set, so a source added later (a new "everything else") starts at its level. */
  private gains: Record<string, number> = {};

  constructor(duration: number) {
    this.duration = duration;
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = () => this.snapshot;

  private emit(partial: Partial<PlayerSnapshot>) {
    this.snapshot = { ...this.snapshot, ...partial };
    this.listeners.forEach((listener) => listener());
  }

  private clock(): HTMLAudioElement | null {
    const original = this.channels.get("original");
    if (original) return original.el;
    const first = this.channels.values().next();
    return first.done ? null : first.value.el;
  }

  currentTime(): number {
    const clock = this.clock();
    return clock && clock.readyState > 0 ? clock.currentTime : this.snapshot.time;
  }

  /** Make the player play exactly these sources (keyed by "original", "rest" or a track id). */
  setSources(sources: { key: string; url: string }[]) {
    const wanted = new Map(sources.map((s) => [s.key, s.url]));
    for (const [key, channel] of [...this.channels]) {
      if (wanted.get(key) !== channel.url) this.dispose(key);
    }
    for (const [key, url] of wanted) {
      if (!this.channels.has(key)) this.add(key, url);
    }
  }

  private add(key: string, url: string) {
    const el = new Audio();
    el.crossOrigin = "anonymous";
    el.preload = "auto";
    const target = Math.max(0, this.gains[key] ?? 0);
    const channel: Channel = { key, url, el, node: null, gain: null, target };
    el.volume = Math.min(1, target);
    el.src = url;
    this.channels.set(key, channel);
    el.addEventListener(
      "loadedmetadata",
      () => {
        el.currentTime = this.currentTime();
        if (this.snapshot.playing) void el.play().catch(() => undefined);
      },
      { once: true },
    );
    if (key === "original") el.addEventListener("ended", this.onEnded);
    if (this.ctx) this.connect(channel);
  }

  private connect(channel: Channel) {
    if (!this.ctx || channel.node) return;
    channel.node = this.ctx.createMediaElementSource(channel.el);
    channel.gain = this.ctx.createGain();
    channel.gain.gain.value = channel.target;
    channel.node.connect(channel.gain).connect(this.ctx.destination);
    channel.el.volume = 1;
  }

  private dispose(key: string) {
    const channel = this.channels.get(key);
    if (!channel) return;
    channel.el.pause();
    channel.el.removeEventListener("ended", this.onEnded);
    channel.node?.disconnect();
    channel.gain?.disconnect();
    channel.el.removeAttribute("src");
    channel.el.load();
    this.channels.delete(key);
  }

  /** Linear gain per source key; sources not listed are silent. */
  setGains(gains: Record<string, number>) {
    this.gains = { ...gains };
    for (const channel of this.channels.values()) {
      channel.target = Math.max(0, gains[channel.key] ?? 0);
      if (channel.gain && this.ctx) {
        channel.gain.gain.setTargetAtTime(channel.target, this.ctx.currentTime, 0.012);
      } else {
        channel.el.volume = Math.min(1, channel.target);
      }
    }
  }

  attachVideo(el: HTMLVideoElement | null) {
    this.video = el;
    if (!el) return;
    el.muted = true;
    const sync = () => {
      el.currentTime = this.currentTime();
      if (this.snapshot.playing) void el.play().catch(() => undefined);
    };
    if (el.readyState > 0) sync();
    else el.addEventListener("loadedmetadata", sync, { once: true });
  }

  async play() {
    if (this.snapshot.playing) return;
    if (!this.ctx) {
      this.ctx = new AudioContext();
      for (const channel of this.channels.values()) this.connect(channel);
    }
    if (this.ctx.state === "suspended") await this.ctx.resume();
    let start = this.currentTime();
    if (start >= this.duration - 0.05) start = 0;
    for (const channel of this.channels.values()) {
      if (channel.el.readyState > 0) channel.el.currentTime = start;
    }
    if (this.video && this.video.readyState > 0) this.video.currentTime = start;
    this.emit({ playing: true, time: start });
    await Promise.all([...this.channels.values()].map((c) => c.el.play().catch(() => undefined)));
    if (this.video) void this.video.play().catch(() => undefined);
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.tick);
  }

  pause() {
    for (const channel of this.channels.values()) channel.el.pause();
    this.video?.pause();
    cancelAnimationFrame(this.raf);
    this.emit({ playing: false, time: this.currentTime() });
  }

  toggle() {
    if (this.snapshot.playing) this.pause();
    else void this.play();
  }

  seek(time: number) {
    const t = Math.max(0, Math.min(this.duration, time));
    for (const channel of this.channels.values()) {
      if (channel.el.readyState > 0) channel.el.currentTime = t;
    }
    if (this.video && this.video.readyState > 0) this.video.currentTime = t;
    this.emit({ time: t });
  }

  setLoop(loop: boolean) {
    this.loop = loop;
  }

  private onEnded = () => {
    if (this.loop) {
      this.seek(0);
      for (const channel of this.channels.values()) void channel.el.play().catch(() => undefined);
      if (this.video) void this.video.play().catch(() => undefined);
      return;
    }
    this.pause();
    this.emit({ time: this.duration });
  };

  private tick = () => {
    const clock = this.clock();
    if (!clock || !this.snapshot.playing) return;
    const t = clock.currentTime;
    for (const channel of this.channels.values()) {
      const el = channel.el;
      if (el !== clock && el.readyState > 0 && Math.abs(el.currentTime - t) > AUDIO_DRIFT) {
        el.currentTime = t;
        if (el.paused) void el.play().catch(() => undefined);
      }
    }
    if (this.video && this.video.readyState > 0 && Math.abs(this.video.currentTime - t) > VIDEO_DRIFT) {
      this.video.currentTime = t;
    }
    const now = performance.now();
    if (now - this.lastEmit > 33) {
      this.lastEmit = now;
      this.emit({ time: t });
    }
    this.raf = requestAnimationFrame(this.tick);
  };

  /** Stop and release everything; the player can be reused afterwards. */
  reset() {
    this.pause();
    for (const key of [...this.channels.keys()]) this.dispose(key);
    const ctx = this.ctx;
    this.ctx = null;
    if (ctx) void ctx.close().catch(() => undefined);
  }
}
