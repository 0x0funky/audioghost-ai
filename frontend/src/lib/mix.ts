import { sliderToGain } from "./format";
import type { ChannelState, Mixer, Sound, Span, Track } from "./types";

export const DEFAULT_CHANNEL: ChannelState = { volume: 75, muted: false, solo: false };
const unity = (value: number) => Math.abs(value - 1) < 1e-3;

export const withoutKey = (soundId: string) => `without:${soundId}`;

/** A track with something to say in the mix: muted, soloed or not at its recorded level. */
export const isChanged = (ch: ChannelState) => ch.muted || ch.solo || ch.volume !== DEFAULT_CHANNEL.volume;

const inside = (ranges: Span[], t: number) => ranges.some(([start, end]) => t >= start && t < end);

/** A channel's state at time `t`: its change counts only inside its "only here" parts. */
function at(ch: ChannelState | undefined, t: number | null): ChannelState {
  if (!ch) return DEFAULT_CHANNEL;
  if (t !== null && ch.ranges?.length && !inside(ch.ranges, t)) return DEFAULT_CHANNEL;
  return ch;
}

/**
 * Linear gain per mixer key (a sound id or "rest") at time `t`: muted or
 * soloed-out = 0. `t` null ignores the "only here" parts.
 */
export function channelGains(keys: string[], mixer: Mixer, t: number | null): Record<string, number> {
  const states = keys.map((key) => at(mixer[key], t));
  const anySolo = states.some((ch) => ch.solo);
  const gains: Record<string, number> = {};
  keys.forEach((key, index) => {
    const ch = states[index];
    gains[key] = ch.muted || (anySolo && !ch.solo) ? 0 : sliderToGain(ch.volume);
  });
  return gains;
}

export interface MixSegment {
  start: number;
  end: number;
  gains: Record<string, number>;
}

const same = (a: Record<string, number>, b: Record<string, number>) =>
  Object.keys(a).length === Object.keys(b).length && Object.keys(a).every((key) => Math.abs(a[key] - (b[key] ?? NaN)) < 1e-6);

/** The mix over time: back-to-back stretches with constant gains (one when nothing is "only here"). */
export function mixSegments(keys: string[], mixer: Mixer, duration: number): MixSegment[] {
  const edges = new Set<number>([0, duration]);
  for (const key of keys) {
    for (const [start, end] of mixer[key]?.ranges ?? []) {
      edges.add(Math.max(0, Math.min(duration, start)));
      edges.add(Math.max(0, Math.min(duration, end)));
    }
  }
  const sorted = [...edges].sort((a, b) => a - b);
  const out: MixSegment[] = [];
  for (let i = 0; i + 1 < sorted.length; i++) {
    const [start, end] = [sorted[i], sorted[i + 1]];
    if (end - start < 1e-4) continue;
    const gains = channelGains(keys, mixer, (start + end) / 2);
    const last = out[out.length - 1];
    if (last && same(last.gains, gains)) last.end = end;
    else out.push({ start, end, gains });
  }
  return out.length ? out : [{ start: 0, end: duration, gains: channelGains(keys, mixer, null) }];
}

/**
 * Player gains per source for a mix. Everything played is model output; the
 * source depends on what changed (the same rule as the server's exports,
 * `sessions.mix_plan`):
 *
 * - nothing changed: the original itself
 * - one track changed: the model's own "without that sound" + the track at its gain
 * - more: "everything else" (the chain) + every track at its gain
 *
 * `gains` holds the mixer's linear gain per sound id and "rest". Sources
 * missing from the result play silent.
 */
export function playPlan(
  sounds: Sound[],
  hasRest: boolean,
  gains: Record<string, number>,
  holdOriginal: boolean,
): Record<string, number> {
  const done = sounds.filter((sound) => sound.take);
  if (holdOriginal || done.length === 0 || !hasRest) return { original: 1 };
  const gain = (key: string) => gains[key] ?? 0;
  const changed = done.filter((sound) => !unity(gain(sound.id)));
  if (changed.length === 0 && unity(gain("rest"))) return { original: 1 };
  if (changed.length === 1 && unity(gain("rest")) && changed[0].take?.without_url) {
    const sound = changed[0];
    return { [withoutKey(sound.id)]: 1, [sound.id]: gain(sound.id) };
  }
  const plan: Record<string, number> = { rest: gain("rest") };
  for (const sound of done) plan[sound.id] = gain(sound.id);
  return plan;
}

/** Mixer gains (per sound) as the server wants them: per id of the take that plays. */
export function takeGains(gains: Record<string, number>, sounds: Sound[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(gains)) {
    if (key === "rest") out.rest = value;
    else {
      const take = sounds.find((sound) => sound.id === key)?.take;
      if (take) out[take.id] = value;
    }
  }
  return out;
}

/** Tracks grouped into sounds, in the order the sounds were added. */
export function groupSounds(tracks: Track[]): Sound[] {
  const order: string[] = [];
  const bySound = new Map<string, Track[]>();
  for (const track of tracks) {
    const key = track.sound ?? track.id;
    if (!bySound.has(key)) {
      bySound.set(key, []);
      order.push(key);
    }
    bySound.get(key)!.push(track);
  }
  return order.map((id) => {
    const rows = bySound.get(id)!;
    const takes = rows.filter((t) => t.status === "done" && t.audio_url);
    const take = takes.find((t) => t.active) ?? takes[takes.length - 1] ?? null;
    return {
      id,
      prompt: (take ?? rows[rows.length - 1]).prompt,
      takes,
      take,
      pending: rows.filter((t) => t.status === "queued" || t.status === "running"),
      failed: rows.filter((t) => t.status === "failed"),
    };
  });
}

/** Sorted, with overlapping and touching parts joined. */
export function mergeSpans(spans: Span[]): Span[] {
  const sorted = spans.filter(([start, end]) => end - start > 1e-3).sort((a, b) => a[0] - b[0]);
  const out: Span[] = [];
  for (const [start, end] of sorted) {
    const last = out[out.length - 1];
    if (last && start <= last[1] + 1e-3) last[1] = Math.max(last[1], end);
    else out.push([start, end]);
  }
  return out;
}
