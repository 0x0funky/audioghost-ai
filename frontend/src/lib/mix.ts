import type { Track } from "./types";

const unity = (value: number) => Math.abs(value - 1) < 1e-3;

/**
 * Player gains per source for a mix. Everything played is model output; the
 * source depends on what changed (the same rule as the server's exports,
 * `sessions.mix_plan`):
 *
 * - nothing changed: the original itself
 * - one track changed: the model's own "without that sound" + the track at its gain
 * - more: "everything else" (the chain) + every track at its gain
 *
 * `gains` holds the mixer's linear gain per track id and "rest" (muted or
 * soloed-out = 0). Sources missing from the result play silent.
 */
export function playPlan(
  done: Track[],
  hasRest: boolean,
  gains: Record<string, number>,
  holdOriginal: boolean,
): Record<string, number> {
  if (holdOriginal || done.length === 0 || !hasRest) return { original: 1 };
  const gain = (key: string) => gains[key] ?? 0;
  const changed = done.filter((track) => !unity(gain(track.id)));
  if (changed.length === 0 && unity(gain("rest"))) return { original: 1 };
  if (changed.length === 1 && unity(gain("rest")) && changed[0].without_url) {
    const track = changed[0];
    return { [withoutKey(track.id)]: 1, [track.id]: gain(track.id) };
  }
  const plan: Record<string, number> = { rest: gain("rest") };
  for (const track of done) plan[track.id] = gain(track.id);
  return plan;
}

export const withoutKey = (trackId: string) => `without:${trackId}`;
