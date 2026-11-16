"use client";

import { memo, useMemo } from "react";

/** Mirrored waveform from a peaks overview, stretched to its box (no redraw on zoom). */
export const Waveform = memo(function Waveform({
  peaks,
  color,
  opacity = 0.9,
  points = 1600,
}: {
  peaks: number[] | null;
  color: string;
  opacity?: number;
  points?: number;
}) {
  const path = useMemo(() => {
    if (!peaks || peaks.length === 0) return null;
    const n = Math.min(points, peaks.length);
    const per = peaks.length / n;
    const top: string[] = [];
    const bottom: string[] = [];
    for (let i = 0; i < n; i++) {
      let max = 0;
      const end = Math.min(peaks.length, Math.floor((i + 1) * per));
      for (let j = Math.floor(i * per); j < end; j++) if (peaks[j] > max) max = peaks[j];
      const a = Math.max(0.02, Math.min(1, max)) * 30;
      const x = ((i + 0.5) / n) * 1000;
      top.push(`${x.toFixed(1)},${(32 - a).toFixed(1)}`);
      bottom.push(`${x.toFixed(1)},${(32 + a).toFixed(1)}`);
    }
    return `M0,32 L${top.join(" L")} L1000,32 L${bottom.reverse().join(" L")} Z`;
  }, [peaks, points]);

  if (!path) return null;
  return (
    <svg
      viewBox="0 0 1000 64"
      preserveAspectRatio="none"
      aria-hidden="true"
      style={{ position: "absolute", left: 0, top: 10, width: "100%", height: "calc(100% - 20px)", opacity }}
    >
      <path d={path} fill={color} />
    </svg>
  );
});
