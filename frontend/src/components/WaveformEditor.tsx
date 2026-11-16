"use client";

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js";
import { Play, Pause, RotateCcw, Scissors, X } from "lucide-react";

interface WaveformEditorProps {
    audioUrl: string;
    onRegionSelect: (region: { start: number; end: number } | null) => void;
    selectedRegion: { start: number; end: number } | null;
}

export default function WaveformEditor({
    audioUrl,
    onRegionSelect,
    selectedRegion
}: WaveformEditorProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const wavesurferRef = useRef<WaveSurfer | null>(null);
    const regionsRef = useRef<RegionsPlugin | null>(null);

    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        if (!containerRef.current) return;

        let isMounted = true;

        // Create regions plugin
        const regions = RegionsPlugin.create();
        regionsRef.current = regions;

        // Create wavesurfer instance
        const wavesurfer = WaveSurfer.create({
            container: containerRef.current,
            waveColor: "rgba(139, 92, 246, 0.5)",
            progressColor: "#8B5CF6",
            cursorColor: "#F472B6",
            cursorWidth: 2,
            barWidth: 3,
            barGap: 2,
            barRadius: 3,
            height: 128,
            normalize: true,
            plugins: [regions],
        });

        wavesurferRef.current = wavesurfer;

        // Event listeners
        wavesurfer.on("ready", () => {
            if (isMounted) {
                setDuration(wavesurfer.getDuration());
                setIsLoaded(true);
            }
        });

        wavesurfer.on("audioprocess", () => {
            if (isMounted) {
                setCurrentTime(wavesurfer.getCurrentTime());
            }
        });

        wavesurfer.on("seeking", () => {
            if (isMounted) {
                setCurrentTime(wavesurfer.getCurrentTime());
            }
        });

        wavesurfer.on("play", () => isMounted && setIsPlaying(true));
        wavesurfer.on("pause", () => isMounted && setIsPlaying(false));

        // Region events
        regions.on("region-created", (region) => {
            // Only allow one region at a time
            regions.getRegions().forEach((r) => {
                if (r.id !== region.id) {
                    r.remove();
                }
            });

            if (isMounted) {
                onRegionSelect({
                    start: region.start,
                    end: region.end,
                });
            }
        });

        regions.on("region-updated", (region) => {
            if (isMounted) {
                onRegionSelect({
                    start: region.start,
                    end: region.end,
                });
            }
        });

        // Catch loading errors silently (including AbortError when component unmounts)
        wavesurfer.on("error", (error) => {
            // Silently ignore AbortError - this happens when unmounting during load
            if (error?.name === "AbortError" || String(error).includes("abort")) {
                return;
            }
            console.warn("WaveSurfer error:", error);
        });

        // Load audio
        wavesurfer.load(audioUrl).catch((error) => {
            // Silently ignore AbortError
            if (error?.name === "AbortError" || String(error).includes("abort")) {
                return;
            }
            console.warn("Failed to load audio:", error);
        });

        return () => {
            isMounted = false;
            // Use setTimeout to ensure any pending operations complete
            // before attempting destruction
            const ws = wavesurfer;
            setTimeout(() => {
                try {
                    ws.destroy();
                } catch {
                    // Ignore AbortError and other destruction errors
                    // This happens when component unmounts during audio loading
                }
            }, 0);
        };
    }, [audioUrl]);


    const togglePlayPause = () => {
        wavesurferRef.current?.playPause();
    };

    const restart = () => {
        wavesurferRef.current?.seekTo(0);
        wavesurferRef.current?.play();
    };

    const createRegion = () => {
        if (!regionsRef.current || !wavesurferRef.current) return;

        const currentPos = wavesurferRef.current.getCurrentTime();
        const dur = wavesurferRef.current.getDuration();

        // Create a region from current position to +5 seconds
        const start = currentPos;
        const end = Math.min(currentPos + 5, dur);

        regionsRef.current.addRegion({
            start,
            end,
            color: "rgba(244, 114, 182, 0.3)",
            drag: true,
            resize: true,
        });
    };

    const clearRegion = () => {
        regionsRef.current?.getRegions().forEach((r) => r.remove());
        onRegionSelect(null);
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    return (
        <div className="waveform-container">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold" style={{ color: "var(--text-primary)" }}>
                    Waveform Editor
                </h3>
                <div className="flex items-center gap-2">
                    {selectedRegion && (
                        <span
                            className="text-sm px-3 py-1 rounded-full"
                            style={{
                                background: "rgba(244, 114, 182, 0.2)",
                                color: "var(--ghost-accent)"
                            }}
                        >
                            Selected: {formatTime(selectedRegion.start)} - {formatTime(selectedRegion.end)}
                        </span>
                    )}
                </div>
            </div>

            {/* Waveform */}
            <div
                ref={containerRef}
                className="rounded-xl overflow-hidden mb-4"
                style={{ background: "var(--bg-primary)" }}
            />

            {/* Loading skeleton */}
            {!isLoaded && (
                <div className="h-32 rounded-xl shimmer mb-4" />
            )}

            {/* Controls */}
            <div className="flex items-center justify-between">
                {/* Playback Controls */}
                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        aria-label={isPlaying ? "Pause" : "Play"}
                        onClick={togglePlayPause}
                        className="w-12 h-12 rounded-xl flex items-center justify-center transition-all hover:scale-105"
                        style={{
                            background: "linear-gradient(135deg, var(--ghost-primary), #7C3AED)",
                        }}
                    >
                        {isPlaying ? (
                            <Pause className="w-6 h-6 text-white" />
                        ) : (
                            <Play className="w-6 h-6 text-white ml-1" />
                        )}
                    </button>

                    <button
                        type="button"
                        aria-label="Restart from the beginning"
                        onClick={restart}
                        className="w-10 h-10 rounded-lg flex items-center justify-center transition-all hover:scale-105"
                        style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)" }}
                    >
                        <RotateCcw className="w-5 h-5" />
                    </button>

                    <span className="ml-4 font-mono text-sm" style={{ color: "var(--text-secondary)" }}>
                        {formatTime(currentTime)} / {formatTime(duration)}
                    </span>
                </div>

                {/* Region Controls */}
                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        aria-label="Select a 5 second region at the playhead for Temporal Lock"
                        onClick={createRegion}
                        className="btn-secondary flex items-center gap-2 text-sm"
                    >
                        <Scissors className="w-4 h-4" />
                        Select Region
                    </button>

                    {selectedRegion && (
                        <button
                            type="button"
                            aria-label="Clear the selected region"
                            onClick={clearRegion}
                            className="p-2 rounded-lg transition-all"
                            style={{ background: "var(--bg-tertiary)", color: "var(--ghost-error)" }}
                        >
                            <X className="w-5 h-5" />
                        </button>
                    )}
                </div>
            </div>

            {/* Instructions */}
            <p className="mt-4 text-sm" style={{ color: "var(--text-muted)" }}>
                💡 Tip: Mark a region where the sound is clearly audible to turn on <strong>Temporal Lock</strong> - the model uses it as a hint to find the right sound. The whole file is still processed.
            </p>
        </div>
    );
}
