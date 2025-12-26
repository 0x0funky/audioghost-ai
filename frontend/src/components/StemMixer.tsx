"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
    Play,
    Pause,
    Download,
    Volume2,
    VolumeX,
    RefreshCw,
    Ghost,
    Leaf,
    SkipBack,
    Music,
    type LucideIcon
} from "lucide-react";
import WaveSurfer from "wavesurfer.js";

interface StemMixerProps {
    taskId: string;
    description: string;
    onNewSeparation: () => void;
    onUploadNew?: () => void;
    audioDuration?: number;
    processingTime?: number;
    modelSize?: string;
}

interface Track {
    id: "ghost" | "clean";
    label: string;
    icon: LucideIcon;
    color: string;
    waveColor: string;
}

const TRACKS: Track[] = [
    {
        id: "ghost",
        label: "Isolated Sound",
        icon: Ghost,
        color: "#F472B6",
        waveColor: "#F472B6"
    },
    {
        id: "clean",
        label: "Without Isolated Sound",
        icon: Leaf,
        color: "#60A5FA",
        waveColor: "#60A5FA"
    },
];

export default function StemMixer({
    taskId,
    description,
    onNewSeparation,
    onUploadNew,
    audioDuration,
    processingTime,
    modelSize
}: StemMixerProps) {
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [muted, setMuted] = useState<Record<string, boolean>>({
        ghost: false,
        clean: false,
    });
    const [isReady, setIsReady] = useState<Record<string, boolean>>({
        ghost: false,
        clean: false,
    });

    const wavesurferRefs = useRef<Record<string, WaveSurfer | null>>({});
    const containerRefs = useRef<Record<string, HTMLDivElement | null>>({});
    const isSeeking = useRef(false);

    const getAudioUrl = (trackId: string) => {
        return `http://localhost:8000/api/tasks/${taskId}/download/${trackId}`;
    };

    useEffect(() => {
        const initWaveSurfers = async () => {
            for (const track of TRACKS) {
                const container = containerRefs.current[track.id];
                if (!container) continue;

                if (wavesurferRefs.current[track.id]) {
                    wavesurferRefs.current[track.id]?.destroy();
                }

                const ws = WaveSurfer.create({
                    container,
                    waveColor: `${track.waveColor}40`,
                    progressColor: track.waveColor,
                    cursorColor: "#ffffff",
                    cursorWidth: 1,
                    barWidth: 2,
                    barGap: 2,
                    barRadius: 2,
                    height: 48,
                    normalize: true,
                    // All tracks are seekable - clicking any will sync all others
                    interact: true,
                    hideScrollbar: true,
                });

                ws.load(getAudioUrl(track.id));

                ws.on("ready", () => {
                    setIsReady(prev => ({ ...prev, [track.id]: true }));
                    if (track.id === "ghost") {
                        setDuration(ws.getDuration());
                    }
                    ws.setMuted(muted[track.id]);
                });

                ws.on("audioprocess", () => {
                    if (!isSeeking.current && track.id === "ghost") {
                        setCurrentTime(ws.getCurrentTime());
                    }
                });

                ws.on("finish", () => {
                    // Only trigger finish logic from the "ghost" track (master)
                    // to prevent multiple finish events causing desync
                    if (track.id === "ghost") {
                        setIsPlaying(false);
                        setCurrentTime(0);
                        // Seek all tracks back to start synchronized
                        Object.values(wavesurferRefs.current).forEach(w => {
                            if (w) {
                                w.pause();
                                w.seekTo(0);
                            }
                        });
                    }
                });

                // Sync seeking across all tracks - when ANY track is seeked
                ws.on("seeking", () => {
                    if (isSeeking.current) return; // Prevent recursive seeking

                    isSeeking.current = true;
                    const progress = ws.getCurrentTime() / ws.getDuration();

                    // Sync all OTHER tracks to the same position
                    Object.entries(wavesurferRefs.current).forEach(([id, w]) => {
                        if (w && id !== track.id) {
                            w.seekTo(progress);
                        }
                    });

                    setCurrentTime(ws.getCurrentTime());

                    // Reset seeking flag after a short delay
                    setTimeout(() => {
                        isSeeking.current = false;
                    }, 50);
                });

                wavesurferRefs.current[track.id] = ws;
            }
        };

        initWaveSurfers();

        return () => {
            // Use setTimeout to avoid AbortError when component unmounts during loading
            const refs = { ...wavesurferRefs.current };
            setTimeout(() => {
                Object.values(refs).forEach(ws => {
                    if (ws) {
                        try { ws.destroy(); } catch { /* ignore AbortError */ }
                    }
                });
            }, 0);
        };
    }, [taskId]);

    // Continuous sync effect - keep all tracks aligned to original during playback
    useEffect(() => {
        let animationFrameId: number;
        const syncInterval = 100; // Sync every 100ms
        let lastSync = 0;

        const syncTracks = (timestamp: number) => {
            if (isPlaying && timestamp - lastSync > syncInterval) {
                lastSync = timestamp;
                const originalWs = wavesurferRefs.current["original"];
                if (originalWs) {
                    const masterTime = originalWs.getCurrentTime();
                    const masterDuration = originalWs.getDuration();
                    const progress = masterTime / masterDuration;

                    // Check if original finished
                    if (masterTime >= masterDuration - 0.1) {
                        // Stop all tracks
                        Object.values(wavesurferRefs.current).forEach(ws => ws?.pause());
                        Object.values(wavesurferRefs.current).forEach(ws => ws?.seekTo(0));
                        setIsPlaying(false);
                        setCurrentTime(0);
                        return;
                    }

                    // Sync other tracks to master
                    Object.entries(wavesurferRefs.current).forEach(([id, ws]) => {
                        if (ws && id !== "original") {
                            const trackTime = ws.getCurrentTime();
                            // Only sync if drift is more than 0.05 seconds
                            if (Math.abs(trackTime - masterTime) > 0.05) {
                                ws.seekTo(progress);
                            }
                        }
                    });

                    setCurrentTime(masterTime);
                }
            }
            animationFrameId = requestAnimationFrame(syncTracks);
        };

        if (isPlaying) {
            animationFrameId = requestAnimationFrame(syncTracks);
        }

        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    }, [isPlaying]);

    // Sync play/pause across all tracks
    const togglePlayAll = useCallback(() => {
        const allReady = Object.values(isReady).every(r => r);
        if (!allReady) return;

        if (isPlaying) {
            // Pause all
            Object.values(wavesurferRefs.current).forEach(ws => ws?.pause());
            setIsPlaying(false);
        } else {
            // First sync all tracks to the same position
            const originalWs = wavesurferRefs.current["original"];
            if (originalWs) {
                const progress = originalWs.getCurrentTime() / originalWs.getDuration();
                Object.entries(wavesurferRefs.current).forEach(([id, ws]) => {
                    if (ws && id !== "original") {
                        ws.seekTo(progress);
                    }
                });
            }

            // Then play all together
            Object.values(wavesurferRefs.current).forEach(ws => ws?.play());
            setIsPlaying(true);
        }
    }, [isPlaying, isReady]);

    const resetToStart = useCallback(() => {
        Object.values(wavesurferRefs.current).forEach(ws => {
            if (ws) {
                ws.pause();
                ws.seekTo(0);
            }
        });
        setIsPlaying(false);
        setCurrentTime(0);
    }, []);

    const toggleMute = useCallback((trackId: string) => {
        const ws = wavesurferRefs.current[trackId];
        if (ws) {
            const newMuted = !muted[trackId];
            ws.setMuted(newMuted);
            setMuted(prev => ({ ...prev, [trackId]: newMuted }));
        }
    }, [muted]);

    const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const progress = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));

        isSeeking.current = true;
        // Seek all tracks to the same position
        Object.values(wavesurferRefs.current).forEach(ws => {
            if (ws) ws.seekTo(progress);
        });
        setCurrentTime(progress * duration);
        isSeeking.current = false;
    }, [duration]);

    const downloadTrack = (trackId: string, label: string) => {
        const link = document.createElement("a");
        link.href = getAudioUrl(trackId);
        link.download = `${taskId}_${label.toLowerCase().replace(/\s+/g, "_")}.wav`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    const allReady = Object.values(isReady).every(r => r);

    return (
        <div
            style={{
                background: "var(--bg-secondary)",
                borderRadius: "16px",
                border: "1px solid var(--glass-border)",
                overflow: "hidden"
            }}
        >
            {/* Header */}
            <div
                style={{
                    padding: "20px 24px",
                    borderBottom: "1px solid var(--glass-border)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between"
                }}
            >
                <div>
                    <h3 style={{
                        fontSize: "1rem",
                        fontWeight: 600,
                        color: "var(--text-primary)",
                        marginBottom: "4px"
                    }}>
                        ✨ Separation Complete
                    </h3>
                    <p style={{
                        fontSize: "0.8rem",
                        color: "var(--text-muted)"
                    }}>
                        "{description}"
                    </p>
                </div>
                <div style={{ display: "flex", gap: "8px" }}>
                    {onUploadNew && (
                        <button
                            onClick={onUploadNew}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "6px",
                                padding: "8px 14px",
                                borderRadius: "8px",
                                background: "var(--bg-tertiary)",
                                color: "var(--text-secondary)",
                                border: "1px solid var(--border-color)",
                                cursor: "pointer",
                                fontSize: "0.8rem",
                                fontWeight: 500
                            }}
                        >
                            ↩ Upload New File
                        </button>
                    )}
                    <button
                        onClick={onNewSeparation}
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "6px",
                            padding: "8px 14px",
                            borderRadius: "8px",
                            background: "var(--bg-tertiary)",
                            color: "var(--text-secondary)",
                            border: "1px solid var(--border-color)",
                            cursor: "pointer",
                            fontSize: "0.8rem",
                            fontWeight: 500
                        }}
                    >
                        <RefreshCw style={{ width: "14px", height: "14px" }} />
                        New
                    </button>
                </div>
            </div>

            {/* Stats Bar */}
            {(audioDuration || processingTime || modelSize) && (
                <div
                    style={{
                        padding: "12px 24px",
                        borderBottom: "1px solid var(--glass-border)",
                        display: "flex",
                        gap: "24px",
                        background: "var(--bg-tertiary)"
                    }}
                >
                    {audioDuration !== undefined && (
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                Audio:
                            </span>
                            <span style={{
                                fontSize: "0.8rem",
                                fontWeight: 600,
                                color: "var(--text-primary)",
                                fontFamily: "monospace"
                            }}>
                                {Math.floor(audioDuration / 60)}:{(audioDuration % 60).toFixed(0).padStart(2, "0")}
                            </span>
                        </div>
                    )}
                    {processingTime !== undefined && (
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                Processing:
                            </span>
                            <span style={{
                                fontSize: "0.8rem",
                                fontWeight: 600,
                                color: "#10B981",
                                fontFamily: "monospace"
                            }}>
                                {processingTime.toFixed(1)}s
                            </span>
                        </div>
                    )}
                    {modelSize && (
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                Model:
                            </span>
                            <span style={{
                                fontSize: "0.8rem",
                                fontWeight: 600,
                                color: "var(--ghost-primary)",
                                textTransform: "capitalize"
                            }}>
                                {modelSize}
                            </span>
                        </div>
                    )}
                    {audioDuration && processingTime && (
                        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginLeft: "auto" }}>
                            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                                Speed:
                            </span>
                            <span style={{
                                fontSize: "0.8rem",
                                fontWeight: 600,
                                color: "#F59E0B",
                                fontFamily: "monospace"
                            }}>
                                {(audioDuration / processingTime).toFixed(1)}x
                            </span>
                        </div>
                    )}
                </div>
            )}

            {/* Transport Controls */}
            <div
                style={{
                    padding: "16px 24px",
                    borderBottom: "1px solid var(--glass-border)",
                    display: "flex",
                    alignItems: "center",
                    gap: "12px",
                    background: "var(--bg-tertiary)"
                }}
            >
                <button
                    onClick={resetToStart}
                    disabled={!allReady}
                    style={{
                        width: "32px",
                        height: "32px",
                        borderRadius: "6px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "var(--bg-secondary)",
                        color: "var(--text-muted)",
                        border: "none",
                        cursor: allReady ? "pointer" : "not-allowed",
                        opacity: allReady ? 1 : 0.5
                    }}
                >
                    <SkipBack style={{ width: "14px", height: "14px" }} />
                </button>

                <button
                    onClick={togglePlayAll}
                    disabled={!allReady}
                    style={{
                        width: "40px",
                        height: "40px",
                        borderRadius: "8px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: isPlaying
                            ? "linear-gradient(135deg, var(--ghost-primary), var(--ghost-accent))"
                            : "linear-gradient(135deg, #6366f1, #8b5cf6)",
                        border: "none",
                        cursor: allReady ? "pointer" : "not-allowed",
                        opacity: allReady ? 1 : 0.5,
                        boxShadow: "0 2px 8px rgba(99, 102, 241, 0.3)"
                    }}
                >
                    {isPlaying ? (
                        <Pause style={{ width: "18px", height: "18px", color: "white" }} />
                    ) : (
                        <Play style={{ width: "18px", height: "18px", color: "white", marginLeft: "2px" }} />
                    )}
                </button>

                <div style={{ flex: 1, display: "flex", alignItems: "center", gap: "12px" }}>
                    <span style={{
                        fontSize: "0.75rem",
                        fontFamily: "monospace",
                        color: "var(--text-muted)",
                        minWidth: "36px"
                    }}>
                        {formatTime(currentTime)}
                    </span>

                    <div
                        style={{
                            flex: 1,
                            height: "4px",
                            borderRadius: "2px",
                            background: "var(--bg-secondary)",
                            cursor: "pointer",
                            position: "relative"
                        }}
                        onClick={handleSeek}
                    >
                        <div
                            style={{
                                position: "absolute",
                                left: 0,
                                top: 0,
                                height: "100%",
                                borderRadius: "2px",
                                background: "linear-gradient(90deg, #6366f1, #8b5cf6)",
                                width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%`,
                                transition: "width 0.1s"
                            }}
                        />
                    </div>

                    <span style={{
                        fontSize: "0.75rem",
                        fontFamily: "monospace",
                        color: "var(--text-muted)",
                        minWidth: "36px"
                    }}>
                        {formatTime(duration)}
                    </span>
                </div>
            </div>

            {/* Tracks */}
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: "16px" }}>
                {TRACKS.map((track) => {
                    const TrackIcon = track.icon;
                    const isMuted = muted[track.id];
                    const trackReady = isReady[track.id];

                    return (
                        <div
                            key={track.id}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "12px",
                                padding: "14px 16px",
                                borderRadius: "12px",
                                background: isMuted ? "var(--bg-tertiary)" : `${track.color}08`,
                                border: `1px solid ${isMuted ? "var(--border-color)" : `${track.color}30`}`,
                                opacity: isMuted ? 0.6 : 1,
                                transition: "all 0.2s ease"
                            }}
                        >
                            {/* Mute Button */}
                            <button
                                onClick={() => toggleMute(track.id)}
                                style={{
                                    width: "32px",
                                    height: "32px",
                                    borderRadius: "8px",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    background: isMuted ? "var(--bg-secondary)" : `${track.color}20`,
                                    color: isMuted ? "var(--text-muted)" : track.color,
                                    border: "none",
                                    cursor: "pointer",
                                    flexShrink: 0
                                }}
                            >
                                {isMuted ? (
                                    <VolumeX style={{ width: "16px", height: "16px" }} />
                                ) : (
                                    <Volume2 style={{ width: "16px", height: "16px" }} />
                                )}
                            </button>

                            {/* Track Label */}
                            <div style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "10px",
                                minWidth: "180px",
                                flexShrink: 0
                            }}>
                                <TrackIcon
                                    style={{
                                        width: "16px",
                                        height: "16px",
                                        color: isMuted ? "var(--text-muted)" : track.color
                                    }}
                                />
                                <span style={{
                                    fontSize: "0.85rem",
                                    fontWeight: 500,
                                    color: isMuted ? "var(--text-muted)" : "var(--text-primary)"
                                }}>
                                    {track.label}
                                </span>
                            </div>

                            {/* Waveform */}
                            <div
                                ref={(el) => { containerRefs.current[track.id] = el; }}
                                style={{
                                    flex: 1,
                                    borderRadius: "8px",
                                    overflow: "hidden",
                                    background: "var(--bg-secondary)",
                                    minHeight: "48px"
                                }}
                            >
                                {!trackReady && (
                                    <div style={{
                                        height: "48px",
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center"
                                    }}>
                                        <span style={{
                                            fontSize: "0.75rem",
                                            color: "var(--text-muted)"
                                        }}>
                                            Loading...
                                        </span>
                                    </div>
                                )}
                            </div>

                            {/* Download */}
                            <button
                                onClick={() => downloadTrack(track.id, track.label)}
                                style={{
                                    width: "32px",
                                    height: "32px",
                                    borderRadius: "8px",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    background: "var(--bg-secondary)",
                                    color: "var(--text-muted)",
                                    border: "none",
                                    cursor: "pointer",
                                    flexShrink: 0
                                }}
                            >
                                <Download style={{ width: "16px", height: "16px" }} />
                            </button>
                        </div>
                    );
                })}
            </div>

            {/* Download All */}
            <div style={{
                padding: "20px 24px",
                borderTop: "1px solid var(--glass-border)"
            }}>
                <button
                    onClick={() => TRACKS.forEach((track) => downloadTrack(track.id, track.label))}
                    style={{
                        width: "100%",
                        padding: "14px",
                        borderRadius: "10px",
                        background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                        color: "white",
                        border: "none",
                        cursor: "pointer",
                        fontSize: "0.9rem",
                        fontWeight: 600,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: "8px",
                        boxShadow: "0 4px 12px rgba(99, 102, 241, 0.3)"
                    }}
                >
                    <Download style={{ width: "18px", height: "18px" }} />
                    Download All Stems
                </button>
            </div>
        </div>
    );
}
