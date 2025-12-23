"use client";

import { useState } from "react";
import {
    Mic,
    Music,
    Volume2,
    Sparkles,
    Clock,
    AlertCircle,
    Cpu,
    Search,
    Sliders
} from "lucide-react";

interface SeparationPanelProps {
    onSeparate: (description: string, mode: "extract" | "remove", modelSize: string, chunkDuration: number) => void;
    isAuthenticated: boolean;
    onAuthRequired: () => void;
    hasRegion: boolean;
}

const QUICK_PROMPTS = [
    { icon: Mic, label: "Voice", prompt: "singing voice", color: "#8B5CF6" },
    { icon: Music, label: "Music", prompt: "background music", color: "#06B6D4" },
    { icon: Volume2, label: "Drums", prompt: "drums and percussion", color: "#F472B6" },
    { icon: Sparkles, label: "Guitar", prompt: "acoustic guitar", color: "#10B981" },
    { icon: Volume2, label: "Bass", prompt: "bass", color: "#F59E0B" },
    { icon: Volume2, label: "Piano", prompt: "piano", color: "#EF4444" },
];

const MODEL_OPTIONS = [
    { value: "small", label: "Small", vram: "~6GB", speed: "Fast" },
    { value: "base", label: "Base", vram: "~7GB", speed: "Balanced" },
    { value: "large", label: "Large", vram: "~10GB", speed: "Best" },
] as const;

export default function SeparationPanel({
    onSeparate,
    isAuthenticated,
    onAuthRequired,
    hasRegion
}: SeparationPanelProps) {
    const [customPrompt, setCustomPrompt] = useState("");
    const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null);
    const [mode, setMode] = useState<"extract" | "remove">("extract");
    const [modelSize, setModelSize] = useState<"small" | "base" | "large">("base");
    const [chunkDuration, setChunkDuration] = useState(25);

    const handleQuickSelect = (prompt: string) => {
        setSelectedPrompt(prompt);
        setCustomPrompt(prompt);
    };

    const handleSeparate = () => {
        if (!isAuthenticated) {
            onAuthRequired();
            return;
        }

        const prompt = customPrompt || selectedPrompt;
        if (!prompt) return;

        onSeparate(prompt, mode, modelSize, chunkDuration);
    };

    const activePrompt = customPrompt || selectedPrompt;

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
                <h3 style={{
                    fontSize: "1rem",
                    fontWeight: 600,
                    color: "var(--text-primary)"
                }}>
                    Separation Settings
                </h3>

                {hasRegion && (
                    <div
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "6px",
                            padding: "6px 12px",
                            borderRadius: "20px",
                            fontSize: "0.75rem",
                            background: "rgba(244, 114, 182, 0.15)",
                            color: "var(--ghost-accent)"
                        }}
                    >
                        <Clock style={{ width: "12px", height: "12px" }} />
                        Temporal Lock
                    </div>
                )}
            </div>

            <div style={{ padding: "24px" }}>

                {/* ============================================ */}
                {/* MAIN INPUT - Describe the sound (PROMINENT) */}
                {/* ============================================ */}
                <div
                    style={{
                        marginBottom: "24px",
                        padding: "20px",
                        borderRadius: "14px",
                        background: "linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(244, 114, 182, 0.1))",
                        border: "1px solid rgba(168, 85, 247, 0.2)"
                    }}
                >
                    <label style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        fontSize: "0.9rem",
                        fontWeight: 600,
                        marginBottom: "12px",
                        color: "var(--text-primary)"
                    }}>
                        <Search style={{ width: "16px", height: "16px", color: "var(--ghost-primary)" }} />
                        Describe the sound you want to {mode}
                    </label>
                    <input
                        type="text"
                        value={customPrompt}
                        onChange={(e) => {
                            setCustomPrompt(e.target.value);
                            setSelectedPrompt(null);
                        }}
                        placeholder="e.g., singing voice, drums, police siren, crowd noise, a dog barking..."
                        style={{
                            width: "100%",
                            padding: "16px 18px",
                            borderRadius: "12px",
                            border: "2px solid var(--ghost-primary)",
                            background: "var(--bg-primary)",
                            color: "var(--text-primary)",
                            fontSize: "1rem",
                            fontWeight: 500,
                            outline: "none",
                            boxShadow: "0 4px 15px rgba(168, 85, 247, 0.15)"
                        }}
                    />

                    {/* Quick Select Tags */}
                    <div style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: "8px",
                        marginTop: "14px"
                    }}>
                        {QUICK_PROMPTS.map(({ icon: Icon, label, prompt, color }) => (
                            <button
                                key={prompt}
                                onClick={() => handleQuickSelect(prompt)}
                                style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "6px",
                                    padding: "8px 12px",
                                    borderRadius: "20px",
                                    border: "none",
                                    cursor: "pointer",
                                    fontSize: "0.8rem",
                                    fontWeight: 500,
                                    transition: "all 0.2s",
                                    background: selectedPrompt === prompt
                                        ? `linear-gradient(135deg, ${color}, ${color}dd)`
                                        : "var(--bg-tertiary)",
                                    color: selectedPrompt === prompt
                                        ? "white"
                                        : "var(--text-secondary)"
                                }}
                            >
                                <Icon style={{ width: "14px", height: "14px" }} />
                                {label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Settings Grid */}
                <div style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "20px",
                    marginBottom: "20px"
                }}>

                    {/* Mode Toggle */}
                    <div>
                        <label style={{
                            display: "block",
                            fontSize: "0.8rem",
                            fontWeight: 500,
                            marginBottom: "10px",
                            color: "var(--text-muted)"
                        }}>
                            Operation
                        </label>
                        <div style={{ display: "flex", gap: "8px" }}>
                            <button
                                onClick={() => setMode("extract")}
                                style={{
                                    flex: 1,
                                    padding: "12px",
                                    borderRadius: "10px",
                                    fontWeight: 500,
                                    fontSize: "0.85rem",
                                    border: "none",
                                    cursor: "pointer",
                                    background: mode === "extract"
                                        ? "linear-gradient(135deg, var(--ghost-primary), #7C3AED)"
                                        : "var(--bg-tertiary)",
                                    color: mode === "extract" ? "white" : "var(--text-secondary)"
                                }}
                            >
                                ✨ Extract
                            </button>
                            <button
                                onClick={() => setMode("remove")}
                                style={{
                                    flex: 1,
                                    padding: "12px",
                                    borderRadius: "10px",
                                    fontWeight: 500,
                                    fontSize: "0.85rem",
                                    border: "none",
                                    cursor: "pointer",
                                    background: mode === "remove"
                                        ? "linear-gradient(135deg, var(--ghost-error), #DC2626)"
                                        : "var(--bg-tertiary)",
                                    color: mode === "remove" ? "white" : "var(--text-secondary)"
                                }}
                            >
                                🗑️ Remove
                            </button>
                        </div>
                    </div>

                    {/* Model Selector */}
                    <div>
                        <label style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "6px",
                            fontSize: "0.8rem",
                            fontWeight: 500,
                            marginBottom: "10px",
                            color: "var(--text-muted)"
                        }}>
                            <Cpu style={{ width: "12px", height: "12px" }} />
                            Model
                        </label>
                        <div style={{ display: "flex", gap: "6px" }}>
                            {MODEL_OPTIONS.map(({ value, label, vram }) => (
                                <button
                                    key={value}
                                    onClick={() => setModelSize(value)}
                                    style={{
                                        flex: 1,
                                        padding: "10px 8px",
                                        borderRadius: "8px",
                                        border: "none",
                                        cursor: "pointer",
                                        textAlign: "center",
                                        background: modelSize === value
                                            ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
                                            : "var(--bg-tertiary)",
                                        color: modelSize === value ? "white" : "var(--text-secondary)"
                                    }}
                                >
                                    <div style={{ fontWeight: 500, fontSize: "0.8rem" }}>{label}</div>
                                    <div style={{ fontSize: "0.65rem", opacity: 0.7, marginTop: "2px" }}>{vram}</div>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Chunk Duration Slider */}
                <div
                    style={{
                        marginBottom: "20px",
                        padding: "16px",
                        borderRadius: "12px",
                        background: "var(--bg-tertiary)",
                        border: "1px solid var(--border-color)"
                    }}
                >
                    <div style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        marginBottom: "12px"
                    }}>
                        <label style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            fontSize: "0.8rem",
                            fontWeight: 500,
                            color: "var(--text-muted)"
                        }}>
                            <Sliders style={{ width: "14px", height: "14px" }} />
                            Chunk Duration
                        </label>
                        <span style={{
                            fontSize: "0.9rem",
                            fontWeight: 600,
                            color: "var(--ghost-primary)",
                            fontFamily: "monospace"
                        }}>
                            {chunkDuration}s
                        </span>
                    </div>

                    <input
                        type="range"
                        min="5"
                        max="60"
                        step="5"
                        value={chunkDuration}
                        onChange={(e) => setChunkDuration(Number(e.target.value))}
                        style={{
                            width: "100%",
                            height: "6px",
                            borderRadius: "3px",
                            background: `linear-gradient(to right, var(--ghost-primary) ${((chunkDuration - 5) / 55) * 100}%, var(--bg-secondary) ${((chunkDuration - 5) / 55) * 100}%)`,
                            cursor: "pointer",
                            appearance: "none",
                            outline: "none"
                        }}
                    />

                    <div style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginTop: "8px",
                        fontSize: "0.65rem",
                        color: "var(--text-muted)"
                    }}>
                        <span>5s (Low VRAM)</span>
                        <span>60s (Fast)</span>
                    </div>

                    <p style={{
                        marginTop: "10px",
                        fontSize: "0.7rem",
                        color: "var(--text-muted)",
                        lineHeight: 1.4
                    }}>
                        ⚡ Smaller chunks = Less VRAM usage, but slower processing & may affect quality at boundaries
                    </p>
                </div>

                {/* Auth Warning */}
                {!isAuthenticated && (
                    <div
                        style={{
                            marginBottom: "16px",
                            padding: "14px 16px",
                            borderRadius: "10px",
                            display: "flex",
                            alignItems: "center",
                            gap: "12px",
                            background: "rgba(245, 158, 11, 0.1)",
                            border: "1px solid rgba(245, 158, 11, 0.25)"
                        }}
                    >
                        <AlertCircle style={{
                            width: "18px",
                            height: "18px",
                            flexShrink: 0,
                            color: "var(--ghost-warning)"
                        }} />
                        <p style={{
                            fontWeight: 500,
                            fontSize: "0.85rem",
                            color: "var(--ghost-warning)"
                        }}>
                            Connect HuggingFace to continue
                        </p>
                    </div>
                )}

                {/* Action Button */}
                <button
                    onClick={handleSeparate}
                    disabled={!activePrompt}
                    style={{
                        width: "100%",
                        padding: "16px",
                        borderRadius: "12px",
                        border: "none",
                        cursor: activePrompt ? "pointer" : "not-allowed",
                        fontSize: "1rem",
                        fontWeight: 600,
                        background: activePrompt
                            ? "linear-gradient(135deg, var(--ghost-primary), var(--ghost-accent))"
                            : "var(--bg-tertiary)",
                        color: activePrompt ? "white" : "var(--text-muted)",
                        opacity: activePrompt ? 1 : 0.6,
                        boxShadow: activePrompt ? "0 4px 15px rgba(168, 85, 247, 0.3)" : "none"
                    }}
                >
                    {mode === "extract" ? "✨ Extract" : "🗑️ Remove"} "{activePrompt || "..."}"
                </button>
            </div>
        </div>
    );
}
