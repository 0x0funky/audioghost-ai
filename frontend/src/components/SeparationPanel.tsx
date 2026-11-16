"use client";

import { useId, useState } from "react";
import {
    Mic,
    Music,
    Volume2,
    Sparkles,
    Clock,
    AlertCircle,
    Cpu,
    Search,
    Sliders,
    Zap,
    Gauge,
    Headphones,
    Crosshair,
} from "lucide-react";

export interface SeparationSettings {
    modelSize: "small" | "base" | "large";
    chunkDuration: number;
    useFloat32: boolean;
    quality: "fast" | "balanced" | "high";
    stereo: boolean;
    modelVariant: "default" | "tv";
}

interface SeparationPanelProps {
    onSeparate: (description: string, mode: "extract" | "remove") => void;
    isAuthenticated: boolean;
    onAuthRequired: () => void;
    hasRegion: boolean;
    settings: SeparationSettings;
    onSettingsChange: (settings: SeparationSettings) => void;
}

const QUICK_PROMPTS = [
    { icon: Mic, label: "Voice", prompt: "singing voice", color: "#8B5CF6" },
    { icon: Mic, label: "Speech", prompt: "a person speaking", color: "#A855F7" },
    { icon: Music, label: "Music", prompt: "background music", color: "#06B6D4" },
    { icon: Volume2, label: "Drums", prompt: "drums and percussion", color: "#F472B6" },
    { icon: Sparkles, label: "Guitar", prompt: "acoustic guitar", color: "#10B981" },
    { icon: Volume2, label: "Bass", prompt: "bass", color: "#F59E0B" },
    { icon: Volume2, label: "Piano", prompt: "piano", color: "#EF4444" },
    { icon: Volume2, label: "Noise", prompt: "background noise", color: "#64748B" },
];

// Peak VRAM in low-memory mode (one 25 s chunk at a time), measured on an RTX 4090.
// fp32 adds the transformer weights a second time. Bigger GPUs batch chunks for speed.
const MODEL_OPTIONS = [
    { value: "small", label: "Small", vram: "~4GB", vramFp32: "~5GB" },
    { value: "base", label: "Base", vram: "~5.5GB", vramFp32: "~8GB" },
    { value: "large", label: "Large", vram: "~9GB", vramFp32: "~15GB" },
] as const;

const QUALITY_OPTIONS = [
    { value: "fast", label: "Fast", hint: "8 steps · ~2× faster" },
    { value: "balanced", label: "Balanced", hint: "16 steps · Meta default" },
    { value: "high", label: "High", hint: "32 steps · ~2× slower" },
] as const;

function Toggle({
    checked,
    onChange,
    label,
    description,
    icon: Icon,
    accent,
}: {
    checked: boolean;
    onChange: (value: boolean) => void;
    label: string;
    description: string;
    icon: typeof Zap;
    accent: string;
}) {
    const id = useId();
    return (
        <div
            style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "12px",
                padding: "14px 16px",
                borderRadius: "10px",
                background: checked ? `${accent}1f` : "var(--bg-tertiary)",
                border: checked ? `1px solid ${accent}55` : "1px solid var(--border-color)",
            }}
        >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <Icon aria-hidden style={{ width: "16px", height: "16px", flexShrink: 0, color: checked ? accent : "var(--text-muted)" }} />
                <div>
                    <div id={`${id}-label`} style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-primary)" }}>
                        {label}
                    </div>
                    <div id={`${id}-desc`} style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "2px" }}>
                        {description}
                    </div>
                </div>
            </div>
            <button
                type="button"
                role="switch"
                aria-checked={checked}
                aria-labelledby={`${id}-label`}
                aria-describedby={`${id}-desc`}
                onClick={() => onChange(!checked)}
                style={{
                    width: "44px",
                    height: "24px",
                    borderRadius: "12px",
                    border: "none",
                    cursor: "pointer",
                    position: "relative",
                    flexShrink: 0,
                    background: checked ? accent : "var(--bg-secondary)",
                    transition: "all 0.2s ease",
                }}
            >
                <div style={{
                    width: "18px",
                    height: "18px",
                    borderRadius: "50%",
                    background: "white",
                    position: "absolute",
                    top: "3px",
                    left: checked ? "23px" : "3px",
                    transition: "left 0.2s ease",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
                }} />
            </button>
        </div>
    );
}

const sectionLabel = {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    fontSize: "0.8rem",
    fontWeight: 500,
    marginBottom: "10px",
    color: "var(--text-muted)",
} as const;

export default function SeparationPanel({
    onSeparate,
    isAuthenticated,
    onAuthRequired,
    hasRegion,
    settings,
    onSettingsChange
}: SeparationPanelProps) {
    const promptId = useId();
    const chunkId = useId();
    // Only prompt and mode are local (reset each time)
    const [customPrompt, setCustomPrompt] = useState("");
    const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null);
    const [mode, setMode] = useState<"extract" | "remove">("extract");

    const { modelSize, chunkDuration, useFloat32, quality, stereo, modelVariant } = settings;

    const updateSetting = <K extends keyof SeparationSettings>(key: K, value: SeparationSettings[K]) => {
        onSettingsChange({ ...settings, [key]: value });
    };

    const handleQuickSelect = (prompt: string) => {
        setSelectedPrompt(prompt);
        setCustomPrompt(prompt);
    };

    const activePrompt = (customPrompt || selectedPrompt || "").trim();

    const handleSeparate = () => {
        if (!isAuthenticated) {
            onAuthRequired();
            return;
        }
        if (!activePrompt) return;
        onSeparate(activePrompt, mode);
    };

    return (
        <section
            aria-label="Separation settings"
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
                <h3 style={{ fontSize: "1rem", fontWeight: 600, color: "var(--text-primary)" }}>
                    Separation Settings
                </h3>

                {hasRegion && (
                    <div
                        title="The selected region tells the model where the sound is audible"
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
                        <Clock aria-hidden style={{ width: "12px", height: "12px" }} />
                        Temporal Lock on
                    </div>
                )}
            </div>

            <div style={{ padding: "24px" }}>
                {/* MAIN INPUT - Describe the sound */}
                <div
                    style={{
                        marginBottom: "24px",
                        padding: "20px",
                        borderRadius: "14px",
                        background: "linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(244, 114, 182, 0.1))",
                        border: "1px solid rgba(168, 85, 247, 0.2)"
                    }}
                >
                    <label htmlFor={promptId} style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        fontSize: "0.9rem",
                        fontWeight: 600,
                        marginBottom: "12px",
                        color: "var(--text-primary)"
                    }}>
                        <Search aria-hidden style={{ width: "16px", height: "16px", color: "var(--ghost-primary)" }} />
                        Describe the sound you want to {mode}
                    </label>
                    <input
                        id={promptId}
                        type="text"
                        value={customPrompt}
                        maxLength={300}
                        onChange={(e) => {
                            setCustomPrompt(e.target.value);
                            setSelectedPrompt(null);
                        }}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") handleSeparate();
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
                    <div role="group" aria-label="Quick prompts" style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: "8px",
                        marginTop: "14px"
                    }}>
                        {QUICK_PROMPTS.map(({ icon: Icon, label, prompt, color }) => (
                            <button
                                key={prompt}
                                type="button"
                                aria-pressed={selectedPrompt === prompt}
                                aria-label={`Use prompt: ${prompt}`}
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
                                    color: selectedPrompt === prompt ? "white" : "var(--text-secondary)"
                                }}
                            >
                                <Icon aria-hidden style={{ width: "14px", height: "14px" }} />
                                {label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Settings Grid */}
                <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                    gap: "20px",
                    marginBottom: "20px"
                }}>
                    {/* Mode Toggle */}
                    <div role="group" aria-labelledby={`${promptId}-op`}>
                        <div id={`${promptId}-op`} style={sectionLabel}>Operation</div>
                        <div style={{ display: "flex", gap: "8px" }}>
                            <button
                                type="button"
                                aria-pressed={mode === "extract"}
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
                                type="button"
                                aria-pressed={mode === "remove"}
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
                    <div role="group" aria-labelledby={`${promptId}-model`}>
                        <div id={`${promptId}-model`} style={sectionLabel}>
                            <Cpu aria-hidden style={{ width: "12px", height: "12px" }} />
                            Model
                        </div>
                        <div style={{ display: "flex", gap: "6px" }}>
                            {MODEL_OPTIONS.map(({ value, label, vram, vramFp32 }) => {
                                const selected = modelSize === value;
                                const memory = useFloat32 ? vramFp32 : vram;
                                return (
                                    <button
                                        key={value}
                                        type="button"
                                        aria-pressed={selected}
                                        aria-label={`${label} model, needs about ${memory.replace("~", "")} of GPU memory${selected ? ", selected" : ""}`}
                                        onClick={() => updateSetting("modelSize", value)}
                                        style={{
                                            flex: 1,
                                            padding: "10px 8px",
                                            borderRadius: "8px",
                                            border: "none",
                                            cursor: "pointer",
                                            textAlign: "center",
                                            background: selected
                                                ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
                                                : "var(--bg-tertiary)",
                                            color: selected ? "white" : "var(--text-secondary)"
                                        }}
                                    >
                                        <div style={{ fontWeight: 500, fontSize: "0.8rem" }}>{label}</div>
                                        <div style={{ fontSize: "0.65rem", opacity: 0.7, marginTop: "2px" }}>
                                            {memory}
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {/* Quality preset */}
                <div role="group" aria-labelledby={`${promptId}-quality`} style={{ marginBottom: "20px" }}>
                    <div id={`${promptId}-quality`} style={sectionLabel}>
                        <Gauge aria-hidden style={{ width: "12px", height: "12px" }} />
                        Speed / Quality
                    </div>
                    <div style={{ display: "flex", gap: "6px" }}>
                        {QUALITY_OPTIONS.map(({ value, label, hint }) => {
                            const selected = quality === value;
                            return (
                                <button
                                    key={value}
                                    type="button"
                                    aria-pressed={selected}
                                    aria-label={`${label}: ${hint}${selected ? ", selected" : ""}`}
                                    onClick={() => updateSetting("quality", value)}
                                    style={{
                                        flex: 1,
                                        padding: "10px 8px",
                                        borderRadius: "8px",
                                        border: "none",
                                        cursor: "pointer",
                                        textAlign: "center",
                                        background: selected
                                            ? "linear-gradient(135deg, #0ea5e9, #6366f1)"
                                            : "var(--bg-tertiary)",
                                        color: selected ? "white" : "var(--text-secondary)"
                                    }}
                                >
                                    <div style={{ fontWeight: 500, fontSize: "0.8rem" }}>{label}</div>
                                    <div style={{ fontSize: "0.65rem", opacity: 0.75, marginTop: "2px" }}>{hint}</div>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Toggles */}
                <div style={{ display: "grid", gap: "10px", marginBottom: "20px" }}>
                    <Toggle
                        checked={modelVariant === "tv"}
                        onChange={(v) => updateSetting("modelVariant", v ? "tv" : "default")}
                        label="Precision model (-tv)"
                        description="Meta's -tv checkpoints pick the right sound more reliably. Separate download."
                        icon={Crosshair}
                        accent="#F472B6"
                    />
                    <Toggle
                        checked={stereo}
                        onChange={(v) => updateSetting("stereo", v)}
                        label="Keep stereo"
                        description="Separates left and right channels (about 2× processing). Mono files stay mono."
                        icon={Headphones}
                        accent="#06B6D4"
                    />
                    <Toggle
                        checked={useFloat32}
                        onChange={(v) => updateSetting("useFloat32", v)}
                        label="High Quality Mode (float32)"
                        description="Full fp32 transformer. More VRAM and slower; the default already keeps the sensitive math in fp32."
                        icon={Zap}
                        accent="#10B981"
                    />
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
                        <label htmlFor={chunkId} style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            fontSize: "0.8rem",
                            fontWeight: 500,
                            color: "var(--text-muted)"
                        }}>
                            <Sliders aria-hidden style={{ width: "14px", height: "14px" }} />
                            Max chunk length
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
                        id={chunkId}
                        type="range"
                        min="5"
                        max="60"
                        step="5"
                        value={chunkDuration}
                        aria-valuetext={`${chunkDuration} seconds`}
                        onChange={(e) => updateSetting("chunkDuration", Number(e.target.value))}
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
                        <span>60s (Fewer chunks)</span>
                    </div>

                    <p style={{
                        marginTop: "10px",
                        fontSize: "0.7rem",
                        color: "var(--text-muted)",
                        lineHeight: 1.4
                    }}>
                        Long files are split into equal chunks that overlap by 2 s and are crossfaded, so there are no clicks at the seams. Shorter chunks use less VRAM.
                    </p>
                </div>

                {/* Auth Warning */}
                {!isAuthenticated && (
                    <div
                        role="status"
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
                        <AlertCircle aria-hidden style={{
                            width: "18px",
                            height: "18px",
                            flexShrink: 0,
                            color: "var(--ghost-warning)"
                        }} />
                        <p style={{ fontWeight: 500, fontSize: "0.85rem", color: "var(--ghost-warning)" }}>
                            Connect HuggingFace to continue
                        </p>
                    </div>
                )}

                {/* Action Button */}
                <button
                    type="button"
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
                    {mode === "extract" ? "✨ Extract" : "🗑️ Remove"} &quot;{activePrompt || "..."}&quot;
                </button>
            </div>
        </section>
    );
}
