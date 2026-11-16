"use client";

import { Ghost, Loader2, CheckCircle2, XCircle } from "lucide-react";

interface ProgressTrackerProps {
    status: "pending" | "processing";
    progress: number;
    message: string;
    queuePosition?: number | null;
    onCancel?: () => void;
}

// Progress ranges reported by the worker (backend/workers/tasks.py).
const STEPS = [
    { step: "Preparing", start: 0, end: 8 },
    { step: "Loading model (first run downloads it)", start: 8, end: 30 },
    { step: "Separating", start: 30, end: 92 },
    { step: "Saving results", start: 92, end: 100 },
];

export default function ProgressTracker({ status, progress, message, queuePosition, onCancel }: ProgressTrackerProps) {
    const title = status === "pending"
        ? queuePosition
            ? `Waiting in Queue (${queuePosition} ahead)`
            : "Waiting in Queue..."
        : "Processing Audio...";

    return (
        <section
            aria-label="Separation progress"
            style={{
                background: "var(--bg-secondary)",
                borderRadius: "20px",
                border: "1px solid var(--glass-border)",
                padding: "48px 40px"
            }}
        >
            {/* Icon */}
            <div style={{ textAlign: "center", marginBottom: "32px" }}>
                <div
                    aria-hidden
                    style={{
                        width: "80px",
                        height: "80px",
                        borderRadius: "20px",
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "linear-gradient(135deg, var(--ghost-primary), var(--ghost-accent))",
                        boxShadow: "0 8px 32px rgba(168, 85, 247, 0.3)"
                    }}
                >
                    {status === "pending" ? (
                        <Ghost style={{ width: "40px", height: "40px", color: "white" }} />
                    ) : (
                        <Loader2
                            style={{
                                width: "40px",
                                height: "40px",
                                color: "white",
                                animation: "spin 1s linear infinite"
                            }}
                        />
                    )}
                </div>
            </div>

            <h3
                style={{
                    fontSize: "1.5rem",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    textAlign: "center",
                    marginBottom: "8px"
                }}
            >
                {title}
            </h3>

            {/* Message (announced to screen readers) */}
            <p
                aria-live="polite"
                style={{
                    fontSize: "0.9rem",
                    color: "var(--text-muted)",
                    textAlign: "center",
                    marginBottom: "32px",
                    minHeight: "1.4em"
                }}
            >
                {message}
            </p>

            {/* Progress Bar */}
            <div style={{ marginBottom: "32px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px" }}>
                    <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Progress</span>
                    <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--ghost-primary)" }}>
                        {progress}%
                    </span>
                </div>
                <div
                    role="progressbar"
                    aria-label="Separation progress"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={progress}
                    aria-valuetext={`${progress}% - ${message}`}
                    style={{
                        height: "8px",
                        borderRadius: "4px",
                        background: "var(--bg-tertiary)",
                        overflow: "hidden"
                    }}
                >
                    <div
                        style={{
                            height: "100%",
                            borderRadius: "4px",
                            background: "linear-gradient(90deg, var(--ghost-primary), var(--ghost-accent))",
                            width: `${progress}%`,
                            transition: "width 0.3s ease"
                        }}
                    />
                </div>
            </div>

            {/* Steps */}
            <ol
                style={{
                    background: "var(--bg-tertiary)",
                    borderRadius: "12px",
                    padding: "20px",
                    listStyle: "none",
                    margin: 0
                }}
            >
                {STEPS.map(({ step, start, end }, index) => {
                    const isComplete = status === "processing" && progress >= end;
                    const isActive = status === "processing" && !isComplete && progress >= start;

                    return (
                        <li
                            key={step}
                            aria-current={isActive ? "step" : undefined}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "14px",
                                padding: "12px 0",
                                borderBottom: index < STEPS.length - 1 ? "1px solid var(--border-color)" : "none"
                            }}
                        >
                            <div
                                aria-hidden
                                style={{
                                    width: "24px",
                                    height: "24px",
                                    borderRadius: "50%",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    flexShrink: 0,
                                    background: isComplete
                                        ? "linear-gradient(135deg, #10B981, #34D399)"
                                        : isActive
                                            ? "var(--ghost-primary)"
                                            : "transparent",
                                    border: isComplete || isActive ? "none" : "2px solid var(--text-muted)"
                                }}
                            >
                                {isComplete ? (
                                    <CheckCircle2 style={{ width: "14px", height: "14px", color: "white" }} />
                                ) : isActive ? (
                                    <Loader2 style={{ width: "12px", height: "12px", color: "white", animation: "spin 1s linear infinite" }} />
                                ) : null}
                            </div>
                            <span
                                style={{
                                    fontSize: "0.9rem",
                                    fontWeight: isComplete || isActive ? 500 : 400,
                                    color: isComplete
                                        ? "var(--text-primary)"
                                        : isActive
                                            ? "var(--ghost-primary)"
                                            : "var(--text-muted)"
                                }}
                            >
                                {step}
                                <span className="sr-only">{isComplete ? " (done)" : isActive ? " (in progress)" : ""}</span>
                            </span>
                        </li>
                    );
                })}
            </ol>

            {onCancel && (
                <div style={{ textAlign: "center", marginTop: "24px" }}>
                    <button
                        type="button"
                        onClick={onCancel}
                        style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "8px",
                            padding: "10px 18px",
                            borderRadius: "10px",
                            border: "1px solid var(--border-color)",
                            background: "var(--bg-tertiary)",
                            color: "var(--text-secondary)",
                            cursor: "pointer",
                            fontSize: "0.85rem",
                            fontWeight: 500
                        }}
                    >
                        <XCircle aria-hidden style={{ width: "16px", height: "16px" }} />
                        Cancel
                    </button>
                </div>
            )}
        </section>
    );
}
