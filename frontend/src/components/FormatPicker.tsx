"use client";

import { DOWNLOAD_FORMATS, type DownloadFormat } from "@/lib/api";

interface FormatPickerProps {
    value: DownloadFormat;
    onChange: (format: DownloadFormat) => void;
}

/** Radio group for the download format (WAV / FLAC / MP3). */
export default function FormatPicker({ value, onChange }: FormatPickerProps) {
    return (
        <div
            role="radiogroup"
            aria-label="Download format"
            style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}
        >
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginRight: "4px" }}>
                Format:
            </span>
            {DOWNLOAD_FORMATS.map(({ value: option, label, hint }) => {
                const selected = option === value;
                return (
                    <button
                        key={option}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        aria-label={`${label}, ${hint}`}
                        title={hint}
                        onClick={() => onChange(option)}
                        style={{
                            padding: "6px 12px",
                            borderRadius: "8px",
                            border: selected ? "1px solid transparent" : "1px solid var(--border-color)",
                            background: selected ? "linear-gradient(135deg, #6366f1, #8b5cf6)" : "var(--bg-tertiary)",
                            color: selected ? "white" : "var(--text-secondary)",
                            cursor: "pointer",
                            fontSize: "0.8rem",
                            fontWeight: 600
                        }}
                    >
                        {label}
                    </button>
                );
            })}
        </div>
    );
}
