"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, Music } from "lucide-react";

interface AudioUploaderProps {
    onFileUpload: (file: File) => void;
}

const AUDIO_EXT = /\.(mp3|wav|flac|ogg|opus|m4a|aac|aiff?|wma)$/i;
const VIDEO_EXT = /\.(mp4|m4v|webm|mov|avi|mkv|mpeg)$/i;

function isMediaFile(file: File) {
    return (
        file.type.startsWith("audio/") ||
        file.type.startsWith("video/") ||
        AUDIO_EXT.test(file.name) ||
        VIDEO_EXT.test(file.name)
    );
}

export default function AudioUploader({ onFileUpload }: AudioUploaderProps) {
    const [isDragOver, setIsDragOver] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const accept = useCallback((file: File | undefined) => {
        if (!file) return;
        if (!isMediaFile(file)) {
            setError(`"${file.name}" doesn't look like an audio or video file.`);
            return;
        }
        setError(null);
        onFileUpload(file);
    }, [onFileUpload]);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
        accept(e.dataTransfer.files[0]);
    }, [accept]);

    const openPicker = () => fileInputRef.current?.click();

    return (
        <div
            className={`upload-zone ${isDragOver ? "dragover" : ""}`}
            role="button"
            tabIndex={0}
            aria-label="Upload an audio or video file. Press Enter to browse."
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={openPicker}
            onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    openPicker();
                }
            }}
        >
            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,video/*"
                onChange={(e) => {
                    accept(e.target.files?.[0]);
                    e.target.value = "";
                }}
                className="hidden"
                tabIndex={-1}
                aria-hidden
            />

            <div className="flex flex-col items-center">
                {/* Icon */}
                <div
                    aria-hidden
                    className="w-20 h-20 rounded-2xl flex items-center justify-center mb-6 animate-float"
                    style={{
                        background: isDragOver
                            ? "linear-gradient(135deg, var(--ghost-primary), var(--ghost-accent))"
                            : "var(--bg-tertiary)"
                    }}
                >
                    {isDragOver ? (
                        <Music className="w-10 h-10 text-white" />
                    ) : (
                        <Upload className="w-10 h-10" style={{ color: "var(--ghost-primary)" }} />
                    )}
                </div>

                {/* Text */}
                <h3 className="text-xl font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
                    {isDragOver ? "Drop your file here" : "Upload Audio or Video"}
                </h3>
                <p className="mb-4" style={{ color: "var(--text-secondary)" }}>
                    Drag & drop or click to browse
                </p>

                {/* Supported formats */}
                <div className="flex items-center gap-2 flex-wrap justify-center" aria-label="Supported formats">
                    {["MP3", "WAV", "FLAC", "M4A", "MP4", "WebM", "MOV", "MKV"].map((format) => (
                        <span
                            key={format}
                            className="px-3 py-1 rounded-full text-xs font-medium"
                            style={{
                                background: "var(--bg-tertiary)",
                                color: "var(--text-muted)"
                            }}
                        >
                            {format}
                        </span>
                    ))}
                </div>

                {error && (
                    <p role="alert" className="mt-4 text-sm" style={{ color: "var(--ghost-error)" }}>
                        {error}
                    </p>
                )}
            </div>
        </div>
    );
}
