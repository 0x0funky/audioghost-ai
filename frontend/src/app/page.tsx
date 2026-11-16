"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Header from "@/components/Header";
import AuthModal from "@/components/AuthModal";
import AudioUploader from "@/components/AudioUploader";
import WaveformEditor from "@/components/WaveformEditor";
import SeparationPanel, { type SeparationSettings } from "@/components/SeparationPanel";
import ProgressTracker from "@/components/ProgressTracker";
import StemMixer from "@/components/StemMixer";
import VideoStemMixer from "@/components/VideoStemMixer";
import { apiUrl, loadPref, savePref } from "@/lib/api";
import type { TaskResult } from "@/lib/types";

type TaskStatus = "idle" | "pending" | "processing" | "completed" | "failed" | "cancelled";

interface TaskState {
  taskId: string | null;
  status: TaskStatus;
  progress: number;
  message: string;
  result: TaskResult | null;
  queuePosition?: number | null;
}

interface SystemStatus {
  reachable: boolean;
  redis: boolean;
  worker: boolean;
}

const IDLE_TASK: TaskState = { taskId: null, status: "idle", progress: 0, message: "", result: null };

const DEFAULT_SETTINGS: SeparationSettings = {
  modelSize: "base",
  chunkDuration: 25,
  useFloat32: false,
  quality: "balanced",
  stereo: false,
  modelVariant: "default",
};

export default function Home() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState<{ start: number; end: number } | null>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);

  // Separation settings persist across "New" and across sessions.
  // (The panel only renders after a file is chosen, so reading localStorage here can't cause a hydration mismatch.)
  const [separationSettings, setSeparationSettings] = useState<SeparationSettings>(
    () => loadPref("settings", DEFAULT_SETTINGS)
  );
  const updateSettings = (settings: SeparationSettings) => {
    setSeparationSettings(settings);
    savePref("settings", settings);
  };

  const [task, setTask] = useState<TaskState>(IDLE_TASK);
  // Polling stops as soon as the task it was started for is no longer current.
  const activeTaskRef = useRef<string | null>(null);

  const checkSystem = useCallback(() => {
    fetch(apiUrl("/api/system/status"))
      .then(res => res.json())
      .then(data => setSystem({ reachable: true, redis: !!data.redis, worker: !!data.worker }))
      .catch(() => setSystem({ reachable: false, redis: false, worker: false }));
  }, []);

  useEffect(() => {
    fetch(apiUrl("/api/auth/status"))
      .then(res => res.json())
      .then(data => setIsAuthenticated(!!data.authenticated))
      .catch(error => console.error("Failed to check auth status:", error));
    checkSystem();
  }, [checkSystem]);

  useEffect(() => {
    document.body.classList.toggle("light-mode", !isDarkMode);
  }, [isDarkMode]);

  const resetTask = () => {
    activeTaskRef.current = null;
    setTask(IDLE_TASK);
  };

  const handleFileUpload = (file: File) => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setIsVideo(file.type.startsWith("video/") || /\.(mp4|webm|mov|avi|mkv|m4v|mpeg)$/i.test(file.name));
    setSelectedRegion(null);
    resetTask();
  };

  const handleReset = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioFile(null);
    setAudioUrl(null);
    setIsVideo(false);
    setSelectedRegion(null);
    resetTask();
  };

  const pollTaskStatus = (taskId: string) => {
    let failures = 0;
    const poll = async () => {
      if (activeTaskRef.current !== taskId) return;
      try {
        const res = await fetch(apiUrl(`/api/tasks/${taskId}`));
        const data = await res.json();
        failures = 0;
        if (activeTaskRef.current !== taskId) return;
        if (data.status === "cancelled") {
          resetTask();
          return;
        }
        setTask({
          taskId,
          status: data.status,
          progress: data.progress,
          message: data.message || "",
          result: data.result || null,
          queuePosition: data.queue_position ?? null,
        });
        if (data.status !== "completed" && data.status !== "failed") {
          setTimeout(poll, 1000);
        }
      } catch (error) {
        failures += 1;
        console.error("Failed to poll task status:", error);
        if (activeTaskRef.current !== taskId) return;
        setTask(prev => ({ ...prev, message: "Lost connection to the backend - retrying..." }));
        setTimeout(poll, Math.min(10000, 1000 * 2 ** Math.min(failures, 4)));
      }
    };
    poll();
  };

  const handleSeparation = async (description: string, mode: "extract" | "remove") => {
    if (!audioFile) return;
    const s = separationSettings;

    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("description", description);
    formData.append("mode", mode);
    formData.append("model_size", s.modelSize);
    formData.append("model_variant", s.modelVariant);
    formData.append("quality", s.quality);
    formData.append("chunk_duration", s.chunkDuration.toString());
    formData.append("use_float32", s.useFloat32.toString());
    formData.append("stereo", s.stereo.toString());
    if (selectedRegion && !isVideo) {
      formData.append("start_time", selectedRegion.start.toString());
      formData.append("end_time", selectedRegion.end.toString());
    }

    setTask({ ...IDLE_TASK, status: "pending", message: "Uploading..." });
    try {
      const res = await fetch(apiUrl("/api/separate/"), { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "The server rejected the request");
      }
      activeTaskRef.current = data.task_id;
      setTask({ ...IDLE_TASK, taskId: data.task_id, status: "pending", message: "Task submitted..." });
      pollTaskStatus(data.task_id);
      checkSystem();
    } catch (error) {
      console.error("Failed to submit separation task:", error);
      const message = error instanceof Error && error.message !== "Failed to fetch"
        ? error.message
        : "Could not reach the AudioGhost backend. Is start.bat running?";
      setTask({ ...IDLE_TASK, status: "failed", message });
      checkSystem();
    }
  };

  const handleCancel = async () => {
    const taskId = task.taskId;
    resetTask();
    if (!taskId) return;
    try {
      await fetch(apiUrl(`/api/tasks/${taskId}`), { method: "DELETE" });
    } catch (error) {
      console.error("Failed to cancel task:", error);
    }
  };

  const systemWarning = (() => {
    if (!system) return null;
    if (!system.reachable) return "Can't reach the AudioGhost backend on port 8000. Start it with start.bat (or uvicorn).";
    if (!system.redis) return "Redis is not running. Start it with start.bat or `docker-compose up -d`.";
    if (!system.worker) return "The separation worker is not running yet. If this persists, check the \"AudioGhost Worker\" window.";
    return null;
  })();

  const isBusy = task.status === "pending" || task.status === "processing";

  return (
    <main
      style={{
        minHeight: "100vh",
        background: "var(--bg-primary)",
        width: "100%"
      }}
    >
      <Header
        isAuthenticated={isAuthenticated}
        onAuthClick={() => setShowAuthModal(true)}
        isDarkMode={isDarkMode}
        onThemeToggle={() => setIsDarkMode(!isDarkMode)}
        onLogoClick={handleReset}
      />

      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          padding: "32px 24px"
        }}
      >
        {systemWarning && (
          <div
            role="alert"
            style={{
              marginBottom: "24px",
              padding: "14px 18px",
              borderRadius: "12px",
              background: "rgba(245, 158, 11, 0.1)",
              border: "1px solid rgba(245, 158, 11, 0.3)",
              color: "var(--ghost-warning)",
              fontSize: "0.9rem",
              display: "flex",
              justifyContent: "space-between",
              gap: "12px",
              alignItems: "center"
            }}
          >
            <span>⚠️ {systemWarning}</span>
            <button
              onClick={checkSystem}
              style={{
                padding: "6px 12px",
                borderRadius: "8px",
                border: "1px solid rgba(245, 158, 11, 0.4)",
                background: "transparent",
                color: "var(--ghost-warning)",
                cursor: "pointer",
                fontSize: "0.8rem",
                flexShrink: 0
              }}
            >
              Check again
            </button>
          </div>
        )}

        {/* Hero Section */}
        {!audioUrl && (
          <div style={{ textAlign: "center", marginBottom: "48px" }}>
            <h1 style={{ fontSize: "3rem", fontWeight: 800, marginBottom: "16px" }}>
              <span className="gradient-text">AudioGhost</span>{" "}
              <span style={{ color: "var(--text-primary)" }}>AI</span>
            </h1>
            <p style={{ fontSize: "1.25rem", marginBottom: "8px", color: "var(--text-secondary)" }}>
              AI-Powered Object-Oriented Audio Separation
            </p>
            <p style={{ color: "var(--text-muted)" }}>
              Describe the sound you want to extract or remove using natural language
            </p>
          </div>
        )}

        {/* Main Content */}
        <div style={{ display: "grid", gap: "24px" }}>
          {/* Upload Zone */}
          {!audioUrl && (
            <AudioUploader onFileUpload={handleFileUpload} />
          )}

          {/* Waveform Editor (Audio) or Video Preview - Hide when results are shown */}
          {audioUrl && task.status !== "completed" && (
            <>
              {/* Section Header with Upload Button */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h2 style={{ fontSize: "1.25rem", fontWeight: 600, color: "var(--text-primary)" }}>
                  {isVideo ? "Video Preview" : "Audio Editor"}
                </h2>
                <button
                  onClick={handleReset}
                  disabled={isBusy}
                  style={{
                    padding: "8px 16px",
                    borderRadius: "8px",
                    background: "var(--bg-tertiary)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-color)",
                    cursor: isBusy ? "not-allowed" : "pointer",
                    opacity: isBusy ? 0.5 : 1,
                    fontSize: "0.875rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.2s ease"
                  }}
                  onMouseOver={(e) => e.currentTarget.style.background = "var(--bg-secondary)"}
                  onMouseOut={(e) => e.currentTarget.style.background = "var(--bg-tertiary)"}
                >
                  ↩ Upload New File
                </button>
              </div>

              {/* Show Video Player or Waveform based on file type */}
              {isVideo ? (
                <div
                  style={{
                    background: "var(--bg-secondary)",
                    borderRadius: "16px",
                    border: "1px solid var(--glass-border)",
                    padding: "16px",
                    overflow: "hidden"
                  }}
                >
                  <video
                    src={audioUrl}
                    controls
                    aria-label={`Preview of ${audioFile?.name ?? "uploaded video"}`}
                    style={{
                      width: "100%",
                      maxHeight: "400px",
                      borderRadius: "12px",
                      background: "#000",
                      objectFit: "contain"
                    }}
                  />
                  <p style={{
                    fontSize: "0.8rem",
                    color: "var(--text-muted)",
                    marginTop: "12px",
                    textAlign: "center"
                  }}>
                    The video&apos;s audio track will be separated; download the result as audio or as a new video
                  </p>
                </div>
              ) : (
                <WaveformEditor
                  audioUrl={audioUrl}
                  onRegionSelect={setSelectedRegion}
                  selectedRegion={selectedRegion}
                />
              )}
            </>
          )}

          {/* Separation Controls */}
          {audioUrl && task.status === "idle" && (
            <SeparationPanel
              onSeparate={handleSeparation}
              isAuthenticated={isAuthenticated}
              onAuthRequired={() => setShowAuthModal(true)}
              hasRegion={!!selectedRegion && !isVideo}
              settings={separationSettings}
              onSettingsChange={updateSettings}
            />
          )}

          {/* Progress Tracker */}
          {isBusy && (
            <ProgressTracker
              status={task.status as "pending" | "processing"}
              progress={task.progress}
              message={task.message}
              queuePosition={task.queuePosition}
              onCancel={task.taskId ? handleCancel : undefined}
            />
          )}

          {/* Results - Stem Mixer (Audio) or Video Stem Mixer */}
          {task.status === "completed" && task.result && task.taskId && (
            task.result.is_video ? (
              <VideoStemMixer
                taskId={task.taskId}
                result={task.result}
                onUploadNew={handleReset}
                onNewSeparation={resetTask}
              />
            ) : (
              <StemMixer
                taskId={task.taskId}
                result={task.result}
                onUploadNew={handleReset}
                onNewSeparation={resetTask}
              />
            )
          )}

          {/* Error State */}
          {task.status === "failed" && (
            <div className="glass-card p-6 text-center" role="alert">
              <div className="text-red-400 text-xl mb-2">❌ Separation Failed</div>
              <p style={{ color: "var(--text-secondary)", whiteSpace: "pre-wrap" }}>{task.message}</p>
              <button
                className="btn-primary mt-4"
                onClick={resetTask}
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Auth Modal */}
      {showAuthModal && (
        <AuthModal
          onClose={() => setShowAuthModal(false)}
          onSuccess={() => {
            setIsAuthenticated(true);
            setShowAuthModal(false);
          }}
        />
      )}
    </main>
  );
}
