"use client";

import { useState, useEffect } from "react";
import Header from "@/components/Header";
import AuthModal from "@/components/AuthModal";
import AudioUploader from "@/components/AudioUploader";
import WaveformEditor from "@/components/WaveformEditor";
import SeparationPanel from "@/components/SeparationPanel";
import ProgressTracker from "@/components/ProgressTracker";
import StemMixer from "@/components/StemMixer";

interface TaskResult {
  original_path: string;
  ghost_path: string;
  clean_path: string;
  description: string;
  mode: string;
  audio_duration?: number;
  processing_time?: number;
  model_size?: string;
}

interface TaskState {
  taskId: string | null;
  status: "idle" | "pending" | "processing" | "completed" | "failed";
  progress: number;
  message: string;
  result: TaskResult | null;
}

export default function Home() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState<{ start: number; end: number } | null>(null);

  const [task, setTask] = useState<TaskState>({
    taskId: null,
    status: "idle",
    progress: 0,
    message: "",
    result: null,
  });

  // Check auth status on mount
  useEffect(() => {
    checkAuthStatus();
  }, []);

  // Toggle theme
  useEffect(() => {
    document.body.classList.toggle("light-mode", !isDarkMode);
  }, [isDarkMode]);

  const checkAuthStatus = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/auth/status");
      const data = await res.json();
      setIsAuthenticated(data.authenticated);
    } catch (error) {
      console.error("Failed to check auth status:", error);
    }
  };

  const handleFileUpload = (file: File) => {
    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    // Reset task state
    setTask({
      taskId: null,
      status: "idle",
      progress: 0,
      message: "",
      result: null,
    });
  };

  const handleReset = () => {
    // Clean up the object URL to free memory
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioFile(null);
    setAudioUrl(null);
    setSelectedRegion(null);
    setTask({
      taskId: null,
      status: "idle",
      progress: 0,
      message: "",
      result: null,
    });
  };

  const handleSeparation = async (
    description: string,
    mode: "extract" | "remove",
    modelSize: string = "base",
    chunkDuration: number = 25
  ) => {
    if (!audioFile) return;

    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("description", description);
    formData.append("mode", mode);
    formData.append("model_size", modelSize);
    formData.append("chunk_duration", chunkDuration.toString());

    if (selectedRegion) {
      formData.append("start_time", selectedRegion.start.toString());
      formData.append("end_time", selectedRegion.end.toString());
    }

    try {
      const res = await fetch("http://localhost:8000/api/separate/", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setTask({
        taskId: data.task_id,
        status: "pending",
        progress: 0,
        message: "Task submitted...",
        result: null,
      });

      // Start polling for status
      pollTaskStatus(data.task_id);
    } catch (error) {
      console.error("Failed to submit separation task:", error);
      setTask(prev => ({
        ...prev,
        status: "failed",
        message: "Failed to submit task",
      }));
    }
  };

  const pollTaskStatus = async (taskId: string) => {
    const poll = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/tasks/${taskId}`);
        const data = await res.json();

        setTask({
          taskId,
          status: data.status,
          progress: data.progress,
          message: data.message || "",
          result: data.result || null,
        });

        if (data.status !== "completed" && data.status !== "failed") {
          setTimeout(poll, 1000);
        }
      } catch (error) {
        console.error("Failed to poll task status:", error);
      }
    };

    poll();
  };

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


          {/* Waveform Editor with Reset Button */}
          {audioUrl && (
            <>
              {/* Reset Button */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h2 style={{ fontSize: "1.25rem", fontWeight: 600, color: "var(--text-primary)" }}>
                  Audio Editor
                </h2>
                <button
                  onClick={handleReset}
                  style={{
                    padding: "8px 16px",
                    borderRadius: "8px",
                    background: "var(--bg-tertiary)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-color)",
                    cursor: "pointer",
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
              <WaveformEditor
                audioUrl={audioUrl}
                onRegionSelect={setSelectedRegion}
                selectedRegion={selectedRegion}
              />
            </>
          )}


          {/* Separation Controls */}
          {audioUrl && task.status === "idle" && (
            <SeparationPanel
              onSeparate={handleSeparation}
              isAuthenticated={isAuthenticated}
              onAuthRequired={() => setShowAuthModal(true)}
              hasRegion={!!selectedRegion}
            />
          )}

          {/* Progress Tracker */}
          {(task.status === "pending" || task.status === "processing") && (
            <ProgressTracker
              status={task.status}
              progress={task.progress}
              message={task.message}
            />
          )}

          {/* Results - Stem Mixer */}
          {task.status === "completed" && task.result && task.taskId && (
            <StemMixer
              taskId={task.taskId}
              description={task.result.description}
              audioDuration={task.result.audio_duration}
              processingTime={task.result.processing_time}
              modelSize={task.result.model_size}
              onNewSeparation={() => {
                setTask({
                  taskId: null,
                  status: "idle",
                  progress: 0,
                  message: "",
                  result: null,
                });
              }}
            />
          )}

          {/* Error State */}
          {task.status === "failed" && (
            <div className="glass-card p-6 text-center">
              <div className="text-red-400 text-xl mb-2">❌ Separation Failed</div>
              <p style={{ color: "var(--text-secondary)" }}>{task.message}</p>
              <button
                className="btn-primary mt-4"
                onClick={() => setTask({ taskId: null, status: "idle", progress: 0, message: "", result: null })}
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
