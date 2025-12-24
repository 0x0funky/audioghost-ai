"use client";

import { useState } from "react";
import { X, Key, ExternalLink, Loader2, CheckCircle } from "lucide-react";
import { API } from "@/lib/api";

interface AuthModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

export default function AuthModal({ onClose, onSuccess }: AuthModalProps) {
  const [token, setToken] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [step, setStep] = useState<"input" | "success">("input");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const res = await fetch(API.auth.login(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Authentication failed");
      }

      setStep("success");
      setTimeout(() => {
        onSuccess();
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0, 0, 0, 0.8)", backdropFilter: "blur(8px)" }}
    >
      <div
        className="glass-card w-full max-w-md p-6 relative"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 rounded-lg transition-colors"
          style={{ color: "var(--text-muted)" }}
        >
          <X className="w-5 h-5" />
        </button>

        {step === "input" ? (
          <>
            {/* Header */}
            <div className="text-center mb-6">
              <div
                className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-4"
                style={{
                  background:
                    "linear-gradient(135deg, var(--ghost-primary), var(--ghost-accent))",
                }}
              >
                <Key className="w-8 h-8 text-white" />
              </div>
              <h2
                className="text-2xl font-bold mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                Connect HuggingFace
              </h2>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Enter your HuggingFace token to access SAM Audio models
              </p>
            </div>

            {/* Instructions */}
            <div
              className="rounded-xl p-4 mb-6"
              style={{ background: "var(--bg-tertiary)" }}
            >
              <h3
                className="font-medium mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                How to get your token:
              </h3>
              <ol
                className="space-y-2 text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                <li className="flex items-start gap-2">
                  <span
                    className="font-bold"
                    style={{ color: "var(--ghost-primary)" }}
                  >
                    1.
                  </span>
                  <span>
                    Request access to{" "}
                    <a
                      href="https://huggingface.co/facebook/sam-audio-large"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline hover:opacity-80"
                      style={{ color: "var(--ghost-secondary)" }}
                    >
                      SAM Audio on HuggingFace
                      <ExternalLink className="w-3 h-3 inline ml-1" />
                    </a>
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span
                    className="font-bold"
                    style={{ color: "var(--ghost-primary)" }}
                  >
                    2.
                  </span>
                  <span>
                    Create an{" "}
                    <a
                      href="https://huggingface.co/settings/tokens"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline hover:opacity-80"
                      style={{ color: "var(--ghost-secondary)" }}
                    >
                      access token
                      <ExternalLink className="w-3 h-3 inline ml-1" />
                    </a>
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span
                    className="font-bold"
                    style={{ color: "var(--ghost-primary)" }}
                  >
                    3.
                  </span>
                  <span>Paste the token below</span>
                </li>
              </ol>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <input
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
                  className="input-ghost font-mono text-sm"
                  disabled={isLoading}
                />
              </div>

              {error && (
                <div
                  className="mb-4 p-3 rounded-lg text-sm"
                  style={{
                    background: "rgba(239, 68, 68, 0.1)",
                    color: "var(--ghost-error)",
                    border: "1px solid rgba(239, 68, 68, 0.2)",
                  }}
                >
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={!token || isLoading}
                className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Verifying...
                  </>
                ) : (
                  "Connect"
                )}
              </button>
            </form>
          </>
        ) : (
          /* Success State */
          <div className="text-center py-8">
            <div
              className="w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4 animate-pulse-glow"
              style={{
                background:
                  "linear-gradient(135deg, var(--ghost-success), var(--ghost-secondary))",
              }}
            >
              <CheckCircle className="w-10 h-10 text-white" />
            </div>
            <h2
              className="text-2xl font-bold mb-2"
              style={{ color: "var(--text-primary)" }}
            >
              Connected!
            </h2>
            <p style={{ color: "var(--text-secondary)" }}>
              You can now use SAM Audio models
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
