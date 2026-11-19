"use client";

import { ExternalLink } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";
import { Button, Dialog, ErrorText } from "./ui";
import styles from "./StartScreen.module.css";

export function HfDialog({ open, onClose, onConnected }: { open: boolean; onClose: () => void; onConnected: () => void }) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.login(token.trim());
      setToken("");
      onConnected();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Connect Hugging Face"
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={!token.trim() || busy} onClick={submit}>
            {busy ? "Checking..." : "Connect"}
          </Button>
        </>
      }
    >
      <ol className={styles.steps}>
        <li>
          Only needed for Meta&apos;s precision (-tv) checkpoints; the standard models need no account. Request
          access to Meta&apos;s SAM-Audio models (approval can take a while):{" "}
          <a href="https://huggingface.co/facebook/sam-audio-large" target="_blank" rel="noreferrer">
            facebook/sam-audio-large <ExternalLink size={12} aria-hidden="true" />
          </a>
        </li>
        <li>
          Create a token with <strong>Read</strong> access:{" "}
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
            huggingface.co/settings/tokens <ExternalLink size={12} aria-hidden="true" />
          </a>
        </li>
        <li>Paste it below. It stays on this computer (backend/.hf_token).</li>
      </ol>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (token.trim() && !busy) void submit();
        }}
      >
        <label htmlFor="hf-token" className={styles.fieldLabel}>
          Access token
        </label>
        <input
          id="hf-token"
          className={styles.textInput}
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder="hf_..."
          value={token}
          onChange={(event) => setToken(event.target.value)}
        />
      </form>
      {error && <ErrorText>{error}</ErrorText>}
    </Dialog>
  );
}
