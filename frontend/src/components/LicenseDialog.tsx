"use client";

import { ExternalLink } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { LicenseInfo } from "@/lib/types";
import { Button, Dialog, ErrorText, Spinner } from "./ui";
import styles from "./StartScreen.module.css";

/** Meta's SAM License, accepted once before the first model download. */
export function LicenseDialog({ open, onClose, onAccepted }: { open: boolean; onClose: () => void; onAccepted: () => void }) {
  const [info, setInfo] = useState<LicenseInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open || info) return;
    let alive = true;
    api
      .license()
      .then((data) => alive && setInfo(data))
      .catch((err) => alive && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      alive = false;
    };
  }, [open, info]);

  const accept = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.acceptLicense();
      onAccepted();
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
      title="SAM-Audio license"
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={!info || busy} onClick={accept}>
            {busy ? "Saving..." : "Accept and continue"}
          </Button>
        </>
      }
    >
      <p className={styles.dialogLead}>
        AudioGhost downloads Meta&apos;s SAM-Audio models from its own public copies, so you don&apos;t need a Hugging Face
        account. The models come with Meta&apos;s SAM License: read it and accept it to continue.
      </p>
      {info ? (
        <>
          <pre className={styles.licenseText} tabIndex={0} aria-label="SAM License text">
            {info.text}
          </pre>
          <a className={styles.licenseLink} href={info.url} target="_blank" rel="noreferrer">
            The original on GitHub <ExternalLink size={12} aria-hidden="true" />
          </a>
        </>
      ) : (
        !error && <Spinner />
      )}
      {error && <ErrorText>{error}</ErrorText>}
    </Dialog>
  );
}
