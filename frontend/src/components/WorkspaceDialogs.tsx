"use client";

import { ArrowDown, ArrowUp } from "lucide-react";
import { useState } from "react";
import type { SeparationSettings, Sound } from "@/lib/types";
import { Button, Dialog, ErrorText, IconButton, Kbd } from "./ui";
import styles from "./Workspace.module.css";

const SHORTCUTS: { title: string; rows: [string[], string][] }[] = [
  {
    title: "Playback",
    rows: [
      [["Space"], "Play or pause (plays the selection when there is one)"],
      [["Left", "Right"], "Move the playhead 1 second (5 with Shift)"],
      [["Home", "End"], "Go to the start or the end"],
      [["L"], "Loop the file, or the selection"],
    ],
  },
  {
    title: "Selecting",
    rows: [
      [["Drag"], "On a track: select a part. On the ruler: move the playhead"],
      [["Shift", "Click"], "Select from the playhead, or stretch the selection, to that point"],
      [["I", "O"], "Start and end the selection at the playhead"],
      [["Esc"], "Let go of the selection, or stop marking hints"],
    ],
  },
  {
    title: "Tracks",
    rows: [
      [["Up", "Down"], "Choose a track"],
      [["M"], "Remove the chosen track from what you hear, or put it back"],
      [["S"], "Hear only the chosen track"],
      [["Ctrl", "Z"], "Undo (Ctrl + Shift + Z redoes)"],
    ],
  },
  {
    title: "View",
    rows: [
      [["+", "-"], "Zoom in and out (also Ctrl + mouse wheel)"],
      [["0"], "Show the whole file"],
      [["Ctrl", "Enter"], "In the sound box: separate"],
      [["?"], "This list"],
    ],
  },
];

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onClose={onClose} title="Keyboard shortcuts" footer={<Button onClick={onClose}>Close</Button>}>
      <div className={styles.shortcuts}>
        {SHORTCUTS.map((group) => (
          <section key={group.title} className={styles.shortcutGroup}>
            <h3 className={styles.shortcutTitle}>{group.title}</h3>
            <dl className={styles.shortcutList}>
              {group.rows.map(([keys, text]) => (
                <div key={text} className={styles.shortcutRow}>
                  <dt className={styles.shortcutKeys}>
                    {keys.map((key) => (
                      <Kbd key={key}>{key}</Kbd>
                    ))}
                  </dt>
                  <dd className={styles.shortcutText}>{text}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </Dialog>
  );
}

const MODEL_LABEL: Record<string, string> = { small: "Small", base: "Base", large: "Large" };
const QUALITY_LABEL: Record<string, string> = { fast: "Fast", balanced: "Balanced", high: "High" };

/** Run the chain of "everything else" again: another seed and, if wanted, another order. */
export function RebuildDialog({
  sounds,
  order: current,
  settings,
  onClose,
  onRebuild,
}: {
  sounds: Sound[];
  /** Sound ids in the order they were removed last time. */
  order: string[];
  settings: SeparationSettings;
  onClose: () => void;
  onRebuild: (order: string[]) => Promise<void>;
}) {
  const [order, setOrder] = useState<string[]>(() => {
    const known = current.filter((id) => sounds.some((s) => s.id === id));
    return [...known, ...sounds.filter((s) => !known.includes(s.id)).map((s) => s.id)];
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const name = (id: string) => sounds.find((s) => s.id === id)?.prompt ?? id;
  const move = (index: number, by: number) =>
    setOrder((list) => {
      const next = [...list];
      const [item] = next.splice(index, 1);
      next.splice(index + by, 0, item);
      return next;
    });
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await onRebuild(order);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onClose={onClose}
      title="Try everything else again"
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={busy} onClick={run}>
            {busy ? "Starting..." : `Rebuild (${order.length} model passes)`}
          </Button>
        </>
      }
    >
      <p className={styles.dialogText}>
        Everything else is made by the model alone: it removes the first sound from the original, the next one from
        what is left, and so on. Each pass regenerates the audio, so <strong>the sound removed first comes out
        cleanest</strong>. If you can still hear a sound in everything else, move it to the top and rebuild.
      </p>
      <ol className={styles.orderList}>
        {order.map((id, index) => (
          <li key={id} className={styles.orderItem}>
            <span className={`${styles.orderIndex} tabular`}>{index + 1}</span>
            <span className={styles.orderName}>{name(id)}</span>
            <IconButton label={`Remove ${name(id)} earlier`} size="small" disabled={index === 0} onClick={() => move(index, -1)}>
              <ArrowUp size={15} aria-hidden="true" />
            </IconButton>
            <IconButton
              label={`Remove ${name(id)} later`}
              size="small"
              disabled={index === order.length - 1}
              onClick={() => move(index, 1)}
            >
              <ArrowDown size={15} aria-hidden="true" />
            </IconButton>
          </li>
        ))}
      </ol>
      <p className={styles.dialogNote}>
        Uses the {MODEL_LABEL[settings.model_size]} model at {QUALITY_LABEL[settings.quality]} quality
        {settings.seed !== null ? ` with seed ${settings.seed}` : " with a new random seed"}. The version you have now
        is kept: switch between them on the Everything else track.
      </p>
      {error && <ErrorText>{error}</ErrorText>}
    </Dialog>
  );
}
