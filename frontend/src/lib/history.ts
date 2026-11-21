"use client";

import { useMemo, useRef, useState } from "react";

export interface HistoryEntry {
  undo: () => void;
  redo: () => void;
  /** What kind of step this is, so steps that no longer apply can be dropped together. */
  tag?: string;
  /** Changes with the same key made in quick succession (a slider drag) undo as one step. */
  key?: string;
}

interface Stored extends HistoryEntry {
  time: number;
}

const MERGE_MS = 900;

/** Undo and redo for the workspace: a stack of reversible steps. */
export function useHistory(limit = 200) {
  const stacks = useRef<{ past: Stored[]; future: Stored[] }>({ past: [], future: [] });
  const [counts, setCounts] = useState({ undo: 0, redo: 0 });

  const controls = useMemo(() => {
    const sync = () => setCounts({ undo: stacks.current.past.length, redo: stacks.current.future.length });
    return {
      push(entry: HistoryEntry) {
        const { past } = stacks.current;
        const now = Date.now();
        const last = past[past.length - 1];
        if (entry.key && last && last.key === entry.key && now - last.time < MERGE_MS) {
          past[past.length - 1] = { ...entry, undo: last.undo, time: now };
        } else {
          past.push({ ...entry, time: now });
          if (past.length > limit) past.shift();
        }
        stacks.current.future = [];
        sync();
      },
      undo() {
        const entry = stacks.current.past.pop();
        if (!entry) return false;
        entry.undo();
        stacks.current.future.push({ ...entry, time: 0 });
        sync();
        return true;
      },
      redo() {
        const entry = stacks.current.future.pop();
        if (!entry) return false;
        entry.redo();
        stacks.current.past.push({ ...entry, time: 0 });
        sync();
        return true;
      },
      /** Forget the steps with this tag (what they would restore is gone). */
      drop(tag: string) {
        stacks.current.past = stacks.current.past.filter((entry) => entry.tag !== tag);
        stacks.current.future = stacks.current.future.filter((entry) => entry.tag !== tag);
        sync();
      },
    };
  }, [limit]);

  return useMemo(
    () => ({ ...controls, canUndo: counts.undo > 0, canRedo: counts.redo > 0 }),
    [controls, counts],
  );
}
