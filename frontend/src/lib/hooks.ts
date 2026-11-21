"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import type { AuthStatus, Batch, Peaks, Session, SessionList, SystemStatus } from "./types";

/** Backend / Redis / worker status, re-checked often while something is wrong. */
export function useSystem() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      const next = await api.system();
      if (cancelled) return;
      setStatus(next);
      const healthy = next.reachable && next.redis && next.worker && !next.worker_starting;
      timer = setTimeout(poll, healthy ? 10_000 : 3_000);
    };
    timer = setTimeout(poll, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return { status, refresh };
}

export function useAuth() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const refresh = useCallback(async () => {
    try {
      setAuth(await api.auth());
    } catch {
      setAuth(null);
    }
  }, []);
  useEffect(() => {
    const first = setTimeout(refresh, 0);
    return () => clearTimeout(first);
  }, [refresh]);
  return { auth, refresh };
}

export function useRecentSessions() {
  const [list, setList] = useState<SessionList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    try {
      setList(await api.sessions());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);
  useEffect(() => {
    const first = setTimeout(refresh, 0);
    return () => clearTimeout(first);
  }, [refresh]);
  return { sessions: list?.sessions ?? null, list, error, refresh };
}

export function useBatches() {
  const [batches, setBatches] = useState<Batch[] | null>(null);
  const refresh = useCallback(async () => {
    try {
      setBatches((await api.batches()).batches);
    } catch {
      setBatches(null);
    }
  }, []);
  useEffect(() => {
    const first = setTimeout(refresh, 0);
    return () => clearTimeout(first);
  }, [refresh]);
  return { batches, refresh };
}

/** A batch, polled while files of it are still queued or running. */
export function useBatch(id: string) {
  const [batch, setBatch] = useState<Batch | null>(null);
  const [error, setError] = useState<{ message: string; status: number } | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const next = await api.batch(id);
        if (cancelled) return;
        setBatch(next);
        setError(null);
        const busy = (next.items ?? []).some((item) => item.status === "queued" || item.status === "running");
        timer = setTimeout(poll, busy ? 1500 : 15_000);
      } catch (err) {
        if (cancelled) return;
        const status = err instanceof ApiError ? err.status : 0;
        setError({ message: err instanceof Error ? err.message : String(err), status });
        if (status !== 404) timer = setTimeout(poll, 3000);
      }
    };
    timer = setTimeout(poll, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return { batch, error, refresh, setBatch };
}

const ACTIVE = new Set(["queued", "running"]);

/** A session, polled every second while a track is queued or running. */
export function useSession(id: string) {
  const [session, setSessionState] = useState<Session | null>(null);
  const [error, setError] = useState<{ message: string; status: number } | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const next = await api.session(id);
        if (cancelled) return;
        setSessionState(next);
        setError(null);
        const busy = next.tracks.some((t) => ACTIVE.has(t.status)) || ACTIVE.has(next.rest_job?.status ?? "");
        timer = setTimeout(poll, busy ? 1000 : 10_000);
      } catch (err) {
        if (cancelled) return;
        const status = err instanceof ApiError ? err.status : 0;
        setError({ message: err instanceof Error ? err.message : String(err), status });
        if (status !== 404) timer = setTimeout(poll, 3000);
      }
    };
    timer = setTimeout(poll, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  // A session handed back by an API call: show it now and restart polling (it may be busy).
  const setSession = useCallback((next: Session) => {
    setSessionState(next);
    setNonce((n) => n + 1);
  }, []);
  return { session, error, refresh, setSession };
}

const peaksCache = new Map<string, Promise<Peaks>>();

export function loadPeaks(url: string): Promise<Peaks> {
  let pending = peaksCache.get(url);
  if (!pending) {
    pending = api.peaks(url);
    pending.catch(() => peaksCache.delete(url));
    peaksCache.set(url, pending);
  }
  return pending;
}

export function usePeaks(url: string | null): number[] | null {
  const [loaded, setLoaded] = useState<{ url: string; peaks: number[] } | null>(null);
  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    loadPeaks(url)
      .then((data) => {
        if (!cancelled) setLoaded({ url, peaks: data.peaks });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [url]);
  return loaded && loaded.url === url ? loaded.peaks : null;
}
