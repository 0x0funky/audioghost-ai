"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { AppHeader } from "@/components/AppHeader";
import { BatchView } from "@/components/BatchView";
import { HfDialog } from "@/components/HfDialog";
import { LicenseDialog } from "@/components/LicenseDialog";
import { StartScreen } from "@/components/StartScreen";
import { Button, Spinner } from "@/components/ui";
import { Workspace } from "@/components/Workspace";
import workspaceStyles from "@/components/Workspace.module.css";
import { useAuth, useSession, useSystem } from "@/lib/hooks";
import type { AuthStatus, Session, SystemStatus } from "@/lib/types";

function SessionView({
  id,
  system,
  auth,
  onHome,
  onConnect,
  onLicense,
  onOpenBatch,
}: {
  id: string;
  system: SystemStatus | null;
  auth: AuthStatus | null;
  onHome: () => void;
  onConnect: () => void;
  onLicense: () => void;
  onOpenBatch: (id: string) => void;
}) {
  const { session, error, refresh, setSession } = useSession(id);
  return (
    <>
      <AppHeader session={session} system={system} auth={auth} onHome={onHome} />
      {session ? (
        <Workspace
          session={session}
          onSessionChange={setSession}
          onRefresh={() => void refresh()}
          system={system}
          auth={auth}
          onConnect={onConnect}
          onLicense={onLicense}
          onOpenBatch={onOpenBatch}
        />
      ) : error?.status === 404 ? (
        <div className={workspaceStyles.state}>
          <p>This project no longer exists: it was deleted.</p>
          <Button variant="primary" onClick={onHome}>
            Start over
          </Button>
        </div>
      ) : (
        <div className={workspaceStyles.state} aria-live="polite">
          {error ? <p>{error.message}</p> : <Spinner size={22} />}
        </div>
      )}
    </>
  );
}

function App() {
  const params = useSearchParams();
  const router = useRouter();
  const sessionId = params.get("session");
  const batchId = params.get("batch");
  const { status: system } = useSystem();
  const { auth, refresh: refreshAuth } = useAuth();
  const [hfOpen, setHfOpen] = useState(false);
  const [licenseOpen, setLicenseOpen] = useState(false);

  const home = () => router.push("/");
  const open = (target: Session | string) => router.push(`/?session=${typeof target === "string" ? target : target.id}`);
  const openBatch = (id: string) => router.push(`/?batch=${id}`);

  return (
    <>
      {sessionId ? (
        <SessionView
          key={sessionId}
          id={sessionId}
          system={system}
          auth={auth}
          onHome={home}
          onConnect={() => setHfOpen(true)}
          onLicense={() => setLicenseOpen(true)}
          onOpenBatch={openBatch}
        />
      ) : batchId ? (
        <>
          <AppHeader system={system} auth={auth} onHome={home} />
          <BatchView key={batchId} id={batchId} onHome={home} onOpenSession={open} />
        </>
      ) : (
        <>
          <AppHeader system={system} auth={auth} onHome={home} />
          <StartScreen
            system={system}
            auth={auth}
            onConnect={() => setHfOpen(true)}
            onLicense={() => setLicenseOpen(true)}
            onOpen={open}
            onOpenBatch={openBatch}
          />
        </>
      )}
      <HfDialog open={hfOpen} onClose={() => setHfOpen(false)} onConnected={() => void refreshAuth()} />
      <LicenseDialog open={licenseOpen} onClose={() => setLicenseOpen(false)} onAccepted={() => void refreshAuth()} />
    </>
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <App />
    </Suspense>
  );
}
