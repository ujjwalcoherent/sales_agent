"use client";

import { useEffect, useState } from "react";
import { PipelineProvider } from "@/contexts/pipeline-context";
import { ProfileProvider } from "@/contexts/profile-context";
import { Sidebar } from "@/components/dashboard/sidebar";
import { TerminalPanel } from "@/components/dashboard/terminal-panel";
import { getMockMode, setMockMode } from "@/lib/api";
import { FlaskConical, X } from "lucide-react";

function MockModeBanner() {
  const [enabled, setEnabled] = useState(false);
  const [recording, setRecording] = useState("");
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    getMockMode().then(s => {
      setEnabled(s.enabled);
      setRecording(s.selected_recording);
    }).catch(() => {});
  }, []);

  if (!enabled || dismissed) return null;

  return (
    <div style={{
      background: "var(--amber-light)", borderBottom: "1px solid var(--accent)",
      padding: "6px 16px", display: "flex", alignItems: "center", gap: 8,
      fontSize: 12, color: "var(--text)", flexShrink: 0,
    }}>
      <FlaskConical size={13} style={{ color: "var(--accent)", flexShrink: 0 }} />
      <span>
        <strong style={{ color: "var(--accent)" }}>Mock Mode ON</strong>
        {recording && <span style={{ color: "var(--text-secondary)" }}> — using recording {recording}</span>}
        {" · "}All pipeline runs and campaigns use cached data · No live API calls
      </span>
      <div style={{ flex: 1 }} />
      <button
        onClick={async () => { await setMockMode(false); setEnabled(false); }}
        style={{ fontSize: 11, color: "var(--accent)", background: "none", border: "1px solid var(--accent)", borderRadius: 5, padding: "2px 8px", cursor: "pointer", fontWeight: 500 }}
      >
        Disable
      </button>
      <button onClick={() => setDismissed(true)} style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", display: "flex", padding: 2 }}>
        <X size={12} />
      </button>
    </div>
  );
}

/**
 * Client wrapper for the dashboard layout.
 * Keeps (dashboard)/layout.tsx as a Server Component while still
 * providing React Context (PipelineContext) and the TerminalPanel
 * to all dashboard pages.
 */
export function DashboardShell({ children }: { children: React.ReactNode }) {
  return (
    <ProfileProvider>
    <PipelineProvider>
      <div style={{ display: "flex", height: "100vh", background: "var(--bg)", overflow: "hidden" }}>
        <Sidebar />
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
          <MockModeBanner />
          {/* Page content fills remaining space above the terminal */}
          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", minHeight: 0 }}>
            {children}
          </div>
          {/* Terminal panel: always visible, collapsible, Ctrl+` to toggle */}
          <TerminalPanel />
        </div>
      </div>
    </PipelineProvider>
    </ProfileProvider>
  );
}
