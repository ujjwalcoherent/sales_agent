"use client";

import { PipelineProvider } from "@/contexts/pipeline-context";
import { Sidebar } from "@/components/dashboard/sidebar";
import { TerminalPanel } from "@/components/dashboard/terminal-panel";

/**
 * Client wrapper for the dashboard layout.
 * Keeps (dashboard)/layout.tsx as a Server Component while still
 * providing React Context (PipelineContext) and the TerminalPanel
 * to all dashboard pages.
 */
export function DashboardShell({ children }: { children: React.ReactNode }) {
  return (
    <PipelineProvider>
      <div style={{ display: "flex", height: "100vh", background: "var(--bg)", overflow: "hidden" }}>
        <Sidebar />
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
          {/* Page content fills remaining space above the terminal */}
          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", minHeight: 0 }}>
            {children}
          </div>
          {/* Terminal panel: always visible, collapsible, Ctrl+` to toggle */}
          <TerminalPanel />
        </div>
      </div>
    </PipelineProvider>
  );
}
