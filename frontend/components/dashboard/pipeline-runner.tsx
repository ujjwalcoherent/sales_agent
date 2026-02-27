"use client";

import { Play, Square, RefreshCw, AlertCircle, CheckCircle2, FlaskConical } from "lucide-react";
import type { PipelineState } from "@/hooks/use-pipeline";

interface PipelineRunnerProps {
  pipeline: PipelineState & {
    run: (mockMode?: boolean) => void;
    reset: () => void;
  };
}

export function PipelineRunner({ pipeline }: PipelineRunnerProps) {
  const { status, progress, currentStep, messages, result, error, run, reset } = pipeline;

  const isRunning  = status === "running";
  const isComplete = status === "complete";
  const isError    = status === "error";

  return (
    <div>
      {/* Top bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 20px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
          flexWrap: "wrap",
        }}
      >
        {/* Action buttons */}
        {!isRunning ? (
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={() => run(false)}
              style={btnStyle("primary")}
              onMouseEnter={(e) => { e.currentTarget.style.opacity = "0.82"; }}
              onMouseLeave={(e) => { e.currentTarget.style.opacity = "1"; }}
            >
              <Play size={12} strokeWidth={2.5} />
              Run Pipeline
            </button>
            <button
              onClick={() => run(true)}
              style={btnStyle("demo")}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-hover)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "var(--surface)"; }}
            >
              <FlaskConical size={12} strokeWidth={2} />
              Mock Replay
            </button>
          </div>
        ) : (
          <button
            onClick={reset}
            style={btnStyle("stop")}
          >
            <Square size={12} strokeWidth={2.5} />
            Stop
          </button>
        )}

        {/* Status display */}
        {status === "idle" && (
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            Ready — <strong>Mock Replay</strong> replays a recorded run in ~45s.
          </span>
        )}

        {isRunning && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 0 }}>
            <span className="dot dot-amber dot-pulse" />
            <span style={{ fontSize: 12, color: "var(--text-secondary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 260 }}>
              {currentStep || "Initializing..."}
            </span>
            <div style={{ flex: 1, height: 3, background: "var(--border)", borderRadius: 4, overflow: "hidden", maxWidth: 180 }}>
              <div
                style={{
                  height: "100%",
                  width: progress > 0 ? `${progress}%` : "30%",
                  background: "linear-gradient(90deg, var(--accent-mid), var(--amber))",
                  borderRadius: 4,
                  transition: progress > 0 ? "width 500ms ease" : "none",
                  ...(progress === 0 ? { animation: "progress-indeterminate 1.4s ease-in-out infinite" } : {}),
                }}
              />
            </div>
            <span style={{ fontSize: 11, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
              {progress > 0 ? `${progress}%` : ""}
            </span>
          </div>
        )}

        {isComplete && result && (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <CheckCircle2 size={13} style={{ color: "var(--green)" }} />
              <span style={{ fontSize: 12, color: "var(--green)" }}>
                {result.leads_generated} leads · {result.run_time_seconds.toFixed(1)}s
              </span>
            </div>
            <button onClick={reset} style={btnStyle("ghost")}>
              <RefreshCw size={11} />
              Reset
            </button>
            <div style={{ marginLeft: "auto", display: "flex", gap: 16 }}>
              {[
                { label: "Trends",    v: result.trends_detected },
                { label: "Companies", v: result.companies_found },
                { label: "Leads",     v: result.leads_generated },
              ].map(({ label, v }) => (
                <div key={label} style={{ textAlign: "center" }}>
                  <div className="num" style={{ fontSize: 15, color: "var(--text)", lineHeight: 1 }}>{v}</div>
                  <div style={{ fontSize: 9, color: "var(--text-xmuted)", marginTop: 2 }}>{label}</div>
                </div>
              ))}
            </div>
          </>
        )}

        {isError && (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <AlertCircle size={13} style={{ color: "var(--red)" }} />
            <span style={{ fontSize: 12, color: "var(--red)" }}>{error || "Pipeline failed"}</span>
            <button onClick={reset} style={{ ...btnStyle("ghost"), color: "var(--red)", borderColor: "var(--red-light)", background: "var(--red-light)" }}>
              <RefreshCw size={11} />
              Retry
            </button>
          </div>
        )}
      </div>

      {/* Dark terminal log — only while running */}
      {isRunning && messages.length > 0 && (
        <div
          style={{
            padding: "8px 20px",
            background: "#1A1916",
            borderBottom: "1px solid #2A2920",
            maxHeight: 72,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 1,
          }}
        >
          {messages.slice(-4).map((entry, i, arr) => (
            <div
              key={i}
              style={{
                fontSize: 11,
                fontFamily: "var(--font-mono, 'Courier New', monospace)",
                color: i === arr.length - 1 ? "#D4C9A0" : "#55534A",
                lineHeight: 1.5,
              }}
            >
              <span style={{ color: "#4A5A2A", marginRight: 8 }}>›</span>
              {typeof entry === "string" ? entry : entry.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Style helpers ──────────────────────────────────────────────────────

function btnStyle(variant: "primary" | "demo" | "stop" | "ghost"): React.CSSProperties {
  const base: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 5,
    padding: "7px 13px",
    borderRadius: 7,
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
    letterSpacing: "0.01em",
    transition: "opacity 150ms, background 150ms",
    flexShrink: 0,
  };
  if (variant === "primary") return { ...base, border: "none", background: "var(--text)", color: "#F8F7F2" };
  if (variant === "demo")    return { ...base, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--accent)" };
  if (variant === "stop")    return { ...base, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text-secondary)" };
  return                            { ...base, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text-secondary)" };
}
