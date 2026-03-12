"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Play, Square, RefreshCw, ChevronDown, ChevronUp, CheckCircle2, AlertCircle } from "lucide-react";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { getMockMode } from "@/lib/api";
import type { LogEntry } from "@/hooks/use-pipeline";

const MIN_HEIGHT = 72;
const DEFAULT_HEIGHT = 200;
const MAX_HEIGHT = 480;
const CLOSE_THRESHOLD = 60; // px below MIN_HEIGHT → snap closed

export function TerminalPanel() {
  // Start with defaults — localStorage is read after hydration in useEffect
  // to avoid SSR/client mismatch that causes React hydration errors.
  const [open, setOpen] = useState(false);
  const [panelHeight, setPanelHeight] = useState(DEFAULT_HEIGHT);
  const [willClose, setWillClose] = useState(false);
  const willCloseRef = useRef(false);
  const logRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartH = useRef(0);

  const { status, progress, currentStep, messages, result, error, run, reset, forceCancel } = usePipelineContext();
  const [lastMockMode, setLastMockMode] = useState(false);
  const [verbosity, setVerbosity] = useState("standard");

  // Read all persisted preferences after hydration (avoids SSR mismatch)
  useEffect(() => {
    const h = parseInt(localStorage.getItem("harbinger_terminal_height") ?? "", 10);
    if (h > 0) setPanelHeight(h);
    if (localStorage.getItem("harbinger_terminal_open") === "true") setOpen(true);
  }, []);

  // Read verbosity from localStorage (syncs with Settings page)
  useEffect(() => {
    const stored = localStorage.getItem("harbinger_terminal_verbosity");
    if (stored) setVerbosity(stored);
    const onStorage = (e: StorageEvent) => {
      if (e.key === "harbinger_terminal_verbosity" && e.newValue) setVerbosity(e.newValue);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Also poll localStorage for same-tab changes (Settings page sets it directly)
  useEffect(() => {
    const id = setInterval(() => {
      const v = localStorage.getItem("harbinger_terminal_verbosity") ?? "standard";
      setVerbosity((prev) => (prev !== v ? v : prev));
    }, 2000);
    return () => clearInterval(id);
  }, []);

  // Persist terminal open/height to localStorage
  useEffect(() => { localStorage.setItem("harbinger_terminal_open", String(open)); }, [open]);
  useEffect(() => { localStorage.setItem("harbinger_terminal_height", String(panelHeight)); }, [panelHeight]);

  const filteredMessages = useMemo(() => filterByVerbosity(messages, verbosity), [messages, verbosity]);

  const isRunning  = status === "running";
  const isComplete = status === "complete";
  const isError    = status === "error";

  // Auto-expand when pipeline starts
  useEffect(() => {
    if (isRunning) setOpen(true);
  }, [isRunning]);

  // Auto-scroll to latest log line
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [filteredMessages]);

  // Ctrl+` to toggle
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.ctrlKey && e.key === "`") {
      e.preventDefault();
      setOpen((v) => !v);
    }
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Drag-to-resize handle (drag down past threshold → close)
  const onResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    dragStartY.current = e.clientY;
    dragStartH.current = panelHeight;

    const onMove = (ev: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = dragStartY.current - ev.clientY;
      const newH = dragStartH.current + delta;
      // overDrag = how many px below MIN_HEIGHT the user has pulled
      const overDrag = newH < MIN_HEIGHT ? MIN_HEIGHT - newH : 0;
      if (overDrag > CLOSE_THRESHOLD) {
        willCloseRef.current = true;
        setWillClose(true);
        setPanelHeight(MIN_HEIGHT);
      } else {
        willCloseRef.current = false;
        setWillClose(false);
        setPanelHeight(Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, newH)));
      }
    };
    const onUp = () => {
      isDragging.current = false;
      if (willCloseRef.current) {
        setOpen(false);
        setPanelHeight(DEFAULT_HEIGHT);
        willCloseRef.current = false;
        setWillClose(false);
      }
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [panelHeight]);

  const totalHeight = open ? panelHeight + 40 + 8 : 40;

  return (
    <div style={{
      flexShrink: 0,
      borderTop: "1px solid var(--term-border)",
      background: "var(--term-bg)",
      display: "flex",
      flexDirection: "column",
      height: totalHeight,
      overflow: "hidden",
      transition: isDragging.current ? "none" : "height 200ms cubic-bezier(0.23,1,0.32,1)",
    }}>

      {/* ── Resize / close handle (only when open) ──────── */}
      {open && (
        <div
          onMouseDown={onResizeMouseDown}
          title="Drag to resize · drag down to close"
          style={{
            height: 8,
            flexShrink: 0,
            cursor: "row-resize",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: willClose ? "rgba(204,102,102,0.10)" : "transparent",
            transition: "background 150ms",
            userSelect: "none",
          }}
        >
          {!willClose
            ? <div style={{ width: 32, height: 2, borderRadius: 1, background: "var(--term-handle)" }} />
            : <span style={{ fontSize: 9, fontWeight: 700, color: "var(--term-red-bright)", letterSpacing: "0.08em", whiteSpace: "nowrap", pointerEvents: "none" }}>↓ RELEASE TO CLOSE</span>
          }
        </div>
      )}

      {/* ── Header bar ─────────────────────────────────── */}
      <div style={{ height: 40, flexShrink: 0, display: "flex", alignItems: "center", gap: 10, padding: "0 14px", borderBottom: open ? "1px solid var(--term-border)" : "none" }}>

        {/* Toggle */}
        <button
          onClick={() => setOpen((v) => !v)}
          style={{ display: "flex", alignItems: "center", gap: 5, background: "none", border: "none", cursor: "pointer", padding: "4px 0" }}
        >
          {open
            ? <ChevronDown size={10} style={{ color: "var(--term-text-dim)" }} />
            : <ChevronUp size={10} style={{ color: "var(--term-text-dim)" }} />}
        </button>

        <div style={{ width: 1, height: 14, background: "var(--term-border)" }} />

        {/* Action button — auto-detects mock mode from server */}
        {!isRunning ? (
          <div style={{ display: "flex", gap: 5 }}>
            <button onClick={async () => {
              const mock = await getMockMode().then(s => s.enabled).catch(() => false);
              setLastMockMode(mock);
              run(mock);
            }} style={tbtn("primary")}>
              <Play size={10} strokeWidth={2.5} /> Run
            </button>
          </div>
        ) : (
          <button onClick={forceCancel} style={tbtn("stop")}>
            <Square size={10} /> Stop
          </button>
        )}

        {/* Status */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, minWidth: 0, overflow: "hidden" }}>
          {status === "idle" && (
            <span style={{ fontSize: 10, color: "var(--term-text-dim)" }}>Ready — Ctrl+` to toggle · mock mode auto-detected</span>
          )}

          {isRunning && (
            <>
              <span className="dot dot-amber dot-pulse" style={{ width: 6, height: 6, flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: "var(--term-text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                {currentStep || "Initializing..."}
              </span>
              <div style={{ width: 100, height: 2, background: "var(--term-border)", borderRadius: 1, overflow: "hidden", flexShrink: 0 }}>
                <div style={{ height: "100%", width: progress > 0 ? `${progress}%` : "30%", background: `linear-gradient(90deg, var(--term-accent-dark), var(--term-accent))`, borderRadius: 1, transition: progress > 0 ? "width 500ms ease" : "none", ...(progress === 0 ? { animation: "progress-indeterminate 1.4s ease-in-out infinite" } : {}) }} />
              </div>
              {progress > 0 && <span style={{ fontSize: 10, color: "var(--term-text-dim)", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>{progress}%</span>}
            </>
          )}

          {isComplete && result && (
            <>
              <CheckCircle2 size={11} style={{ color: "var(--term-green)", flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: "var(--term-green)" }}>
                {result.leads_generated} leads · {result.run_time_seconds.toFixed(1)}s
              </span>
              <button onClick={reset} style={{ ...tbtn("ghost"), marginLeft: 4 }}>
                <RefreshCw size={10} /> Reset
              </button>
            </>
          )}

          {isError && (
            <>
              <AlertCircle size={11} style={{ color: "var(--term-red)", flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: "var(--term-red)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 200 }}>{error || "Pipeline failed"}</span>
              {error?.includes("Force Reset") ? (
                <button
                  onClick={async () => { await forceCancel(); run(lastMockMode); }}
                  style={{ ...tbtn("ghost"), color: "#fff", background: "var(--term-red)", borderColor: "var(--term-red)", marginLeft: 4, fontWeight: 700 }}
                >
                  Force Reset
                </button>
              ) : (
                <button onClick={reset} style={{ ...tbtn("ghost"), color: "var(--term-red)", borderColor: "var(--term-red-dim)", marginLeft: 4 }}>
                  <RefreshCw size={10} /> Retry
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* ── Log body ─────────────────────────────────────── */}
      <div
        ref={logRef}
        style={{ flex: 1, overflowY: "auto", padding: "8px 14px 12px", display: "flex", flexDirection: "column", gap: 0 }}
      >
        {filteredMessages.length === 0 ? (
          <div style={{ fontSize: 11, color: "var(--term-text-dim)", fontFamily: "'Courier New', monospace", padding: "4px 0" }}>
            › No output yet. Hit Run above (uses mock data when mock mode is enabled).
          </div>
        ) : (
          filteredMessages.map((entry, i, arr) => {
            const isLatest = i === arr.length - 1;
            const isWarn = entry.level === "warning";
            const isErr = entry.level === "error";
            return (
              <div
                key={i}
                style={{
                  fontSize: 11,
                  fontFamily: "'Courier New', monospace",
                  color: isErr ? "var(--term-red)" : isWarn ? "var(--term-accent)" : isLatest ? "var(--term-text)" : "var(--term-text-dim)",
                  lineHeight: 1.6,
                  paddingTop: 1,
                  paddingBottom: 1,
                }}
              >
                <span style={{ color: isErr ? "var(--term-red-dim)" : isWarn ? "var(--term-accent-dark)" : isLatest ? "var(--term-handle)" : "var(--term-text-xdim)", marginRight: 8 }}>›</span>
                {entry.text}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ── Verbosity filter ────────────────────────────────
// Minimal:  step markers, pipeline start/complete, errors
// Standard: + phase summaries, synthesis results, key metrics
// Verbose:  everything (raw technical detail)

const MINIMAL_PATTERNS = [
  /^===/, /^STEP \d/, /Pipeline START/, /Pipeline.*completed/i,
  /^Starting (pipeline|mock)/i, /complete:/, /\d+ leads/,
];

const NOISE_PATTERNS = [
  /^V10:/, /^V4:/, /^V9/, /^Optuna/, /^Hybrid similarity/,
  /^Leiden:/, /^Cluster quality/, /^CMI (gate|similarity)/,
  /dedup.*threshold/, /^Article sampling/, /^Phase \d\.\d/,
  /^At threshold/, /cooldown/, /Rate limited/,
  /^Track [AB]/, /providers exhausted/, /^\s*$/,
  /applying cooldown/, /Hard failure on/, /^All providers/,
  /waiting.*cooldown/, /^Sanitized synthesis/,
];

function filterByVerbosity(messages: LogEntry[], verbosity: string): LogEntry[] {
  if (verbosity === "verbose") return messages;

  return messages.filter((entry) => {
    const t = entry.text;

    if (verbosity === "minimal") {
      // Only show step markers, start/complete, and errors
      if (entry.level === "error") return true;
      return MINIMAL_PATTERNS.some((p) => p.test(t));
    }

    // Standard: filter out noisy technical details
    if (NOISE_PATTERNS.some((p) => p.test(t))) return false;
    return true;
  });
}

function tbtn(v: "primary" | "demo" | "stop" | "ghost"): React.CSSProperties {
  const base: React.CSSProperties = {
    display: "flex", alignItems: "center", gap: 4,
    padding: "4px 10px", borderRadius: 5, fontSize: 10, fontWeight: 600,
    cursor: "pointer", letterSpacing: "0.02em", flexShrink: 0,
  };
  if (v === "primary") return { ...base, border: "none",                              background: "var(--term-accent)", color: "var(--term-bg)" };
  if (v === "demo")    return { ...base, border: "1px solid var(--term-text-xdim)",   background: "transparent",        color: "var(--term-accent)" };
  if (v === "stop")    return { ...base, border: "1px solid var(--term-text-xdim)",   background: "transparent",        color: "var(--term-text-muted)" };
  return                      { ...base, border: "1px solid var(--term-border)",       background: "transparent",        color: "var(--term-text-dim)" };
}
