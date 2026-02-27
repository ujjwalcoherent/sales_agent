"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Play, Square, RefreshCw, FlaskConical, ChevronDown, ChevronUp, Terminal, CheckCircle2, AlertCircle } from "lucide-react";
import { usePipelineContext } from "@/contexts/pipeline-context";
import type { LogEntry } from "@/hooks/use-pipeline";

const MIN_HEIGHT = 72;
const DEFAULT_HEIGHT = 200;
const MAX_HEIGHT = 480;

export function TerminalPanel() {
  const [open, setOpen] = useState(false);
  const [panelHeight, setPanelHeight] = useState(DEFAULT_HEIGHT);
  const logRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartH = useRef(0);

  const { status, progress, currentStep, messages, result, error, run, reset } = usePipelineContext();
  const [verbosity, setVerbosity] = useState("standard");

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

  // Drag-to-resize handle
  const onResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    dragStartY.current = e.clientY;
    dragStartH.current = panelHeight;

    const onMove = (ev: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = dragStartY.current - ev.clientY;
      setPanelHeight(Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, dragStartH.current + delta)));
    };
    const onUp = () => {
      isDragging.current = false;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [panelHeight]);

  const totalHeight = open ? panelHeight + 40 : 40;

  return (
    <div style={{
      flexShrink: 0,
      borderTop: "1px solid #252318",
      background: "#131210",
      display: "flex",
      flexDirection: "column",
      height: totalHeight,
      overflow: "hidden",
      transition: isDragging.current ? "none" : "height 200ms cubic-bezier(0.23,1,0.32,1)",
    }}>

      {/* ── Resize handle (only when open) ──────────────── */}
      {open && (
        <div
          onMouseDown={onResizeMouseDown}
          title="Drag to resize"
          style={{
            height: 5,
            flexShrink: 0,
            cursor: "row-resize",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "transparent",
          }}
        >
          <div style={{ width: 28, height: 2, borderRadius: 1, background: "#3A3828" }} />
        </div>
      )}

      {/* ── Header bar ─────────────────────────────────── */}
      <div style={{ height: 40, flexShrink: 0, display: "flex", alignItems: "center", gap: 10, padding: "0 14px", borderBottom: open ? "1px solid #252318" : "none" }}>

        {/* Toggle */}
        <button
          onClick={() => setOpen((v) => !v)}
          style={{ display: "flex", alignItems: "center", gap: 5, background: "none", border: "none", cursor: "pointer", padding: "4px 0" }}
        >
          <Terminal size={12} style={{ color: "#6A6858" }} />
          <span style={{ fontSize: 11, color: "#6A6858", fontWeight: 700, letterSpacing: "0.06em" }}>PIPELINE</span>
          {open
            ? <ChevronDown size={10} style={{ color: "#4A4838" }} />
            : <ChevronUp size={10} style={{ color: "#4A4838" }} />}
        </button>

        <div style={{ width: 1, height: 14, background: "#2A2820" }} />

        {/* Action buttons */}
        {!isRunning ? (
          <div style={{ display: "flex", gap: 5 }}>
            <button onClick={() => run(false)} style={tbtn("primary")}>
              <Play size={10} strokeWidth={2.5} /> Run
            </button>
            <button onClick={() => run(true)} style={tbtn("demo")}>
              <FlaskConical size={10} /> Mock
            </button>
          </div>
        ) : (
          <button onClick={reset} style={tbtn("stop")}>
            <Square size={10} /> Stop
          </button>
        )}

        {/* Status */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, minWidth: 0, overflow: "hidden" }}>
          {status === "idle" && (
            <span style={{ fontSize: 10, color: "#3A3828" }}>Ready — Ctrl+` to toggle</span>
          )}

          {isRunning && (
            <>
              <span className="dot dot-amber dot-pulse" style={{ width: 6, height: 6, flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: "#9A9080", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                {currentStep || "Initializing..."}
              </span>
              <div style={{ width: 100, height: 2, background: "#2A2820", borderRadius: 1, overflow: "hidden", flexShrink: 0 }}>
                <div style={{ height: "100%", width: progress > 0 ? `${progress}%` : "30%", background: "linear-gradient(90deg, #8A5520, #C4892A)", borderRadius: 1, transition: progress > 0 ? "width 500ms ease" : "none", ...(progress === 0 ? { animation: "progress-indeterminate 1.4s ease-in-out infinite" } : {}) }} />
              </div>
              {progress > 0 && <span style={{ fontSize: 10, color: "#6A6858", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>{progress}%</span>}
            </>
          )}

          {isComplete && result && (
            <>
              <CheckCircle2 size={11} style={{ color: "#5ABF7A", flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: "#5ABF7A" }}>
                {result.leads_generated} leads · {result.run_time_seconds.toFixed(1)}s
              </span>
              <button onClick={reset} style={{ ...tbtn("ghost"), marginLeft: 4 }}>
                <RefreshCw size={10} /> Reset
              </button>
            </>
          )}

          {isError && (
            <>
              <AlertCircle size={11} style={{ color: "#CC6666", flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: "#CC6666", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{error || "Pipeline failed"}</span>
              <button onClick={reset} style={{ ...tbtn("ghost"), color: "#CC6666", borderColor: "#3A2020", marginLeft: 4 }}>
                <RefreshCw size={10} /> Retry
              </button>
            </>
          )}
        </div>

        <span style={{ fontSize: 9, color: "#2A2820", flexShrink: 0, letterSpacing: "0.04em" }}>Ctrl+`</span>
      </div>

      {/* ── Log body ─────────────────────────────────────── */}
      <div
        ref={logRef}
        style={{ flex: 1, overflowY: "auto", padding: "8px 14px 12px", display: "flex", flexDirection: "column", gap: 0 }}
      >
        {filteredMessages.length === 0 ? (
          <div style={{ fontSize: 11, color: "#3A3828", fontFamily: "'Courier New', monospace", padding: "4px 0" }}>
            › No output yet. Hit Run or Mock above.
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
                  color: isErr ? "#CC6666" : isWarn ? "#C4892A" : isLatest ? "#E8DFC0" : "#7A7060",
                  lineHeight: 1.6,
                  paddingTop: 1,
                  paddingBottom: 1,
                }}
              >
                <span style={{ color: isErr ? "#993333" : isWarn ? "#8A5520" : isLatest ? "#6A7A3A" : "#3A4018", marginRight: 8 }}>›</span>
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
  if (v === "primary") return { ...base, border: "none",                background: "#C4892A", color: "#0E0D09" };
  if (v === "demo")    return { ...base, border: "1px solid #4A4030",   background: "transparent", color: "#C4892A" };
  if (v === "stop")    return { ...base, border: "1px solid #3A3828",   background: "transparent", color: "#8A8070" };
  return                      { ...base, border: "1px solid #2A2820",   background: "transparent", color: "#6A6858" };
}
