"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  Clock, Activity, TrendingUp, Users, AlertCircle,
  CheckCircle, XCircle, Loader2, Play, Copy, Check,
  ChevronDown, ChevronUp, Database, Zap, GitBranch,
} from "lucide-react";
import { api } from "@/lib/api";
import type { PipelineRunSummary } from "@/lib/types";

// ── Pipeline step definitions ─────────────────────────────────────────
const PIPELINE_STEPS = [
  { key: "source_intel",     label: "News Scraping",  desc: "Scrapes RSS feeds & live news"             },
  { key: "analysis",         label: "AI Clustering",  desc: "Groups articles into market trends"        },
  { key: "impact",           label: "Impact Eval",    desc: "Scores business impact per trend"          },
  { key: "quality",          label: "Quality Filter", desc: "Filters low-signal trends"                 },
  { key: "causal_council",   label: "AI Council",     desc: "Multi-agent causality verification"        },
  { key: "lead_crystallize", label: "Lead Harvest",   desc: "Identifies companies from trends"          },
  { key: "lead_gen",         label: "Outreach Gen",   desc: "Enriches contacts, generates email drafts" },
  { key: "learning_update",  label: "Self-Learning",  desc: "Updates model weights from this run"       },
];

function getStepIndex(step: string): number {
  const norm = step.replace(/\s+/g, "_").toLowerCase().replace("_complete", "");
  const idx = PIPELINE_STEPS.findIndex(s => norm.includes(s.key) || s.key.includes(norm));
  return idx;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
}

function formatTimeAgo(iso: string): string {
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  } catch { return ""; }
}

function formatAbsTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-IN", {
      day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
    });
  } catch { return iso; }
}

// Parse the run_id "20260227_092014" → "27 Feb 2026, 09:20 AM"
function friendlyRunTime(runId: string): string {
  try {
    const [dp, tp] = runId.split("_");
    const y = dp.slice(0, 4), m = dp.slice(4, 6), d = dp.slice(6, 8);
    const h = tp.slice(0, 2), min = tp.slice(2, 4);
    return new Date(`${y}-${m}-${d}T${h}:${min}`).toLocaleString("en-IN", {
      day: "numeric", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  } catch { return runId; }
}

// ── Status config map ─────────────────────────────────────────────────
const STATUS_MAP: Record<string, { color: string; bg: string; label: string }> = {
  completed: { color: "var(--green)",        bg: "var(--green-light)",  label: "Completed" },
  failed:    { color: "var(--red)",          bg: "var(--red-light)",    label: "Failed"    },
  running:   { color: "var(--accent)",       bg: "var(--accent-light)", label: "Running"   },
};

// ── Page ──────────────────────────────────────────────────────────────

export default function HistoryPage() {
  const [runs, setRuns] = useState<PipelineRunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getPipelineRuns(20).then(setRuns).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const completed   = runs.filter(r => r.status === "completed");
  const failed      = runs.filter(r => r.status === "failed");
  const totalTrends = completed.reduce((s, r) => s + r.trends_detected, 0);
  const totalLeads  = completed.reduce((s, r) => s + r.leads_generated, 0);
  const avgDuration = completed.length > 0
    ? completed.reduce((s, r) => s + r.elapsed_seconds, 0) / completed.length
    : 0;

  return (
    <>
      {/* ── Header ──────────────────────────────────── */}
      <div style={{ padding: "20px 24px 18px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 18 }}>
          <h1 className="font-display" style={{ fontSize: 22, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Pipeline History
          </h1>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{runs.length} runs recorded</span>
        </div>

        {/* ── Summary KPI cards ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
          {[
            { label: "Total Runs",       value: runs.length,               sub: "all time",           color: "var(--accent)"         },
            { label: "Completed",        value: completed.length,          sub: `${failed.length} failed`, color: "var(--green)"    },
            { label: "Trends Detected",  value: totalTrends,               sub: "across all runs",    color: "var(--blue)"           },
            { label: "Leads Generated",  value: totalLeads,                sub: "all sources",        color: "var(--green)"          },
            { label: "Avg Duration",     value: formatDuration(avgDuration), sub: "per completed run", color: "var(--text-secondary)" },
          ].map(({ label, value, sub, color }) => (
            <div key={label} style={{ padding: "14px 16px", background: "var(--bg)", borderRadius: 10, border: "1px solid var(--border)" }}>
              <div className="num" style={{ fontSize: 26, color, lineHeight: 1, marginBottom: 4 }}>{value}</div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text)", marginBottom: 2, letterSpacing: "0.01em" }}>{label}</div>
              <div style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Run list ─────────────────────────────────── */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        {loading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {[0, 1, 2, 3].map(i => (
              <div key={i} className="card" style={{ padding: "18px 20px" }}>
                <div style={{ display: "flex", gap: 14 }}>
                  <div className="skeleton" style={{ width: 36, height: 36, borderRadius: 10, flexShrink: 0 }} />
                  <div style={{ flex: 1 }}>
                    <div className="skeleton" style={{ height: 16, width: "35%", marginBottom: 8 }} />
                    <div className="skeleton" style={{ height: 11, width: "60%", marginBottom: 12 }} />
                    <div className="skeleton" style={{ height: 4, width: "100%", borderRadius: 2 }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : runs.length === 0 ? (
          <div style={{ padding: "60px 24px", textAlign: "center" }}>
            <Play size={32} style={{ color: "var(--text-xmuted)", margin: "0 auto 14px", display: "block" }} />
            <p style={{ fontSize: 14, color: "var(--text-muted)", fontWeight: 600, marginBottom: 6 }}>No pipeline runs yet</p>
            <p style={{ fontSize: 12, color: "var(--text-xmuted)" }}>
              Go to the Dashboard and run the pipeline to start generating intelligence.
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {runs.map((run, idx) => (
              <RunCard key={run.run_id} run={run} runNumber={runs.length - idx} />
            ))}
          </div>
        )}
      </div>
    </>
  );
}

// ── Run Card ──────────────────────────────────────────────────────────

function RunCard({ run, runNumber }: { run: PipelineRunSummary; runNumber: number }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied]     = useState(false);

  const stepIdx  = getStepIndex(run.current_step);
  const isFailed = run.status === "failed";

  const { color: statusColor, bg: statusBg, label: statusLabel } =
    STATUS_MAP[run.status] ?? { color: "var(--text-muted)", bg: "var(--surface-raised)", label: run.status };

  const copyId = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(run.run_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="card" style={{ overflow: "hidden" }}>

      {/* ── Card header (clickable to expand) ── */}
      <div
        style={{ padding: "16px 20px 12px", cursor: "pointer", display: "flex", alignItems: "flex-start", gap: 14 }}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Status icon */}
        <div style={{ width: 38, height: 38, borderRadius: 10, background: statusBg, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>
          {run.status === "completed" && <CheckCircle size={17} style={{ color: statusColor }} />}
          {run.status === "failed"    && <XCircle     size={17} style={{ color: statusColor }} />}
          {run.status === "running"   && <Loader2     size={17} style={{ color: statusColor, animation: "spin 1s linear infinite" }} />}
          {!["completed","failed","running"].includes(run.status) && <Clock size={17} style={{ color: statusColor }} />}
        </div>

        {/* Text content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Title row */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3, flexWrap: "wrap" }}>
            <span style={{ fontSize: 15, fontWeight: 700, color: "var(--text)" }}>Run #{runNumber}</span>
            <span className="badge" style={{ background: statusBg, color: statusColor, fontSize: 10 }}>
              {statusLabel}
            </span>
            <span style={{ fontSize: 12, color: "var(--text-muted)", marginLeft: "auto", whiteSpace: "nowrap" }}>
              {formatTimeAgo(run.started_at)}
              <span style={{ color: "var(--text-xmuted)", marginLeft: 6 }}>· {formatAbsTime(run.started_at)}</span>
            </span>
            {run.status === "completed" && run.trends_detected > 0 && (
              <Link
                href={`/history/${run.run_id}`}
                onClick={e => e.stopPropagation()}
                style={{
                  display: "flex", alignItems: "center", gap: 5,
                  padding: "4px 10px", fontSize: 11, fontWeight: 600,
                  color: "var(--accent)", border: "1px solid var(--accent)55",
                  background: "var(--accent-light)", borderRadius: 6, textDecoration: "none",
                  flexShrink: 0, transition: "opacity 150ms",
                }}
              >
                <GitBranch size={11} /> View Tree
              </Link>
            )}
          </div>

          {/* Run ID with copy */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12 }}>
            <span style={{ fontSize: 11, color: "var(--text-xmuted)", fontFamily: "monospace", letterSpacing: "0.04em" }}>
              {friendlyRunTime(run.run_id)}
            </span>
            <span style={{ color: "var(--border-strong)", fontSize: 10 }}>·</span>
            <span style={{ fontSize: 10, color: "var(--text-xmuted)", fontFamily: "monospace" }}>
              {run.run_id}
            </span>
            <button
              onClick={copyId}
              style={{
                display: "flex", alignItems: "center", gap: 3,
                padding: "2px 7px", fontSize: 9,
                color: copied ? "var(--green)" : "var(--text-xmuted)",
                background: copied ? "var(--green-light)" : "var(--surface-raised)",
                border: `1px solid ${copied ? "var(--green)" : "var(--border)"}`,
                borderRadius: 4, cursor: "pointer", transition: "all 200ms",
              }}
            >
              {copied ? <Check size={9} /> : <Copy size={9} />}
              {copied ? "Copied!" : "Copy ID"}
            </button>
          </div>

          {/* Stats row */}
          <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
            <StatChip icon={Clock}    label="Duration"   value={formatDuration(run.elapsed_seconds)} color="var(--text-secondary)" />
            <StatChip icon={TrendingUp} label="Trends"  value={run.trends_detected}  color={run.trends_detected  > 0 ? "var(--blue)"  : "var(--text-xmuted)"} />
            <StatChip icon={Database} label="Companies"  value={run.companies_found}  color={run.companies_found  > 0 ? "var(--green)" : "var(--text-xmuted)"} />
            <StatChip icon={Users}    label="Leads"      value={run.leads_generated}  color={run.leads_generated  > 0 ? "var(--green)" : "var(--text-xmuted)"} />
            {run.status !== "completed" && (
              <StatChip icon={Activity} label="at step" value={run.current_step.replace(/_/g, " ")} color="var(--accent)" />
            )}
          </div>
        </div>

        {/* Expand chevron */}
        <div style={{ flexShrink: 0, color: "var(--text-xmuted)", marginTop: 6 }}>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>

      {/* ── Pipeline step progress bar ── */}
      <div style={{ padding: "0 20px 16px" }}>
        <div style={{ display: "flex", gap: 3, alignItems: "flex-start" }}>
          {PIPELINE_STEPS.map((step, i) => {
            let state: "done" | "current" | "pending" = "pending";
            if (run.status === "completed")             state = "done";
            else if (i < stepIdx)                       state = "done";
            else if (i === stepIdx && !isFailed)        state = "current";

            const barColor =
              state === "done"    ? (isFailed ? "var(--red)" : "var(--green)") :
              state === "current" ? "var(--accent)" :
              "var(--border)";

            const labelColor =
              state === "done"    ? (isFailed ? "var(--red)" : "var(--text-secondary)") :
              state === "current" ? "var(--accent)" :
              "var(--text-xmuted)";

            return (
              <div key={step.key} style={{ flex: 1, textAlign: "center" }} title={`${step.label}: ${step.desc}`}>
                <div style={{ height: 4, borderRadius: 2, background: barColor, marginBottom: 5, transition: "background 300ms", position: "relative", overflow: "hidden" }}>
                  {state === "current" && (
                    <div style={{ position: "absolute", inset: 0, background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)", animation: "shimmer 1.4s infinite", backgroundSize: "200% 100%" }} />
                  )}
                </div>
                <span style={{ fontSize: 9, color: labelColor, fontWeight: state !== "pending" ? 700 : 400, letterSpacing: "0.01em", display: "block", lineHeight: 1.2 }}>
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Expanded: errors + step context ── */}
      {expanded && (
        <div style={{ borderTop: "1px solid var(--border)" }}>
          {/* Step context */}
          <div style={{ padding: "12px 20px", background: "var(--bg)", display: "flex", gap: 10, alignItems: "center" }}>
            <Zap size={12} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              Last step:{" "}
              <span style={{ color: "var(--text)", fontWeight: 600 }}>
                {run.current_step.replace(/_/g, " ")}
              </span>
            </span>
            {run.progress_pct < 100 && run.progress_pct > 0 && (
              <>
                <span style={{ color: "var(--border-strong)", fontSize: 10 }}>·</span>
                <span style={{ fontSize: 11, color: "var(--amber)" }}>{run.progress_pct}% complete</span>
              </>
            )}
          </div>

          {/* Errors */}
          {run.errors.length > 0 && (
            <div style={{ padding: "12px 20px", background: "var(--red-light)", borderTop: "1px solid var(--border)" }}>
              <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                <AlertCircle size={13} style={{ color: "var(--red)", flexShrink: 0, marginTop: 1 }} />
                <div>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "var(--red)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    {run.errors.length} Error{run.errors.length > 1 ? "s" : ""}
                  </div>
                  {run.errors.map((err, i) => (
                    <div key={i} style={{ fontSize: 12, color: "var(--red)", lineHeight: 1.6, marginBottom: 4 }}>
                      {err}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatChip({ label, value, icon: Icon, color }: { label: string; value: string | number; icon: React.ElementType; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <Icon size={11} style={{ color }} />
      <span className="num" style={{ fontSize: 13, color, fontWeight: 700 }}>{value}</span>
      <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{label}</span>
    </div>
  );
}
