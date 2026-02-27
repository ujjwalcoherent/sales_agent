"use client";

import { useState, useEffect } from "react";
import { Clock, Activity, TrendingUp, Users, AlertCircle, CheckCircle, XCircle, Loader2, Play } from "lucide-react";
import { api } from "@/lib/api";
import type { PipelineRunSummary } from "@/lib/types";

function statusIcon(status: string) {
  if (status === "completed") return <CheckCircle size={13} style={{ color: "var(--green)" }} />;
  if (status === "failed") return <XCircle size={13} style={{ color: "var(--red)" }} />;
  if (status === "running") return <Loader2 size={13} style={{ color: "var(--accent)", animation: "spin 1s linear infinite" }} />;
  return <Clock size={13} style={{ color: "var(--text-muted)" }} />;
}

function statusBadge(status: string) {
  const map: Record<string, { bg: string; color: string }> = {
    completed: { bg: "var(--green-light)", color: "var(--green)" },
    failed: { bg: "var(--red-light)", color: "var(--red)" },
    running: { bg: "var(--amber-light)", color: "var(--amber)" },
  };
  const { bg, color } = map[status] ?? { bg: "var(--surface-raised)", color: "var(--text-muted)" };
  return (
    <span className="badge" style={{ background: bg, color, fontSize: 10 }}>
      {status}
    </span>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString("en-IN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

export default function HistoryPage() {
  const [runs, setRuns] = useState<PipelineRunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getPipelineRuns(20)
      .then(setRuns)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const completedRuns = runs.filter(r => r.status === "completed");
  const totalTrends = completedRuns.reduce((s, r) => s + r.trends_detected, 0);
  const totalLeads = completedRuns.reduce((s, r) => s + r.leads_generated, 0);
  const avgDuration = completedRuns.length > 0
    ? completedRuns.reduce((s, r) => s + r.elapsed_seconds, 0) / completedRuns.length
    : 0;

  return (
    <>
      {/* Header */}
      <div style={{ padding: "16px 24px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 14 }}>
          <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Pipeline History
          </h1>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {runs.length} runs
          </span>
        </div>

        {/* Summary KPIs */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          {[
            { label: "Total Runs", value: runs.length, icon: Activity, color: "var(--accent)" },
            { label: "Completed", value: completedRuns.length, icon: CheckCircle, color: "var(--green)" },
            { label: "Total Trends", value: totalTrends, icon: TrendingUp, color: "var(--blue)" },
            { label: "Total Leads", value: totalLeads, icon: Users, color: "var(--green)" },
            { label: "Avg Duration", value: formatDuration(avgDuration), icon: Clock, color: "var(--text-secondary)" },
          ].map(({ label, value, icon: Icon, color }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 12px", background: "var(--surface-raised)", borderRadius: 7 }}>
              <Icon size={12} style={{ color }} />
              <span className="num" style={{ fontSize: 16, color, lineHeight: 1 }}>{value}</span>
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Table */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        {loading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {[0, 1, 2, 3].map(i => (
              <div key={i} className="card" style={{ padding: "14px 16px" }}>
                <div className="skeleton" style={{ height: 14, width: "60%", marginBottom: 6 }} />
                <div className="skeleton" style={{ height: 11, width: "40%" }} />
              </div>
            ))}
          </div>
        ) : runs.length === 0 ? (
          <div style={{ padding: "50px 24px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
            <Play size={28} style={{ color: "var(--text-xmuted)", margin: "0 auto 10px" }} />
            <p>No pipeline runs yet</p>
            <p style={{ fontSize: 12, color: "var(--text-xmuted)", marginTop: 4 }}>
              Go to the Dashboard and run the pipeline to generate intelligence.
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {runs.map((run) => (
              <RunCard key={run.run_id} run={run} />
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function RunCard({ run }: { run: PipelineRunSummary }) {
  return (
    <div className="card" style={{ padding: "14px 18px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        {statusIcon(run.status)}
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", fontFamily: "var(--font-mono, monospace)" }}>
          {run.run_id}
        </span>
        {statusBadge(run.status)}
        <span style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: "auto" }}>
          {formatTime(run.started_at)}
        </span>
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <StatPill label="Duration" value={formatDuration(run.elapsed_seconds)} />
        <StatPill label="Trends" value={run.trends_detected} color={run.trends_detected > 0 ? "var(--blue)" : undefined} />
        <StatPill label="Companies" value={run.companies_found} color={run.companies_found > 0 ? "var(--green)" : undefined} />
        <StatPill label="Leads" value={run.leads_generated} color={run.leads_generated > 0 ? "var(--green)" : undefined} />
        <StatPill label="Step" value={run.current_step.replace(/_/g, " ")} />
        {run.progress_pct < 100 && (
          <StatPill label="Progress" value={`${run.progress_pct}%`} color="var(--amber)" />
        )}
      </div>

      {run.errors.length > 0 && (
        <div style={{ marginTop: 8, padding: "6px 10px", background: "var(--red-light)", borderRadius: 6, fontSize: 11, color: "var(--red)", display: "flex", gap: 6, alignItems: "flex-start" }}>
          <AlertCircle size={11} style={{ flexShrink: 0, marginTop: 1 }} />
          <span>{run.errors[0]}</span>
        </div>
      )}
    </div>
  );
}

function StatPill({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{label}:</span>
      <span className="num" style={{ fontSize: 12, color: color || "var(--text-secondary)" }}>{value}</span>
    </div>
  );
}
