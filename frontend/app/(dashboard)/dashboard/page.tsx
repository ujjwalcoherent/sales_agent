"use client";

import { useState, useCallback, useEffect } from "react";
import { Clock, RefreshCw, Zap } from "lucide-react";
import { KpiCards } from "@/components/dashboard/kpi-cards";
import { TrendsFeed } from "@/components/dashboard/trends-feed";
import { LeadsCompact } from "@/components/dashboard/leads-compact";
import { LeadDetailPanel } from "@/components/dashboard/lead-detail-panel";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import type { LeadRecord, TrendData, DashboardStats } from "@/lib/types";

export default function DashboardPage() {
  const { status, leads, trends, lastRunTime, initialLoading } = usePipelineContext();
  const [selectedTrend, setSelectedTrend] = useState<TrendData | null>(null);
  const [leadsLoading, setLeadsLoading] = useState(false);
  const [extraLeads, setExtraLeads] = useState<LeadRecord[]>([]);
  const [panelLead, setPanelLead] = useState<LeadRecord | null>(null);
  const [runCount, setRunCount] = useState(0);

  // Load run count on mount
  useEffect(() => {
    api.getPipelineRuns(100).then((runs) => setRunCount(runs.length)).catch(() => {});
  }, []);

  const refreshLeads = useCallback(async () => {
    setLeadsLoading(true);
    try {
      const { leads: fresh } = await api.getLeads({ limit: 100 });
      if (fresh.length > 0) setExtraLeads(fresh);
    } catch {
      // keep current data
    } finally {
      setLeadsLoading(false);
    }
  }, []);

  // Merge real leads (from refresh) over context leads (from pipeline)
  const displayLeads = extraLeads.length > 0 ? extraLeads : leads;

  // Trends from context (populated after pipeline run or mount fetch)
  const displayTrends = trends;

  const stats: DashboardStats = {
    trendsDetected:  displayTrends.length,
    companiesFound:  new Set(displayLeads.map((l) => l.company_name)).size,
    leadsGenerated:  displayLeads.length,
    pipelineRuns:    runCount,
  };

  return (
    <>
      {/* Page header */}
      <div style={{ padding: "14px 24px", display: "flex", alignItems: "center", gap: 10, borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <h1 className="font-display" style={{ fontSize: 19, color: "var(--text)", letterSpacing: "-0.02em" }}>
          Dashboard
        </h1>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {lastRunTime && (
            <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-muted)", padding: "5px 10px", background: "var(--surface-raised)", borderRadius: 6, border: "1px solid var(--border)" }}>
              <Clock size={10} /> {lastRunTime}
            </span>
          )}
          <button
            onClick={refreshLeads}
            disabled={leadsLoading}
            style={{ display: "flex", alignItems: "center", gap: 5, padding: "5px 12px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", cursor: leadsLoading ? "not-allowed" : "pointer", fontSize: 12, color: "var(--text-secondary)", opacity: leadsLoading ? 0.6 : 1 }}
          >
            <RefreshCw size={12} style={{ animation: leadsLoading ? "spin 1s linear infinite" : "none" }} />
            Refresh
          </button>
        </div>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
        <KpiCards stats={stats} loading={initialLoading || status === "running"} />

        {/* Two-column layout */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, minHeight: 480 }}>
          {/* Trends */}
          <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <SectionHeader title="Market Trends" count={displayTrends.length} unit="signals" />
            <div style={{ flex: 1, overflow: "auto", padding: 10 }}>
              <TrendsFeed trends={displayTrends} loading={initialLoading || status === "running"} selectedId={selectedTrend?.id} onSelect={setSelectedTrend} />
            </div>
          </div>

          {/* Leads */}
          <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <SectionHeader title="Leads" count={displayLeads.length} unit="generated" />
            <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
              {initialLoading ? (
                <div style={{ padding: 10, display: "flex", flexDirection: "column", gap: 8 }}>
                  {[0, 1, 2].map((i) => (
                    <div key={i} style={{ padding: "10px 12px", borderRadius: 8, background: "var(--surface-raised)" }}>
                      <div className="skeleton" style={{ height: 13, width: "60%", marginBottom: 6 }} />
                      <div className="skeleton" style={{ height: 11, width: "80%" }} />
                    </div>
                  ))}
                </div>
              ) : (
                <LeadsCompact leads={displayLeads} maxVisible={6} onSelect={setPanelLead} />
              )}
            </div>
          </div>
        </div>

        {/* Idle hint */}
        {displayLeads.length === 0 && status === "idle" && !initialLoading && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "14px 18px", borderRadius: 10, background: "var(--accent-light)", border: "1px solid var(--border)" }}>
            <Zap size={16} style={{ color: "var(--accent)", flexShrink: 0 }} />
            <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              Use <strong>Run Pipeline</strong> or <strong>Mock Replay</strong> in the Pipeline bar below to get started.
            </span>
          </div>
        )}
      </div>

      <LeadDetailPanel lead={panelLead} onClose={() => setPanelLead(null)} />
    </>
  );
}

function SectionHeader({ title, count, unit }: { title: string; count: number; unit: string }) {
  return (
    <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
      <h2 className="font-display" style={{ fontSize: 14, color: "var(--text)", letterSpacing: "-0.01em" }}>{title}</h2>
      <span style={{ fontSize: 11, background: "var(--surface-raised)", color: "var(--text-muted)", padding: "2px 8px", borderRadius: 999 }}>
        {count} {unit}
      </span>
    </div>
  );
}
