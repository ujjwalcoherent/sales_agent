"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Zap, RefreshCw, Building2, Users, TrendingUp, BarChart2,
  Clock, CheckCircle, XCircle, AlertCircle, Loader2,
  ArrowRight, Globe, ChevronRight, Activity,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import { usePipelineContext } from "@/contexts/pipeline-context";
import type { LeadRecord, HealthResponse, PipelineRunSummary } from "@/lib/types";

/* ── KPI card ─────────────────────────────────────────────── */

function KpiCard({
  label, value, sub, icon, color, loading,
}: {
  label: string;
  value: number | string;
  sub?: string;
  icon: React.ReactNode;
  color: string;
  loading?: boolean;
}) {
  return (
    <div className="card" style={{ padding: "16px 18px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {label}
        </span>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: `${color}18`,
          color: color,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          {icon}
        </div>
      </div>
      {loading ? (
        <div className="skeleton" style={{ height: 28, width: "60%", borderRadius: 6 }} />
      ) : (
        <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
          <span className="num" style={{ fontSize: 28, fontWeight: 700, color: "var(--text)", lineHeight: 1, letterSpacing: "-0.03em" }}>
            {value}
          </span>
          {sub && <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{sub}</span>}
        </div>
      )}
    </div>
  );
}

/* ── Run Pipeline button / state ─────────────────────────── */

function RunPipelineButton({
  status,
  onClick,
}: {
  status: string;
  onClick: () => void;
}) {
  const isRunning = status === "running";

  return (
    <button
      onClick={onClick}
      disabled={isRunning}
      style={{
        display: "inline-flex", alignItems: "center", gap: 7,
        padding: "8px 18px", borderRadius: 8, fontSize: 13, fontWeight: 600,
        background: isRunning ? "var(--surface-raised)" : "var(--accent)",
        color: isRunning ? "var(--text-muted)" : "#fff",
        border: "none", cursor: isRunning ? "not-allowed" : "pointer",
        transition: "all 150ms",
      }}
    >
      {isRunning
        ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} />
        : <Zap size={13} />
      }
      {isRunning ? "Pipeline running..." : "Run Pipeline"}
    </button>
  );
}

/* ── Lead row ─────────────────────────────────────────────── */

function LeadRow({
  lead,
  onClick,
}: {
  lead: LeadRecord;
  onClick: () => void;
}) {
  const conf = lead.confidence ?? 0;
  const confColor = conf >= 0.75 ? "var(--green)" : conf >= 0.5 ? "var(--accent)" : "var(--text-muted)";

  return (
    <div
      onClick={onClick}
      style={{
        display: "flex", alignItems: "center", gap: 12, padding: "10px 14px",
        cursor: "pointer", borderBottom: "1px solid var(--border)",
        transition: "background 120ms",
      }}
      onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-raised)")}
      onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
    >
      {/* Company initial */}
      <div style={{
        width: 32, height: 32, borderRadius: 8, flexShrink: 0,
        background: "var(--accent-light)", color: "var(--accent)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 13, fontWeight: 700,
      }}>
        {(lead.company_name?.[0] ?? "?").toUpperCase()}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {lead.company_name}
        </div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {lead.trend_title || lead.reason_relevant || lead.lead_type}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
        {/* Confidence */}
        <span style={{
          fontSize: 10, fontWeight: 600, padding: "2px 6px", borderRadius: 999,
          color: confColor,
          background: conf >= 0.75 ? "var(--green-light)" : conf >= 0.5 ? "var(--amber-light)" : "var(--surface-raised)",
        }}>
          {Math.round(conf * 100)}%
        </span>
        {/* Contact */}
        {lead.contact_name && (
          <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
            {lead.contact_name}
          </span>
        )}
        <ChevronRight size={13} style={{ color: "var(--text-muted)" }} />
      </div>
    </div>
  );
}

/* ── Provider badge ───────────────────────────────────────── */

function ProviderBadge({ name, status }: { name: string; status: string }) {
  const isOk = status === "available";
  const isDeg = status === "degraded";

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6, padding: "5px 10px",
      borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface-raised)",
      fontSize: 11,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: "50%", flexShrink: 0,
        background: isOk ? "var(--green)" : isDeg ? "var(--amber)" : "var(--red)",
      }} />
      <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
        {name.replace(/_/g, " ")}
      </span>
      <span style={{ color: isOk ? "var(--green)" : isDeg ? "var(--amber)" : "var(--red)", fontWeight: 600 }}>
        {status}
      </span>
    </div>
  );
}

/* ── Run history row ──────────────────────────────────────── */

function RunRow({ run }: { run: PipelineRunSummary }) {
  const isOk = run.status === "complete";
  const isFail = run.status === "error" || run.status === "failed";

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12, padding: "8px 14px",
      borderBottom: "1px solid var(--border)", fontSize: 12,
    }}>
      <div style={{ flexShrink: 0, color: isOk ? "var(--green)" : isFail ? "var(--red)" : "var(--text-muted)" }}>
        {isOk ? <CheckCircle size={13} /> : isFail ? <XCircle size={13} /> : <Clock size={13} />}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {run.current_step || run.status}
        </div>
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>
          {formatDate(run.started_at, "medium")}
          {run.elapsed_seconds > 0 && ` · ${Math.round(run.elapsed_seconds / 60)} min`}
        </div>
      </div>
      <div style={{ display: "flex", gap: 10, flexShrink: 0, fontSize: 11, color: "var(--text-secondary)" }}>
        {run.trends_detected > 0 && (
          <span><span className="num" style={{ fontWeight: 600 }}>{run.trends_detected}</span> trends</span>
        )}
        {run.leads_generated > 0 && (
          <span><span className="num" style={{ fontWeight: 600 }}>{run.leads_generated}</span> leads</span>
        )}
      </div>
    </div>
  );
}

/* ── Page ──────────────────────────────────────────────────── */

const PIPELINE_MODES = [
  { id: "industry_first", label: "Industry", icon: "⟳" },
  { id: "company_first", label: "Company", icon: "⬛" },
  { id: "report_driven", label: "Report", icon: "⊞" },
] as const;

export default function DashboardPage() {
  const router = useRouter();
  const { status, leads: contextLeads, trends, lastRunTime, initialLoading } = usePipelineContext();

  const [freshLeads, setFreshLeads] = useState<LeadRecord[]>([]);
  const [leadsLoading, setLeadsLoading] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [runs, setRuns] = useState<PipelineRunSummary[]>([]);
  const [runCount, setRunCount] = useState(0);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [startingPipeline, setStartingPipeline] = useState(false);
  const [selectedMode, setSelectedMode] = useState<string>("industry_first");

  /* ── Load on mount ── */

  useEffect(() => {
    loadHealth();
    loadRuns();
    loadFreshLeads();
  }, []);

  async function loadHealth() {
    try {
      const h = await api.health();
      setHealth(h);
    } catch { /* non-critical */ }
  }

  async function loadRuns() {
    try {
      const r = await api.getPipelineRuns(10);
      setRuns(r);
      setRunCount(r.length);
    } catch { /* non-critical */ }
  }

  const loadFreshLeads = useCallback(async () => {
    setLeadsLoading(true);
    try {
      const { leads } = await api.getLatestLeads(5);
      setFreshLeads(leads);
    } catch { /* keep context leads */ } finally {
      setLeadsLoading(false);
    }
  }, []);

  /* ── Run pipeline ── */

  async function handleRunPipeline() {
    setPipelineError(null);
    setStartingPipeline(true);
    try {
      const mockStatus = await api.getMockMode().catch(() => ({ enabled: false }));
      await api.runPipeline(mockStatus.enabled, undefined, undefined, { mode: selectedMode });
      // Pipeline context will pick up running status via its own polling/stream
    } catch (err) {
      setPipelineError(err instanceof Error ? err.message : "Failed to start pipeline");
    } finally {
      setStartingPipeline(false);
    }
  }

  /* ── Derived ── */

  const displayLeads = freshLeads.length > 0 ? freshLeads : contextLeads.slice(0, 5);
  const displayTrends = trends;
  const isRunning = status === "running" || startingPipeline;
  const companiesFound = new Set(displayLeads.map(l => l.company_name)).size;

  const providerEntries = health?.providers
    ? Object.entries(health.providers).filter(([, p]) => p.status !== "available").slice(0, 8)
    : [];
  const allProviderEntries = health?.providers ? Object.entries(health.providers) : [];

  return (
    <>
      {/* ── Page header ── */}
      <div style={{
        padding: "14px 24px",
        borderBottom: "1px solid var(--border)",
        background: "var(--surface)",
        flexShrink: 0,
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <h1 className="font-display" style={{ fontSize: 19, color: "var(--text)", letterSpacing: "-0.02em" }}>
          Dashboard
        </h1>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {lastRunTime && (
            <span style={{
              display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "var(--text-muted)",
              padding: "5px 10px", background: "var(--surface-raised)", borderRadius: 6,
              border: "1px solid var(--border)",
            }}>
              <Clock size={10} /> {lastRunTime}
            </span>
          )}
          <button
            onClick={loadFreshLeads}
            disabled={leadsLoading}
            style={{
              display: "flex", alignItems: "center", gap: 5,
              padding: "5px 12px", borderRadius: 7, fontSize: 12,
              border: "1px solid var(--border)", background: "var(--surface)",
              color: "var(--text-secondary)", cursor: leadsLoading ? "not-allowed" : "pointer",
              opacity: leadsLoading ? 0.6 : 1,
            }}
          >
            <RefreshCw size={11} style={{ animation: leadsLoading ? "spin 1s linear infinite" : "none" }} />
            Refresh
          </button>
          {/* Mode selector — determines which pipeline mode to run */}
          {!isRunning && (
            <div style={{ display: "flex", borderRadius: 8, border: "1px solid var(--border)", overflow: "hidden" }}>
              {PIPELINE_MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelectedMode(m.id)}
                  title={m.id.replace(/_/g, " ")}
                  style={{
                    padding: "5px 10px", fontSize: 11, fontWeight: 600,
                    border: "none", borderRight: "1px solid var(--border)", cursor: "pointer",
                    background: selectedMode === m.id ? "var(--accent)" : "var(--surface)",
                    color: selectedMode === m.id ? "#fff" : "var(--text-secondary)",
                    transition: "all 120ms",
                    whiteSpace: "nowrap",
                  }}
                >
                  {m.label}
                </button>
              ))}
            </div>
          )}
          <RunPipelineButton status={isRunning ? "running" : status} onClick={handleRunPipeline} />
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{ flex: 1, overflow: "auto", padding: "20px 24px", display: "flex", flexDirection: "column", gap: 18 }}>

        {/* Pipeline error */}
        {pipelineError && (
          <div style={{
            padding: "10px 14px", background: "var(--red-light)", color: "var(--red)",
            borderRadius: 8, fontSize: 12, display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <AlertCircle size={13} /> {pipelineError}
            </span>
            <button onClick={() => setPipelineError(null)} style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 16 }}>×</button>
          </div>
        )}

        {/* ── Running indicator ── */}
        {isRunning && (
          <div style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "12px 16px", borderRadius: 10,
            background: "var(--blue-light)", border: "1px solid var(--blue)",
          }}>
            <Loader2 size={14} style={{ color: "var(--blue)", animation: "spin 1s linear infinite", flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--blue)" }}>Pipeline is running</div>
              <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>
                Fetching news, detecting trends, and generating leads — this takes 20-40 minutes.
              </div>
            </div>
          </div>
        )}

        {/* ── KPI row ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
          <KpiCard
            label="Leads Generated"
            value={displayLeads.length > 0 ? contextLeads.length || displayLeads.length : 0}
            icon={<Users size={15} />}
            color="var(--green)"
            loading={initialLoading}
          />
          <KpiCard
            label="Trends Detected"
            value={displayTrends.length}
            icon={<TrendingUp size={15} />}
            color="var(--accent)"
            loading={initialLoading}
          />
          <KpiCard
            label="Companies Found"
            value={companiesFound}
            icon={<Building2 size={15} />}
            color="var(--blue)"
            loading={initialLoading}
          />
          <KpiCard
            label="Pipeline Runs"
            value={runCount}
            icon={<BarChart2 size={15} />}
            color="var(--text-secondary)"
            loading={false}
          />
        </div>

        {/* ── Two-column: Leads + Trends ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>

          {/* Recent leads */}
          <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{
              padding: "12px 14px", borderBottom: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0,
            }}>
              <h2 className="font-display" style={{ fontSize: 14, color: "var(--text)", letterSpacing: "-0.01em" }}>
                Recent Leads
              </h2>
              <button
                onClick={() => router.push("/leads")}
                style={{
                  fontSize: 11, color: "var(--accent)", background: "none", border: "none",
                  cursor: "pointer", display: "flex", alignItems: "center", gap: 3, fontWeight: 500,
                }}
              >
                View all <ArrowRight size={11} />
              </button>
            </div>

            <div style={{ flex: 1, overflow: "auto" }}>
              {initialLoading || leadsLoading ? (
                <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 8 }}>
                  {[0, 1, 2, 3, 4].map(i => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0" }}>
                      <div className="skeleton" style={{ width: 32, height: 32, borderRadius: 8, flexShrink: 0 }} />
                      <div style={{ flex: 1 }}>
                        <div className="skeleton" style={{ height: 13, width: "60%", marginBottom: 5 }} />
                        <div className="skeleton" style={{ height: 10, width: "80%" }} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : displayLeads.length === 0 ? (
                <div style={{
                  padding: "36px 20px", textAlign: "center",
                  color: "var(--text-muted)", fontSize: 12,
                }}>
                  <Users size={24} style={{ margin: "0 auto 10px", opacity: 0.3 }} />
                  <div style={{ fontWeight: 500, color: "var(--text-secondary)", marginBottom: 4 }}>No leads yet</div>
                  <div style={{ lineHeight: 1.6, maxWidth: 240, margin: "0 auto" }}>
                    Run the pipeline or create a campaign to generate leads.
                  </div>
                </div>
              ) : (
                displayLeads.map((lead, i) => (
                  <LeadRow
                    key={lead.id ?? i}
                    lead={lead}
                    onClick={() => router.push("/leads")}
                  />
                ))
              )}
            </div>
          </div>

          {/* Market trends */}
          <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{
              padding: "12px 14px", borderBottom: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0,
            }}>
              <h2 className="font-display" style={{ fontSize: 14, color: "var(--text)", letterSpacing: "-0.01em" }}>
                Market Trends
              </h2>
              <span style={{
                fontSize: 11, background: "var(--surface-raised)", color: "var(--text-muted)",
                padding: "2px 8px", borderRadius: 999, border: "1px solid var(--border)",
              }}>
                {displayTrends.length} signals
              </span>
            </div>

            <div style={{ flex: 1, overflow: "auto" }}>
              {initialLoading ? (
                <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 8 }}>
                  {[0, 1, 2, 3].map(i => (
                    <div key={i} style={{ padding: "10px 0", borderBottom: "1px solid var(--border)" }}>
                      <div className="skeleton" style={{ height: 13, width: "70%", marginBottom: 6 }} />
                      <div className="skeleton" style={{ height: 10, width: "90%" }} />
                    </div>
                  ))}
                </div>
              ) : displayTrends.length === 0 ? (
                <div style={{
                  padding: "36px 20px", textAlign: "center",
                  color: "var(--text-muted)", fontSize: 12,
                }}>
                  <Activity size={24} style={{ margin: "0 auto 10px", opacity: 0.3 }} />
                  <div style={{ fontWeight: 500, color: "var(--text-secondary)", marginBottom: 4 }}>No trends detected</div>
                  <div style={{ lineHeight: 1.6, maxWidth: 240, margin: "0 auto" }}>
                    Run the pipeline to detect market trends from live news.
                  </div>
                </div>
              ) : (
                displayTrends.slice(0, 6).map(trend => (
                  <div
                    key={trend.id}
                    style={{
                      padding: "10px 14px", borderBottom: "1px solid var(--border)",
                      cursor: "default",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 4 }}>
                      <span style={{
                        fontSize: 10, padding: "2px 7px", borderRadius: 999, flexShrink: 0, marginTop: 1,
                        fontWeight: 600,
                        background: trend.severity === "high" ? "var(--red-light)" :
                          trend.severity === "medium" ? "var(--amber-light)" : "var(--surface-raised)",
                        color: trend.severity === "high" ? "var(--red)" :
                          trend.severity === "medium" ? "var(--amber)" : "var(--text-muted)",
                      }}>
                        {trend.severity}
                      </span>
                      <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", lineHeight: 1.4 }}>
                        {trend.title}
                      </span>
                    </div>
                    {trend.summary && (
                      <div style={{
                        fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5,
                        overflow: "hidden", display: "-webkit-box",
                        WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
                      }}>
                        {trend.summary}
                      </div>
                    )}
                    {trend.industries?.length > 0 && (
                      <div style={{ display: "flex", gap: 4, marginTop: 5, flexWrap: "wrap" }}>
                        {trend.industries.slice(0, 3).map(ind => (
                          <span key={ind} style={{
                            fontSize: 10, padding: "1px 6px", borderRadius: 999,
                            background: "var(--surface-raised)", color: "var(--text-muted)",
                            border: "1px solid var(--border)",
                          }}>
                            {ind}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* ── Bottom row: System health + Run history ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>

          {/* System health */}
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{
              padding: "12px 14px", borderBottom: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "space-between",
            }}>
              <h2 className="font-display" style={{ fontSize: 14, color: "var(--text)", letterSpacing: "-0.01em" }}>
                System Health
              </h2>
              {health && (
                <span style={{
                  fontSize: 10, padding: "2px 8px", borderRadius: 999, fontWeight: 600,
                  background: health.status === "healthy" ? "var(--green-light)" : "var(--amber-light)",
                  color: health.status === "healthy" ? "var(--green)" : "var(--amber)",
                }}>
                  {health.status}
                </span>
              )}
            </div>

            <div style={{ padding: "12px 14px" }}>
              {!health ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {[0, 1, 2].map(i => (
                    <div key={i} className="skeleton" style={{ height: 30, borderRadius: 7 }} />
                  ))}
                </div>
              ) : allProviderEntries.length === 0 ? (
                <div style={{ fontSize: 12, color: "var(--text-muted)", padding: "8px 0" }}>No provider data</div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {allProviderEntries.slice(0, 8).map(([name, p]) => (
                    <ProviderBadge key={name} name={name} status={p.status} />
                  ))}
                  {health.config && (
                    <div style={{
                      marginTop: 8, padding: "8px 10px",
                      background: "var(--surface-raised)", borderRadius: 7,
                      border: "1px solid var(--border)",
                      fontSize: 11, color: "var(--text-secondary)",
                      display: "flex", gap: 12,
                    }}>
                      <span>Country: <strong>{health.config.country ?? "—"}</strong></span>
                      <span>Max trends: <strong>{health.config.max_trends ?? "—"}</strong></span>
                      {health.config.mock_mode && (
                        <span style={{ color: "var(--amber)", fontWeight: 600 }}>MOCK MODE</span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Recent runs */}
          <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{
              padding: "12px 14px", borderBottom: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0,
            }}>
              <h2 className="font-display" style={{ fontSize: 14, color: "var(--text)", letterSpacing: "-0.01em" }}>
                Pipeline Runs
              </h2>
              <span style={{
                fontSize: 11, background: "var(--surface-raised)", color: "var(--text-muted)",
                padding: "2px 8px", borderRadius: 999, border: "1px solid var(--border)",
              }}>
                {runCount} total
              </span>
            </div>

            <div style={{ flex: 1, overflow: "auto" }}>
              {runs.length === 0 ? (
                <div style={{
                  padding: "28px 20px", textAlign: "center",
                  color: "var(--text-muted)", fontSize: 12,
                }}>
                  <BarChart2 size={22} style={{ margin: "0 auto 8px", opacity: 0.3 }} />
                  <div>No pipeline runs yet</div>
                </div>
              ) : (
                runs.map(run => <RunRow key={run.run_id} run={run} />)
              )}
            </div>
          </div>
        </div>

        {/* ── Idle hint ── */}
        {displayLeads.length === 0 && status === "idle" && !initialLoading && (
          <div style={{
            display: "flex", alignItems: "center", gap: 12,
            padding: "14px 18px", borderRadius: 10,
            background: "var(--accent-light)", border: "1px solid var(--border)",
          }}>
            <Zap size={16} style={{ color: "var(--accent)", flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: 2 }}>
                Ready to get started
              </div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                Click <strong>Run Pipeline</strong> above to detect market trends and generate B2B leads automatically,
                or go to <strong>Campaigns</strong> to target specific companies.
              </div>
            </div>
            <button
              onClick={() => router.push("/campaigns")}
              style={{
                marginLeft: "auto", flexShrink: 0, display: "flex", alignItems: "center", gap: 5,
                fontSize: 12, fontWeight: 600, color: "var(--accent)",
                background: "none", border: "1px solid var(--accent)", borderRadius: 7,
                padding: "6px 12px", cursor: "pointer",
              }}
            >
              Campaigns <ArrowRight size={12} />
            </button>
          </div>
        )}
      </div>
    </>
  );
}
