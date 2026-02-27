"use client";

import { useState, useMemo, useEffect } from "react";
import { Search, SlidersHorizontal, X, ArrowDown, Users, AlertTriangle, Quote, Target } from "lucide-react";
import { TrendsFeed } from "@/components/dashboard/trends-feed";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import type { TrendData, Severity } from "@/lib/types";

const SEVERITY_OPTIONS: { label: string; value: Severity | "all" }[] = [
  { label: "All",        value: "all"       },
  { label: "High",       value: "high"      },
  { label: "Medium",     value: "medium"    },
  { label: "Low",        value: "low"       },
  { label: "Negligible", value: "negligible"},
];

const TYPE_OPTIONS = ["all", "regulatory", "policy", "macro", "technology", "industry", "talent"] as const;

export default function TrendsPage() {
  const { trends: contextTrends, initialLoading } = usePipelineContext();
  const [trends, setTrends] = useState<TrendData[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch]     = useState("");
  const [severity, setSeverity] = useState<Severity | "all">("all");
  const [type, setType]         = useState<string>("all");
  const [selected, setSelected] = useState<TrendData | null>(null);

  // Load from context, fall back to API
  useEffect(() => {
    if (contextTrends.length > 0) {
      setTrends(contextTrends);
      setLoading(false);
      return;
    }
    if (initialLoading) return; // wait for context to finish
    // Context empty after loading — try fetching from latest pipeline result
    (async () => {
      try {
        const runs = await api.getPipelineRuns(1);
        if (runs.length > 0 && runs[0].status === "completed") {
          const result = await api.getPipelineResult(runs[0].run_id);
          if (result.trends.length > 0) setTrends(result.trends);
        }
      } catch { /* no data */ }
      setLoading(false);
    })();
  }, [contextTrends, initialLoading]);

  const filtered = useMemo(() => {
    return trends.filter((t) => {
      if (severity !== "all" && t.severity !== severity) return false;
      if (type !== "all" && t.trend_type !== type) return false;
      if (search) {
        const q = search.toLowerCase();
        return (
          t.title.toLowerCase().includes(q) ||
          t.summary.toLowerCase().includes(q) ||
          t.industries.some((i) => i.toLowerCase().includes(q))
        );
      }
      return true;
    });
  }, [trends, search, severity, type]);

  return (
    <>
      {/* Header */}
      <div
        style={{
          padding: "16px 24px 14px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 14 }}>
          <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Market Trends
          </h1>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {filtered.length} of {trends.length} signals
          </span>
        </div>

        {/* Filter bar */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {/* Search */}
          <div style={{ position: "relative", flex: "1 1 200px", maxWidth: 320 }}>
            <Search
              size={13}
              style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}
            />
            <input
              type="text"
              placeholder="Search trends..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                width: "100%",
                padding: "7px 10px 7px 30px",
                borderRadius: 7,
                border: "1px solid var(--border)",
                background: "var(--surface)",
                fontSize: 12,
                color: "var(--text)",
                outline: "none",
              }}
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", display: "flex" }}
              >
                <X size={12} />
              </button>
            )}
          </div>

          {/* Severity tabs */}
          <div style={{ display: "flex", gap: 2, background: "var(--surface-raised)", borderRadius: 7, padding: 3 }}>
            {SEVERITY_OPTIONS.map(({ label, value }) => (
              <button
                key={value}
                onClick={() => setSeverity(value)}
                style={{
                  padding: "4px 10px",
                  borderRadius: 5,
                  border: "none",
                  fontSize: 11,
                  fontWeight: severity === value ? 600 : 400,
                  cursor: "pointer",
                  background: severity === value ? "var(--surface)" : "transparent",
                  color: severity === value ? "var(--text)" : "var(--text-muted)",
                  boxShadow: severity === value ? "var(--shadow-sm)" : "none",
                  transition: "all 150ms",
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Type filter */}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <SlidersHorizontal size={13} style={{ color: "var(--text-muted)" }} />
            <select
              value={type}
              onChange={(e) => setType(e.target.value)}
              style={{
                padding: "6px 10px",
                borderRadius: 7,
                border: "1px solid var(--border)",
                background: "var(--surface)",
                fontSize: 12,
                color: "var(--text-secondary)",
                cursor: "pointer",
                outline: "none",
              }}
            >
              {TYPE_OPTIONS.map((t) => (
                <option key={t} value={t}>{t === "all" ? "All types" : t.charAt(0).toUpperCase() + t.slice(1)}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex", minHeight: 0 }}>
        {/* List */}
        <div
          style={{
            flex: selected ? "0 0 420px" : "1 1 100%",
            overflow: "auto",
            padding: "14px 20px",
            borderRight: selected ? "1px solid var(--border)" : "none",
            transition: "flex 250ms ease",
          }}
        >
          {loading ? (
            <TrendsFeed trends={[]} loading />
          ) : filtered.length === 0 ? (
            <div style={{ padding: "40px 20px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
              {trends.length === 0
                ? "No trends yet — run the pipeline to detect market signals."
                : "No trends match your filters."}
            </div>
          ) : (
            <TrendsFeed
              trends={filtered}
              selectedId={selected?.id}
              onSelect={(t) => setSelected(selected?.id === t.id ? null : t)}
            />
          )}
        </div>

        {/* Detail pane */}
        {selected && (
          <div className="animate-slide-in-right" style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
            <TrendDetail trend={selected} onClose={() => setSelected(null)} />
          </div>
        )}
      </div>
    </>
  );
}

// ── Trend Detail Panel ─────────────────────────────────────────────────

/** Parse impact strings like "Segment (size, region) — EVIDENCE: quote" into parts */
function parseImpactEntry(entry: string): { segment: string; detail: string } {
  const dashIdx = entry.indexOf(" — ");
  if (dashIdx === -1) return { segment: entry, detail: "" };
  return { segment: entry.substring(0, dashIdx), detail: entry.substring(dashIdx + 3) };
}

function TrendDetail({ trend, onClose }: { trend: TrendData; onClose: () => void }) {
  const scoreColor = trend.trend_score >= 0.75 ? "var(--green)" : trend.trend_score >= 0.5 ? "var(--accent)" : "var(--text-muted)";
  const councilPct = trend.council_confidence > 0 ? Math.round(trend.council_confidence * 100) : null;

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16, gap: 12 }}>
        <h2 className="font-display" style={{ fontSize: 18, color: "var(--text)", lineHeight: 1.3, letterSpacing: "-0.02em" }}>
          {trend.title}
        </h2>
        <button
          onClick={onClose}
          style={{ flexShrink: 0, background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", display: "flex", padding: 4 }}
        >
          <X size={16} />
        </button>
      </div>

      {/* Score row */}
      <div style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
        {[
          { label: "Trend Score",       value: `${(trend.trend_score * 100).toFixed(0)}`, color: scoreColor },
          { label: "Actionability",     value: `${(trend.actionability_score * 100).toFixed(0)}`, color: "var(--blue)" },
          { label: "OSS Score",         value: trend.oss_score.toFixed(2), color: "var(--green)" },
          { label: "Articles",          value: String(trend.article_count), color: "var(--text-secondary)" },
          ...(councilPct !== null ? [{ label: "Council", value: `${councilPct}`, color: councilPct >= 50 ? "var(--green)" : "var(--amber)" }] : []),
        ].map(({ label, value, color }) => (
          <div key={label} style={{ padding: "8px 12px", background: "var(--surface-raised)", borderRadius: 8, minWidth: 70, textAlign: "center" }}>
            <div className="num" style={{ fontSize: 20, color, lineHeight: 1 }}>{value}</div>
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3, whiteSpace: "nowrap" }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Summary */}
      <DetailSection label="SUMMARY">
        <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65 }}>{trend.summary}</p>
      </DetailSection>

      {/* ── FIRST-ORDER IMPACT ── */}
      {trend.direct_impact?.length > 0 && (
        <DetailSection label="FIRST-ORDER IMPACT" icon={<Target size={11} style={{ color: "var(--red)" }} />}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {trend.direct_impact.map((entry, i) => {
              const { segment, detail } = parseImpactEntry(entry);
              return (
                <div key={i} style={{ padding: "8px 10px", borderLeft: "2px solid var(--red)", background: "var(--red-light)", borderRadius: "0 7px 7px 0" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", marginBottom: detail ? 3 : 0 }}>{segment}</div>
                  {detail && <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5 }}>{detail}</div>}
                </div>
              );
            })}
          </div>
        </DetailSection>
      )}

      {/* ── SECOND-ORDER IMPACT ── */}
      {trend.indirect_impact?.length > 0 && (
        <DetailSection label="SECOND-ORDER IMPACT" icon={<ArrowDown size={11} style={{ color: "var(--amber)" }} />}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {trend.indirect_impact.map((entry, i) => {
              const { segment, detail } = parseImpactEntry(entry);
              return (
                <div key={i} style={{ padding: "8px 10px", borderLeft: "2px solid var(--amber)", background: "var(--amber-light)", borderRadius: "0 7px 7px 0" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", marginBottom: detail ? 3 : 0 }}>{segment}</div>
                  {detail && <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5 }}>{detail}</div>}
                </div>
              );
            })}
          </div>
        </DetailSection>
      )}

      {/* ── PAIN POINTS ── */}
      {trend.midsize_pain_points?.length > 0 && (
        <DetailSection label="MID-SIZE COMPANY PAIN POINTS" icon={<AlertTriangle size={11} style={{ color: "var(--amber)" }} />}>
          <ul style={{ margin: 0, paddingLeft: 16, display: "flex", flexDirection: "column", gap: 4 }}>
            {trend.midsize_pain_points.map((pt, i) => (
              <li key={i} style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>{pt}</li>
            ))}
          </ul>
        </DetailSection>
      )}

      {/* ── PITCH ANGLE + WHO NEEDS HELP ── */}
      {(trend.pitch_angle || trend.who_needs_help) && (
        <DetailSection label="SALES ANGLE">
          {trend.who_needs_help && (
            <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: trend.pitch_angle ? 8 : 0 }}>
              <span style={{ fontWeight: 600, color: "var(--text-muted)", fontSize: 11 }}>WHO: </span>
              {trend.who_needs_help}
            </div>
          )}
          {trend.pitch_angle && (
            <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.6, fontStyle: "italic", padding: "10px 14px", background: "var(--accent-light)", borderRadius: 8, borderLeft: "2px solid var(--accent)" }}>
              {trend.pitch_angle}
            </div>
          )}
        </DetailSection>
      )}

      {/* ── TARGET ROLES ── */}
      {trend.target_roles?.length > 0 && (
        <DetailSection label="TARGET CONTACTS" icon={<Users size={11} style={{ color: "var(--blue)" }} />}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {trend.target_roles.map((role, i) => (
              <span key={i} className="badge badge-blue">{role}</span>
            ))}
          </div>
        </DetailSection>
      )}

      {/* Actionable insight */}
      {trend.actionable_insight && (
        <DetailSection label="ACTIONABLE INSIGHT">
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65, paddingLeft: 10, borderLeft: "2px solid var(--accent)", background: "var(--accent-light)", padding: "10px 12px", borderRadius: "0 7px 7px 0" }}>
            {trend.actionable_insight}
          </div>
        </DetailSection>
      )}

      {/* Causal chain */}
      {trend.causal_chain?.length > 0 && (
        <DetailSection label="CAUSAL CHAIN">
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {trend.causal_chain.map((step, i) => (
              <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                <span className="num" style={{ fontSize: 11, color: "var(--accent)", minWidth: 18, textAlign: "right" }}>{i + 1}.</span>
                <span style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>{step}</span>
              </div>
            ))}
          </div>
        </DetailSection>
      )}

      {/* ── EVIDENCE ── */}
      {trend.evidence_citations?.length > 0 && (
        <DetailSection label="EVIDENCE" icon={<Quote size={11} style={{ color: "var(--text-xmuted)" }} />}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {trend.evidence_citations.map((cite, i) => (
              <div key={i} style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5, paddingLeft: 10, borderLeft: "1px solid var(--border)" }}>
                {cite}
              </div>
            ))}
          </div>
        </DetailSection>
      )}

      {/* 5W1H */}
      {Object.keys(trend.event_5w1h || {}).length > 0 && (
        <DetailSection label="EVENT 5W1H">
          <div style={{ display: "grid", gridTemplateColumns: "80px 1fr", gap: "4px 12px" }}>
            {Object.entries(trend.event_5w1h).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", textTransform: "capitalize" }}>{k}</span>
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{v}</span>
              </div>
            ))}
          </div>
        </DetailSection>
      )}

      {/* Buying intent */}
      {Object.keys(trend.buying_intent || {}).length > 0 && (
        <DetailSection label="BUYING INTENT">
          <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: "4px 12px" }}>
            {Object.entries(trend.buying_intent).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", textTransform: "capitalize" }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{v}</span>
              </div>
            ))}
          </div>
        </DetailSection>
      )}

      {/* Industries + Keywords */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <DetailSection label="INDUSTRIES">
          <TagList items={trend.industries} />
        </DetailSection>
        {trend.keywords?.length > 0 && (
          <DetailSection label="KEYWORDS">
            <TagList items={trend.keywords} className="badge-muted" />
          </DetailSection>
        )}
      </div>

      {/* Companies */}
      {trend.affected_companies?.length > 0 && (
        <DetailSection label="COMPANIES MENTIONED">
          <TagList items={trend.affected_companies} className="badge-blue" />
        </DetailSection>
      )}
    </div>
  );
}

function DetailSection({ label, icon, children }: { label: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}>
        {icon}
        <span style={{ fontSize: 10, color: "var(--text-xmuted)", letterSpacing: "0.07em", fontWeight: 600 }}>{label}</span>
      </div>
      {children}
    </div>
  );
}

function TagList({ items, className = "badge-amber" }: { items: string[]; className?: string }) {
  const unique = [...new Set(items)];
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {unique.map((item, idx) => (
        <span key={`${item}-${idx}`} className={`badge ${className}`}>{item}</span>
      ))}
    </div>
  );
}
