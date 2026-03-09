"use client";

import { useState, useMemo, useEffect } from "react";
import Link from "next/link";
import { Search, SlidersHorizontal, X, ArrowDown, Users, AlertTriangle, Quote, Target, ExternalLink, Newspaper, Building2, ChevronRight, Star } from "lucide-react";
import { TrendsFeed } from "@/components/dashboard/trends-feed";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import { extractDomain, parseSnippet, confidenceColor, TYPE_CLASSES } from "@/lib/utils";
import { TrendSection } from "@/components/ui/detail-section";
import { BadgeTagList } from "@/components/ui/tag-list";
import type { TrendData, Severity, LeadRecord } from "@/lib/types";

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
          t.industries.some((i) => i.toLowerCase().includes(q)) ||
          t.keywords?.some((k) => k.toLowerCase().includes(q)) ||
          t.affected_companies?.some((c) => c.toLowerCase().includes(q)) ||
          t.actionable_insight?.toLowerCase().includes(q) ||
          t.pitch_angle?.toLowerCase().includes(q) ||
          t.who_needs_help?.toLowerCase().includes(q) ||
          t.causal_chain?.some((s) => s.toLowerCase().includes(q)) ||
          t.midsize_pain_points?.some((p) => p.toLowerCase().includes(q)) ||
          t.article_snippets?.some((s) => s.toLowerCase().includes(q)) ||
          t.target_roles?.some((r) => r.toLowerCase().includes(q)) ||
          Object.values(t.event_5w1h || {}).some((v) => v.toLowerCase().includes(q)) ||
          Object.values(t.buying_intent || {}).some((v) => v.toLowerCase().includes(q))
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
  const { leads: contextLeads } = usePipelineContext();
  const [apiLeads, setApiLeads] = useState<LeadRecord[]>([]);

  // Fetch from API when context is empty (e.g., page refresh without active run)
  useEffect(() => {
    if (contextLeads.length === 0) {
      api.getLeads({ limit: 200 }).then(({ leads }) => setApiLeads(leads)).catch(() => {});
    }
  }, [contextLeads.length]);

  const allLeads = contextLeads.length > 0 ? contextLeads : apiLeads;
  // Match leads by trend — exact match OR fuzzy (trend title contained in lead's trend_title or vice versa)
  const titleLower = trend.title.toLowerCase();
  const titleWords = titleLower.split(/\s+/).filter((w) => w.length > 4).slice(0, 5);
  const trendLeads = allLeads.filter((l) => {
    const lt = (l.trend_title || "").toLowerCase();
    // Exact match
    if (lt === titleLower) return true;
    // Contains match (either direction)
    if (lt.includes(titleLower) || titleLower.includes(lt)) return true;
    // Keyword overlap: if 3+ significant words from the title appear in the lead's trend_title
    if (titleWords.length >= 3) {
      const matches = titleWords.filter((w) => lt.includes(w));
      if (matches.length >= 3) return true;
    }
    return false;
  });
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
      <TrendSection label="SUMMARY">
        <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65 }}>{trend.summary}</p>
      </TrendSection>

      {/* ── FIRST-ORDER IMPACT ── */}
      {trend.direct_impact?.length > 0 && (
        <TrendSection label="FIRST-ORDER IMPACT" icon={<Target size={11} style={{ color: "var(--red)" }} />}>
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
        </TrendSection>
      )}

      {/* ── SECOND-ORDER IMPACT ── */}
      {trend.indirect_impact?.length > 0 && (
        <TrendSection label="SECOND-ORDER IMPACT" icon={<ArrowDown size={11} style={{ color: "var(--amber)" }} />}>
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
        </TrendSection>
      )}

      {/* ── COUNCIL ANALYSIS ── */}
      {trend.council_confidence != null && trend.council_confidence > 0 && (
        <TrendSection label="COUNCIL ANALYSIS">
          {/* Confidence meter */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.75rem" }}>
            <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", minWidth: 80 }}>Confidence</span>
            <div style={{ flex: 1, height: 8, background: "var(--surface-raised)", borderRadius: 4, overflow: "hidden" }}>
              <div style={{
                width: `${Math.round(trend.council_confidence * 100)}%`,
                height: "100%", borderRadius: 4,
                background: trend.council_confidence > 0.7 ? "#22c55e" : trend.council_confidence > 0.4 ? "#f59e0b" : "#ef4444"
              }} />
            </div>
            <span style={{ fontWeight: 600, color: "var(--text)", fontSize: "0.875rem" }}>
              {Math.round(trend.council_confidence * 100)}%
            </span>
          </div>
          {/* Pain points */}
          {trend.midsize_pain_points?.length > 0 && (
            <div style={{ marginBottom: "0.75rem" }}>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginBottom: "0.4rem" }}>Pain Points</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
                {trend.midsize_pain_points.map((p: string, i: number) => (
                  <span key={i} style={{ fontSize: "0.72rem", padding: "3px 8px", borderRadius: 4, background: "#ef444422", color: "#ef4444" }}>{p}</span>
                ))}
              </div>
            </div>
          )}
          {/* Target roles */}
          {trend.target_roles?.length > 0 && (
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginBottom: "0.4rem" }}>Decision-Makers to Target</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
                {trend.target_roles.map((r: string, i: number) => (
                  <span key={i} style={{ fontSize: "0.72rem", padding: "3px 8px", borderRadius: 4, background: "#3b82f622", color: "#3b82f6" }}>{r}</span>
                ))}
              </div>
            </div>
          )}
        </TrendSection>
      )}

      {/* ── BUYING INTENT SIGNALS ── */}
      {trend.buying_intent && Object.keys(trend.buying_intent).length > 0 && (
        <TrendSection label="BUYING INTENT SIGNALS">
          <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: "4px 12px" }}>
            {Object.entries(trend.buying_intent).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", textTransform: "capitalize" }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{v}</span>
              </div>
            ))}
          </div>
        </TrendSection>
      )}

      {/* ── EVIDENCE CITATIONS ── */}
      {trend.evidence_citations?.length > 0 && (
        <TrendSection label={`EVIDENCE (${trend.evidence_citations.length})`} icon={<Quote size={11} style={{ color: "var(--text-xmuted)" }} />}>
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {trend.evidence_citations.slice(0, 5).map((cite, i) => (
              <div key={i} style={{ fontSize: "0.78rem", color: "var(--text-muted)", padding: "0.35rem 0", borderBottom: "1px solid var(--border)" }}>
                • {cite}
              </div>
            ))}
          </div>
        </TrendSection>
      )}

      {/* ── PAIN POINTS ── */}
      {trend.midsize_pain_points?.length > 0 && (
        <TrendSection label="MID-SIZE COMPANY PAIN POINTS" icon={<AlertTriangle size={11} style={{ color: "var(--amber)" }} />}>
          <ul style={{ margin: 0, paddingLeft: 16, display: "flex", flexDirection: "column", gap: 4 }}>
            {trend.midsize_pain_points.map((pt, i) => (
              <li key={i} style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>{pt}</li>
            ))}
          </ul>
        </TrendSection>
      )}

      {/* ── PITCH ANGLE + WHO NEEDS HELP ── */}
      {(trend.pitch_angle || trend.who_needs_help) && (
        <TrendSection label="SALES ANGLE">
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
        </TrendSection>
      )}

      {/* ── TARGET ROLES ── */}
      {trend.target_roles?.length > 0 && (
        <TrendSection label="TARGET CONTACTS" icon={<Users size={11} style={{ color: "var(--blue)" }} />}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {trend.target_roles.map((role, i) => (
              <span key={i} className="badge badge-blue">{role}</span>
            ))}
          </div>
        </TrendSection>
      )}

      {/* Actionable insight */}
      {trend.actionable_insight && (
        <TrendSection label="ACTIONABLE INSIGHT">
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65, paddingLeft: 10, borderLeft: "2px solid var(--accent)", background: "var(--accent-light)", padding: "10px 12px", borderRadius: "0 7px 7px 0" }}>
            {trend.actionable_insight}
          </div>
        </TrendSection>
      )}

      {/* Causal chain */}
      {trend.causal_chain?.length > 0 && (
        <TrendSection label="CAUSAL CHAIN">
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {trend.causal_chain.map((step, i) => (
              <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                <span className="num" style={{ fontSize: 11, color: "var(--accent)", minWidth: 18, textAlign: "right" }}>{i + 1}.</span>
                <span style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>{step}</span>
              </div>
            ))}
          </div>
        </TrendSection>
      )}

      {/* ── EVIDENCE ── */}
      {trend.evidence_citations?.length > 0 && (
        <TrendSection label="EVIDENCE" icon={<Quote size={11} style={{ color: "var(--text-xmuted)" }} />}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {trend.evidence_citations.map((cite, i) => (
              <div key={i} style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5, paddingLeft: 10, borderLeft: "1px solid var(--border)" }}>
                {cite}
              </div>
            ))}
          </div>
        </TrendSection>
      )}

      {/* 5W1H */}
      {Object.keys(trend.event_5w1h || {}).length > 0 && (
        <TrendSection label="EVENT 5W1H">
          <div style={{ display: "grid", gridTemplateColumns: "80px 1fr", gap: "4px 12px" }}>
            {Object.entries(trend.event_5w1h).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", textTransform: "capitalize" }}>{k}</span>
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{v}</span>
              </div>
            ))}
          </div>
        </TrendSection>
      )}

      {/* Buying intent */}
      {Object.keys(trend.buying_intent || {}).length > 0 && (
        <TrendSection label="BUYING INTENT">
          <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: "4px 12px" }}>
            {Object.entries(trend.buying_intent).map(([k, v]) => (
              <div key={k} style={{ display: "contents" }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", textTransform: "capitalize" }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{v}</span>
              </div>
            ))}
          </div>
        </TrendSection>
      )}

      {/* Industries + Keywords */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <TrendSection label="INDUSTRIES">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {[...new Set(trend.industries)].map((ind) => (
              <Link
                key={ind}
                href={`/companies?industry=${encodeURIComponent(ind)}`}
                style={{ textDecoration: "none" }}
              >
                <span className="badge badge-amber" style={{ cursor: "pointer", transition: "opacity 150ms" }}
                  onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.8")}
                  onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
                >
                  {ind}
                </span>
              </Link>
            ))}
          </div>
        </TrendSection>
        {trend.keywords?.length > 0 && (
          <TrendSection label="KEYWORDS">
            <BadgeTagList items={trend.keywords} className="badge-muted" />
          </TrendSection>
        )}
      </div>

      {/* Companies — clickable, navigate to company detail */}
      {trend.affected_companies?.length > 0 && (
        <TrendSection label="COMPANIES MENTIONED" icon={<Building2 size={11} style={{ color: "var(--blue)" }} />}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {[...new Set(trend.affected_companies)].map((name) => (
              <Link
                key={name}
                href={`/companies/${encodeURIComponent(name)}`}
                style={{ textDecoration: "none" }}
              >
                <span
                  className="badge badge-blue"
                  style={{ cursor: "pointer", transition: "all 150ms", display: "inline-flex", alignItems: "center", gap: 3 }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = "var(--blue)"; e.currentTarget.style.color = "#fff"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = ""; e.currentTarget.style.color = ""; }}
                >
                  {name}
                  <ExternalLink size={8} style={{ opacity: 0.6 }} />
                </span>
              </Link>
            ))}
          </div>
        </TrendSection>
      )}

      {/* Leads generated from this trend */}
      <TrendSection label={`LEADS (${trendLeads.length})`} icon={<Star size={11} style={{ color: "var(--accent)" }} />}>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {trendLeads.slice(0, 10).map((lead, i) => {
              const cc = confidenceColor(lead.confidence);
              return (
                <Link
                  key={lead.id ?? i}
                  href={`/leads/${lead.id ?? i}`}
                  style={{ textDecoration: "none" }}
                >
                  <div
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "8px 10px", borderRadius: 7,
                      background: "var(--surface-raised)", border: "1px solid transparent",
                      transition: "border-color 150ms, background 150ms", cursor: "pointer",
                    }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)"; }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "transparent"; (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
                  >
                    <div style={{ width: 28, height: 28, borderRadius: 6, background: cc.bg, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <span className="num" style={{ fontSize: 11, fontWeight: 700, color: cc.text }}>{Math.round(lead.confidence * 100)}</span>
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {lead.company_name}
                      </div>
                      <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                        {lead.contact_name || lead.people?.[0]?.person_name || lead.lead_type} · H{lead.hop}
                      </div>
                    </div>
                    <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`} style={{ fontSize: 9 }}>{lead.lead_type}</span>
                    <ChevronRight size={11} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
                  </div>
                </Link>
              );
            })}
            {trendLeads.length > 10 && (
              <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", padding: "4px 0" }}>
                Showing 10 of {trendLeads.length}
              </div>
            )}
            <Link
              href={`/leads?trend=${encodeURIComponent(trend.title)}`}
              style={{
                display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                padding: "8px 14px", marginTop: 6, borderRadius: 7,
                border: "1px solid var(--accent)33", background: "var(--accent-light)",
                fontSize: 11, fontWeight: 600, color: "var(--accent)", textDecoration: "none",
              }}
            >
              <Target size={11} /> View All Leads for this Trend
              <ChevronRight size={11} />
            </Link>
          </div>
        </TrendSection>

      {/* Source Articles */}
      <SourceArticlesSection snippets={trend.article_snippets} links={trend.source_links} />
    </div>
  );
}


/** Shared source articles section — used in trends detail and lead detail */
function SourceArticlesSection({ snippets, links }: { snippets?: string[]; links?: string[] }) {
  const hasSnippets = snippets && snippets.length > 0;
  const hasLinks = links && links.length > 0;
  if (!hasSnippets && !hasLinks) return null;

  const count = Math.max(snippets?.length ?? 0, links?.length ?? 0);

  return (
    <TrendSection
      label={`SOURCE ARTICLES (${count})`}
      icon={<Newspaper size={11} style={{ color: "var(--blue)" }} />}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {snippets?.map((snippet, i) => {
          const { title, body } = parseSnippet(snippet);
          const link = links?.[i];
          const domain = link ? extractDomain(link) : null;

          return (
            <a
              key={i}
              href={link || "#"}
              target={link ? "_blank" : undefined}
              rel={link ? "noopener noreferrer" : undefined}
              style={{
                display: "block",
                padding: "10px 12px",
                background: "var(--surface-raised)",
                borderRadius: 8,
                borderLeft: "3px solid var(--blue)",
                textDecoration: "none",
                transition: "background 150ms, box-shadow 150ms",
                cursor: link ? "pointer" : "default",
              }}
              onMouseEnter={(e) => {
                if (link) {
                  (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)";
                  (e.currentTarget as HTMLElement).style.boxShadow = "var(--shadow-sm)";
                }
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)";
                (e.currentTarget as HTMLElement).style.boxShadow = "none";
              }}
            >
              {/* Domain + external link icon */}
              {domain && (
                <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 4 }}>
                  <span style={{ fontSize: 10, color: "var(--blue)", fontWeight: 500 }}>{domain}</span>
                  <ExternalLink size={9} style={{ color: "var(--blue)", opacity: 0.6 }} />
                </div>
              )}

              {/* Title */}
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", lineHeight: 1.4, marginBottom: body ? 4 : 0 }}>
                {title}
              </div>

              {/* Body excerpt */}
              {body && (
                <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                  {body.length > 250 ? body.substring(0, 250) + "..." : body}
                </div>
              )}
            </a>
          );
        })}

        {/* Orphan links (more links than snippets) */}
        {links?.slice(snippets?.length ?? 0).map((link, i) => {
          const domain = extractDomain(link);
          return (
            <a
              key={`link-${i}`}
              href={link}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "8px 12px",
                borderRadius: 8,
                background: "var(--surface-raised)",
                borderLeft: "3px solid var(--blue)",
                textDecoration: "none",
                fontSize: 12,
                color: "var(--blue)",
                transition: "background 150ms",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
            >
              <ExternalLink size={11} />
              <span style={{ fontWeight: 500 }}>{domain}</span>
              <span style={{ color: "var(--text-xmuted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                {link}
              </span>
            </a>
          );
        })}
      </div>
    </TrendSection>
  );
}
