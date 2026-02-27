"use client";

import { useEffect, useState, useCallback } from "react";
import { use } from "react";
import Link from "next/link";
import {
  ReactFlow, Node, Edge, Background, Controls, MiniMap,
  Handle, Position, useNodesState, useEdgesState,
  MarkerType, BackgroundVariant,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import Dagre from "@dagrejs/dagre";
import {
  ArrowLeft, CheckCircle, XCircle, Loader2, Clock,
  TrendingUp, Users, Building2, AlertCircle, Layers,
  Zap, X, ChevronRight, ExternalLink, Mail, User,
} from "lucide-react";
import { api } from "@/lib/api";
import type { PipelineRunResult, TrendData, LeadRecord } from "@/lib/types";

// ── Node data types ────────────────────────────────────────────────────

type RunNodeData   = { label: string; status: string; duration: string; trends: number; leads: number; companies: number };
type TrendNodeData = { title: string; severity: string; articleCount: number; industries: string[]; oss: number; index: number };
type LeadNodeData  = { company: string; leadType: string; confidence: number; role: string; leadId?: number; leadIdx: number };
type PlusNodeData  = { count: number };

type AppNode =
  | Node<RunNodeData,   "run">
  | Node<TrendNodeData, "trend">
  | Node<LeadNodeData,  "lead">
  | Node<PlusNodeData,  "plus">;

// ── Colour maps ────────────────────────────────────────────────────────

const SEVERITY_COLOR: Record<string, string> = {
  high:       "var(--red)",
  medium:     "var(--amber)",
  low:        "var(--green)",
  negligible: "var(--text-muted)",
};

const TYPE_COLOR: Record<string, string> = {
  pain:         "var(--red)",
  opportunity:  "var(--green)",
  risk:         "var(--amber)",
  intelligence: "var(--blue)",
};

const TYPE_BG: Record<string, string> = {
  pain:         "var(--red-light)",
  opportunity:  "var(--green-light)",
  risk:         "var(--amber-light)",
  intelligence: "var(--blue-light)",
};

// ── Helpers ────────────────────────────────────────────────────────────

function formatDuration(s: number) {
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

// ── Node sizes ────────────────────────────────────────────────────────

const NODE_DIM: Record<string, { w: number; h: number }> = {
  run:   { w: 310, h: 115 },
  trend: { w: 245, h: 105 },
  lead:  { w: 205, h: 85  },
  plus:  { w: 120, h: 44  },
};

// ── Dagre layout ──────────────────────────────────────────────────────

function applyLayout(nodes: AppNode[], edges: Edge[]): AppNode[] {
  const g = new Dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", ranksep: 95, nodesep: 45, marginx: 55, marginy: 55 });
  nodes.forEach((n) => {
    const dim = NODE_DIM[n.type ?? "trend"] ?? { w: 200, h: 80 };
    g.setNode(n.id, { width: dim.w, height: dim.h });
  });
  edges.forEach((e) => g.setEdge(e.source, e.target));
  Dagre.layout(g);
  return nodes.map((n) => {
    const { x, y } = g.node(n.id);
    const dim = NODE_DIM[n.type ?? "trend"] ?? { w: 200, h: 80 };
    return { ...n, position: { x: x - dim.w / 2, y: y - dim.h / 2 } };
  });
}

// ── Build graph ────────────────────────────────────────────────────────

const MAX_LEADS_PER_TREND = 5;
const MAX_TRENDS          = 18;

function buildGraph(result: PipelineRunResult): { nodes: AppNode[]; edges: Edge[] } {
  const nodes: AppNode[] = [];
  const edges: Edge[]    = [];

  nodes.push({
    id: "run", type: "run", position: { x: 0, y: 0 },
    data: {
      label:     result.run_id,
      status:    result.status,
      duration:  formatDuration(result.run_time_seconds),
      trends:    result.trends_detected,
      leads:     result.leads_generated,
      companies: result.companies_found,
    },
  });

  const shownTrends  = result.trends.slice(0, MAX_TRENDS);
  const hiddenTrends = result.trends.length - shownTrends.length;

  shownTrends.forEach((t, i) => {
    const tid = `trend-${i}`;
    nodes.push({
      id: tid, type: "trend", position: { x: 0, y: 0 },
      data: {
        title:        t.title,
        severity:     t.severity,
        articleCount: t.article_count,
        industries:   t.industries ?? [],
        oss:          t.oss_score,
        index:        i,
      },
    });
    edges.push({
      id: `e-run-${tid}`, source: "run", target: tid,
      type: "smoothstep",
      style: { stroke: "var(--border-strong)", strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "var(--border-strong)", width: 12, height: 12 },
    });

    const trendLeads = result.leads.filter((l) => l.trend_title === t.title);
    const shown      = trendLeads.slice(0, MAX_LEADS_PER_TREND);
    const overflow   = trendLeads.length - shown.length;

    shown.forEach((l, li) => {
      const lid   = `lead-${tid}-${li}`;
      const color = TYPE_COLOR[l.lead_type] ?? "var(--text-muted)";
      nodes.push({
        id: lid, type: "lead", position: { x: 0, y: 0 },
        data: {
          company:    l.company_name,
          leadType:   l.lead_type,
          confidence: l.confidence,
          role:       l.contact_role || "",
          leadId:     l.id,
          leadIdx:    result.leads.indexOf(l),
        },
      });
      edges.push({
        id: `e-${tid}-${lid}`, source: tid, target: lid,
        type: "smoothstep",
        style: { stroke: color + "66", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 10, height: 10 },
      });
    });

    if (overflow > 0) {
      const pid = `plus-${tid}`;
      nodes.push({ id: pid, type: "plus", position: { x: 0, y: 0 }, data: { count: overflow } });
      edges.push({
        id: `e-${tid}-${pid}`, source: tid, target: pid,
        type: "smoothstep",
        style: { stroke: "var(--border)", strokeWidth: 1, strokeDasharray: "4 2" },
      });
    }
  });

  if (hiddenTrends > 0) {
    nodes.push({ id: "plus-trends", type: "plus", position: { x: 0, y: 0 }, data: { count: hiddenTrends } });
    edges.push({
      id: "e-run-plus-trends", source: "run", target: "plus-trends",
      type: "smoothstep",
      style: { stroke: "var(--border)", strokeWidth: 1, strokeDasharray: "4 2" },
    });
  }

  return { nodes: applyLayout(nodes as AppNode[], edges), edges };
}

// ── Shared sub-components ──────────────────────────────────────────────

function NodeStat({ icon: Icon, value, label, color }: {
  icon: React.ElementType; value: string | number; label: string; color: string;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
      <Icon size={11} style={{ color }} />
      <span style={{ fontSize: 13, fontWeight: 700, color, lineHeight: 1 }}>{value}</span>
      <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>{label}</span>
    </div>
  );
}

// ── Custom Node: Run ──────────────────────────────────────────────────

function RunNode({ data, selected }: NodeProps & { data: RunNodeData }) {
  const [hov, setHov] = useState(false);
  const sc = data.status === "completed" ? "var(--green)" : data.status === "failed" ? "var(--red)" : "var(--amber)";
  return (
    <div
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        width: NODE_DIM.run.w, background: "var(--surface)",
        border: `2px solid ${selected ? sc : hov ? sc + "99" : sc + "55"}`,
        borderRadius: 14,
        boxShadow: selected ? `0 0 0 4px ${sc}22, var(--shadow-md)` : hov ? "var(--shadow-md)" : "var(--shadow-sm)",
        overflow: "hidden", transition: "all 180ms ease",
      }}
    >
      <div style={{ padding: "10px 14px", background: sc + "18", borderBottom: `1px solid ${sc}22`, display: "flex", alignItems: "center", gap: 8 }}>
        {data.status === "completed" && <CheckCircle size={14} style={{ color: sc }} />}
        {data.status === "failed"    && <XCircle     size={14} style={{ color: sc }} />}
        {data.status === "running"   && <Loader2     size={14} style={{ color: sc, animation: "spin 1s linear infinite" }} />}
        <span style={{ fontSize: 11, fontWeight: 800, color: "var(--text)", fontFamily: "monospace" }}>{data.label}</span>
        <span style={{ marginLeft: "auto", fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 99, background: sc + "22", color: sc, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          {data.status}
        </span>
      </div>
      <div style={{ padding: "12px 14px", display: "flex", gap: 10, justifyContent: "space-between" }}>
        <NodeStat icon={TrendingUp} value={data.trends}    label="trends"    color="var(--blue)"       />
        <NodeStat icon={Building2}  value={data.companies} label="companies" color="var(--green)"      />
        <NodeStat icon={Users}      value={data.leads}     label="leads"     color="var(--accent)"     />
        <NodeStat icon={Clock}      value={data.duration}  label="runtime"   color="var(--text-muted)" />
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: sc, width: 10, height: 10, border: "2px solid var(--surface)" }} />
    </div>
  );
}

// ── Custom Node: Trend ────────────────────────────────────────────────

function TrendNode({ data, selected }: NodeProps & { data: TrendNodeData }) {
  const [hov, setHov] = useState(false);
  const sc  = SEVERITY_COLOR[data.severity] ?? "var(--text-muted)";
  const ind = data.industries.slice(0, 2).join(" · ");
  return (
    <div
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        width: NODE_DIM.trend.w, background: "var(--surface)",
        border: selected ? `2px solid ${sc}` : `1px solid ${hov ? sc + "77" : "var(--border)"}`,
        borderTop: `3px solid ${sc}`, borderRadius: 12,
        boxShadow: selected ? `0 0 0 4px ${sc}1A, var(--shadow-md)` : hov ? "var(--shadow-md)" : "var(--shadow-sm)",
        overflow: "hidden", cursor: "pointer",
        transform: hov && !selected ? "translateY(-2px)" : "none",
        transition: "all 180ms ease",
      }}
    >
      <Handle type="target" position={Position.Top}    style={{ background: sc, width: 7, height: 7, border: "2px solid var(--surface)" }} />
      <div style={{ padding: "11px 13px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 7, marginBottom: 7 }}>
          <TrendingUp size={12} style={{ color: sc, flexShrink: 0, marginTop: 1 }} />
          <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", lineHeight: 1.4, flex: 1 }}>
            {data.title.length > 68 ? data.title.slice(0, 66) + "…" : data.title}
          </div>
        </div>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 99, background: sc + "22", color: sc, textTransform: "uppercase", letterSpacing: "0.06em" }}>
            {data.severity}
          </span>
          {data.articleCount > 0 && (
            <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>{data.articleCount} art.</span>
          )}
          {ind && (
            <span style={{ fontSize: 9, color: "var(--text-xmuted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 95 }}>
              {ind}
            </span>
          )}
          {(hov || selected) && (
            <span style={{ marginLeft: "auto", fontSize: 9, color: sc, display: "flex", alignItems: "center", gap: 2, fontWeight: 600 }}>
              <ChevronRight size={9} />details
            </span>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: sc, width: 7, height: 7, border: "2px solid var(--surface)" }} />
    </div>
  );
}

// ── Custom Node: Lead ─────────────────────────────────────────────────

function LeadNode({ data, selected }: NodeProps & { data: LeadNodeData }) {
  const [hov, setHov] = useState(false);
  const color = TYPE_COLOR[data.leadType] ?? "var(--text-muted)";
  const bg    = TYPE_BG[data.leadType]   ?? "var(--surface-raised)";
  const conf  = Math.round(data.confidence * 100);
  return (
    <div
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        width: NODE_DIM.lead.w, background: "var(--surface)",
        border: selected ? `2px solid ${color}` : `1px solid ${hov ? color + "99" : color + "44"}`,
        borderRadius: 10,
        boxShadow: selected ? `0 0 0 4px ${color}1A, var(--shadow-md)` : hov ? "var(--shadow-sm)" : "var(--shadow-xs)",
        overflow: "hidden", cursor: "pointer",
        transform: hov && !selected ? "translateY(-2px)" : "none",
        transition: "all 180ms ease",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: color, width: 6, height: 6, border: "2px solid var(--surface)" }} />
      {/* Confidence fill bar */}
      <div style={{ height: 2, background: `linear-gradient(90deg, ${color}, ${color}44)`, width: `${conf}%` }} />
      <div style={{ padding: "9px 11px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6, marginBottom: 5 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
            {data.company}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
            <span style={{ fontSize: 14, fontWeight: 800, color, lineHeight: 1 }}>{conf}</span>
            {(hov || selected) && <ExternalLink size={10} style={{ color }} />}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 99, background: bg, color, textTransform: "uppercase", letterSpacing: "0.04em", flexShrink: 0 }}>
            {data.leadType}
          </span>
          {data.role && (
            <span style={{ fontSize: 9, color: "var(--text-xmuted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
              {data.role}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Custom Node: Plus ─────────────────────────────────────────────────

function PlusNode({ data }: NodeProps & { data: PlusNodeData }) {
  return (
    <div style={{
      width: NODE_DIM.plus.w, height: NODE_DIM.plus.h,
      background: "var(--surface-raised)", border: "1px dashed var(--border-strong)",
      borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 11, color: "var(--text-muted)", fontWeight: 600,
    }}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      +{data.count} more
    </div>
  );
}

const NODE_TYPES = { run: RunNode, trend: TrendNode, lead: LeadNode, plus: PlusNode } as const;

// ── Detail panel shared pieces ─────────────────────────────────────────

function ScoreRow({ label, value, color }: { label: string; value: number | null | undefined; color: string }) {
  const pct = value != null ? Math.min(100, Math.max(0, Math.round(value * 100))) : 0;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{label}</span>
        <span style={{ fontSize: 10, fontWeight: 700, color }}>{value != null ? pct : "—"}</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: "var(--surface-raised)", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", borderRadius: 2, background: color, transition: "width 500ms ease" }} />
      </div>
    </div>
  );
}

function PanelSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.09em", color: "var(--text-xmuted)", textTransform: "uppercase", marginBottom: 9, display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
        {title}
        <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
      </div>
      {children}
    </div>
  );
}

// ── Trend detail panel ─────────────────────────────────────────────────

function TrendPanel({ trend, allLeads, onClose }: { trend: TrendData; allLeads: LeadRecord[]; onClose: () => void }) {
  const sc         = SEVERITY_COLOR[trend.severity] ?? "var(--text-muted)";
  const trendLeads = allLeads.filter((l) => l.trend_title === trend.title);

  return (
    <div style={{
      position: "absolute", top: 0, right: 0, bottom: 0, width: 390,
      background: "var(--surface)", borderLeft: "1px solid var(--border)",
      boxShadow: "var(--shadow-lg)", zIndex: 20, display: "flex", flexDirection: "column",
      animation: "slide-in-right 220ms ease forwards",
    }}>
      {/* Sticky header */}
      <div style={{ padding: "15px 18px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 7 }}>
              <TrendingUp size={12} style={{ color: sc }} />
              <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 8px", borderRadius: 99, background: sc + "22", color: sc, textTransform: "uppercase", letterSpacing: "0.07em" }}>
                {trend.severity} severity
              </span>
              <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>{trend.article_count} articles</span>
            </div>
            <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text)", lineHeight: 1.4 }}>{trend.title}</div>
          </div>
          <button onClick={onClose} style={{ padding: "6px 7px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--bg)", cursor: "pointer", color: "var(--text-muted)", display: "flex", flexShrink: 0, marginTop: 1 }}>
            <X size={13} />
          </button>
        </div>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 18px" }}>

        {/* Summary */}
        {trend.summary && (
          <PanelSection title="Summary">
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0 }}>{trend.summary}</p>
          </PanelSection>
        )}

        {/* Scores */}
        <PanelSection title="Signal Scores">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <ScoreRow label="OSS Score"         value={trend.oss_score}           color="var(--accent)" />
            <ScoreRow label="Trend Score"       value={trend.trend_score}         color="var(--blue)"   />
            <ScoreRow label="Actionability"     value={trend.actionability_score} color="var(--green)"  />
            <ScoreRow label="Council Confid."   value={trend.council_confidence}  color={sc}            />
          </div>
        </PanelSection>

        {/* Industries */}
        {trend.industries?.length > 0 && (
          <PanelSection title="Industries">
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {trend.industries.map((ind) => (
                <span key={ind} style={{ fontSize: 10, padding: "3px 9px", borderRadius: 99, background: "var(--surface-raised)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}>
                  {ind}
                </span>
              ))}
            </div>
          </PanelSection>
        )}

        {/* Causal chain */}
        {trend.causal_chain?.length > 0 && (
          <PanelSection title="Causal Chain">
            <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
              {trend.causal_chain.slice(0, 5).map((step, i) => (
                <div key={i} style={{ display: "flex", gap: 9, alignItems: "flex-start" }}>
                  <div style={{ width: 18, height: 18, borderRadius: "50%", background: sc + "22", border: `1px solid ${sc}55`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontSize: 9, fontWeight: 700, color: sc, marginTop: 1 }}>
                    {i + 1}
                  </div>
                  <p style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.55, margin: 0 }}>{step}</p>
                </div>
              ))}
            </div>
          </PanelSection>
        )}

        {/* Actionable insight */}
        {trend.actionable_insight && (
          <PanelSection title="Actionable Insight">
            <p style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6, margin: 0, padding: "10px 13px", background: sc + "0C", borderLeft: `3px solid ${sc}`, borderRadius: "0 7px 7px 0" }}>
              {trend.actionable_insight}
            </p>
          </PanelSection>
        )}

        {/* Leads */}
        {trendLeads.length > 0 && (
          <PanelSection title={`Leads from this Trend (${trendLeads.length})`}>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {trendLeads.slice(0, 6).map((l, i) => {
                const c    = TYPE_COLOR[l.lead_type] ?? "var(--text-muted)";
                const conf = Math.round(l.confidence * 100);
                return (
                  <Link key={i} href={`/leads/${l.id ?? i}`} style={{ textDecoration: "none" }}>
                    <div
                      style={{ padding: "9px 12px", borderRadius: 8, border: `1px solid ${c}33`, background: "var(--bg)", transition: "all 140ms", display: "flex", alignItems: "center", gap: 8 }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = c + "0D"; e.currentTarget.style.borderColor = c + "77"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = "var(--bg)"; e.currentTarget.style.borderColor = c + "33"; }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{l.company_name}</div>
                        {l.contact_role && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>{l.contact_role}</div>}
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 5, flexShrink: 0 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: c }}>{conf}</span>
                        <ChevronRight size={11} style={{ color: "var(--text-xmuted)" }} />
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </PanelSection>
        )}
      </div>
    </div>
  );
}

// ── Lead detail panel ──────────────────────────────────────────────────

function LeadPanel({ lead, leadIdx, onClose }: { lead: LeadRecord; leadIdx: number; onClose: () => void }) {
  const color = TYPE_COLOR[lead.lead_type] ?? "var(--text-muted)";
  const bg    = TYPE_BG[lead.lead_type]   ?? "var(--surface-raised)";
  const conf  = Math.round(lead.confidence * 100);
  const href  = `/leads/${lead.id ?? leadIdx}`;

  return (
    <div style={{
      position: "absolute", top: 0, right: 0, bottom: 0, width: 390,
      background: "var(--surface)", borderLeft: "1px solid var(--border)",
      boxShadow: "var(--shadow-lg)", zIndex: 20, display: "flex", flexDirection: "column",
      animation: "slide-in-right 220ms ease forwards",
    }}>
      {/* Sticky header */}
      <div style={{ padding: "15px 18px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 8px", borderRadius: 99, background: bg, color, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {lead.lead_type}
              </span>
              <span style={{ fontSize: 14, fontWeight: 800, color }}>{conf}</span>
              <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>confidence</span>
            </div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)" }}>{lead.company_name}</div>
            {(lead.company_city || lead.company_state) && (
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                {[lead.company_city, lead.company_state].filter(Boolean).join(", ")}
              </div>
            )}
          </div>
          <button onClick={onClose} style={{ padding: "6px 7px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--bg)", cursor: "pointer", color: "var(--text-muted)", display: "flex", flexShrink: 0, marginTop: 1 }}>
            <X size={13} />
          </button>
        </div>
        {/* Confidence bar */}
        <div style={{ marginTop: 10, height: 4, borderRadius: 2, background: "var(--surface-raised)", overflow: "hidden" }}>
          <div style={{ width: `${conf}%`, height: "100%", background: color, borderRadius: 2, transition: "width 500ms ease" }} />
        </div>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 18px" }}>

        {/* Contact */}
        {(lead.contact_name || lead.contact_role || lead.contact_email) && (
          <PanelSection title="Contact">
            <div style={{ padding: "11px 13px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 9, display: "flex", flexDirection: "column", gap: 7 }}>
              {lead.contact_name && (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <User size={11} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{lead.contact_name}</span>
                </div>
              )}
              {lead.contact_role && (
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: 19 }}>{lead.contact_role}</div>
              )}
              {lead.contact_email && (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Mail size={11} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                  <a href={`mailto:${lead.contact_email}`} style={{ fontSize: 11, color: "var(--blue)", textDecoration: "none" }}>{lead.contact_email}</a>
                </div>
              )}
            </div>
          </PanelSection>
        )}

        {/* Opening line */}
        {lead.opening_line && (
          <PanelSection title="Opening Line">
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0, padding: "11px 13px", background: color + "0C", borderLeft: `3px solid ${color}`, borderRadius: "0 7px 7px 0", fontStyle: "italic" }}>
              &ldquo;{lead.opening_line}&rdquo;
            </p>
          </PanelSection>
        )}

        {/* Pain point */}
        {lead.pain_point && (
          <PanelSection title="Pain Point">
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0 }}>{lead.pain_point}</p>
          </PanelSection>
        )}

        {/* Trigger event */}
        {lead.trigger_event && (
          <PanelSection title="Trigger Event">
            <div style={{ display: "flex", gap: 9, alignItems: "flex-start", padding: "9px 12px", background: "var(--amber-light)", borderRadius: 7, border: "1px solid var(--amber)33" }}>
              <Zap size={11} style={{ color: "var(--amber)", flexShrink: 0, marginTop: 2 }} />
              <p style={{ fontSize: 11, color: "var(--text-secondary)", margin: 0, lineHeight: 1.55 }}>{lead.trigger_event}</p>
            </div>
          </PanelSection>
        )}

        {/* Email subject preview */}
        {lead.email_subject && (
          <PanelSection title="Email Subject">
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", padding: "9px 12px", background: "var(--bg)", borderRadius: 7, border: "1px solid var(--border)" }}>
              {lead.email_subject}
            </div>
          </PanelSection>
        )}
      </div>

      {/* CTA footer */}
      <div style={{ padding: "12px 18px", borderTop: "1px solid var(--border)", flexShrink: 0 }}>
        <Link href={href} style={{ textDecoration: "none", display: "block" }}>
          <div
            style={{ padding: "12px 16px", borderRadius: 9, background: color, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, fontSize: 13, fontWeight: 700, cursor: "pointer", transition: "opacity 150ms" }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.88")}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
          >
            <ExternalLink size={14} /> View Full Lead Profile
          </div>
        </Link>
      </div>
    </div>
  );
}

// ── Legend ────────────────────────────────────────────────────────────

function Legend() {
  return (
    <div style={{ position: "absolute", bottom: 16, left: 16, zIndex: 10, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px", boxShadow: "var(--shadow-md)" }}>
      <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.07em", marginBottom: 9 }}>LEGEND</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {[
          { color: "var(--red)",   label: "High severity trend"  },
          { color: "var(--amber)", label: "Medium severity trend" },
          { color: "var(--green)", label: "Low / Opportunity"     },
          { color: "var(--red)",   label: "Pain lead"             },
          { color: "var(--blue)",  label: "Intelligence lead"     },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{ width: 9, height: 9, borderRadius: "50%", background: color, flexShrink: 0 }} />
            <span style={{ fontSize: 10, color: "var(--text-secondary)" }}>{label}</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid var(--border)", fontSize: 9, color: "var(--text-xmuted)" }}>
        Click any node to inspect
      </div>
    </div>
  );
}

// ── Stat bar ──────────────────────────────────────────────────────────

function StatBar({ result }: { result: PipelineRunResult }) {
  return (
    <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
      {[
        { icon: TrendingUp, v: result.trends_detected,                   label: "trends",    c: "var(--blue)"      },
        { icon: Building2,  v: result.companies_found,                   label: "companies", c: "var(--accent)"    },
        { icon: Users,      v: result.leads_generated,                   label: "leads",     c: "var(--green)"     },
        { icon: Clock,      v: formatDuration(result.run_time_seconds),  label: "runtime",   c: "var(--text-muted)" },
      ].map(({ icon: Icon, v, label, c }) => (
        <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <Icon size={12} style={{ color: c }} />
          <span style={{ fontSize: 14, fontWeight: 700, color: c }}>{v}</span>
          <span style={{ fontSize: 11, color: "var(--text-xmuted)" }}>{label}</span>
        </div>
      ))}
    </div>
  );
}

// ── Step strip ───────────────────────────────────────────────────────

const STEPS = [
  "source_intel","analysis","impact","quality","causal_council","lead_crystallize","lead_gen","learning_update",
];
const STEP_LABELS: Record<string, string> = {
  source_intel: "Sources", analysis: "Analysis", impact: "Impact", quality: "Quality",
  causal_council: "Council", lead_crystallize: "Leads", lead_gen: "Outreach", learning_update: "Learning",
};

// ── Main page ─────────────────────────────────────────────────────────

type Selection = { type: "trend"; index: number } | { type: "lead"; index: number } | null;

export default function PipelineTreePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [result, setResult]     = useState<PipelineRunResult | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>(null);

  const [nodes, setNodes, onNodesChange] = useNodesState<AppNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    api.getPipelineResult(id)
      .then((r) => {
        setResult(r);
        if (r.trends.length > 0 || r.leads.length > 0) {
          const { nodes: n, edges: e } = buildGraph(r);
          setNodes(n as AppNode[]);
          setEdges(e);
        }
      })
      .catch(() => setError("Failed to load pipeline result."))
      .finally(() => setLoading(false));
  }, [id, setNodes, setEdges]);

  const handleNodeClick = useCallback((_evt: React.MouseEvent, node: Node) => {
    if (node.type === "trend") {
      setSelection({ type: "trend", index: (node.data as TrendNodeData).index });
    } else if (node.type === "lead") {
      setSelection({ type: "lead", index: (node.data as LeadNodeData).leadIdx });
    } else {
      setSelection(null);
    }
  }, []);

  const selectedTrend = selection?.type === "trend" && result ? (result.trends[selection.index] ?? null) : null;
  const selectedLead  = selection?.type === "lead"  && result ? (result.leads[selection.index]  ?? null) : null;
  const selectedIdx   = selection?.index ?? 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "var(--bg)" }}>

      {/* Header */}
      <div style={{ padding: "12px 22px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0, display: "flex", alignItems: "center", gap: 14 }}>
        <Link href="/history" style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--text-secondary)", textDecoration: "none", padding: "5px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", flexShrink: 0 }}>
          <ArrowLeft size={11} /> History
        </Link>
        <div style={{ width: 1, height: 18, background: "var(--border)" }} />
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text)", fontFamily: "monospace" }}>{id}</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
            Pipeline intelligence tree · {result ? "click any node to inspect" : "loading…"}
          </div>
        </div>
        {result && <div style={{ marginLeft: "auto" }}><StatBar result={result} /></div>}
      </div>

      {/* Step strip */}
      <div style={{ padding: "10px 22px", borderBottom: "1px solid var(--border)", background: "var(--surface-raised)", flexShrink: 0, display: "flex", gap: 4, alignItems: "center" }}>
        {STEPS.map((step, i) => (
          <div key={step} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
              <div style={{ height: 3, width: 56, borderRadius: 2, background: result ? "var(--green)" : "var(--border)" }} />
              <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.04em", color: result ? "var(--text-secondary)" : "var(--text-xmuted)" }}>
                {STEP_LABELS[step]}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div style={{ width: 8, height: 1, background: "var(--border-strong)", flexShrink: 0 }} />
            )}
          </div>
        ))}
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, position: "relative" }}>
        {loading ? (
          <div style={{ padding: 60, textAlign: "center" }}>
            <Loader2 size={28} style={{ color: "var(--text-xmuted)", margin: "0 auto 12px", display: "block", animation: "spin 1s linear infinite" }} />
            <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Loading pipeline data…</p>
          </div>
        ) : error ? (
          <div style={{ padding: 60, textAlign: "center" }}>
            <AlertCircle size={28} style={{ color: "var(--red)", margin: "0 auto 12px", display: "block" }} />
            <p style={{ fontSize: 13, color: "var(--red)" }}>{error}</p>
          </div>
        ) : nodes.length === 0 ? (
          <div style={{ padding: 60, textAlign: "center" }}>
            <Layers size={28} style={{ color: "var(--text-xmuted)", margin: "0 auto 14px", display: "block" }} />
            <p style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 600, marginBottom: 6 }}>No graph data yet</p>
            <p style={{ fontSize: 12, color: "var(--text-xmuted)" }}>This run may not have produced trends or leads yet.</p>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={NODE_TYPES}
            onNodeClick={handleNodeClick}
            onPaneClick={() => setSelection(null)}
            fitView
            fitViewOptions={{ padding: 0.12, maxZoom: 1.2 }}
            minZoom={0.12}
            maxZoom={2.5}
            proOptions={{ hideAttribution: true }}
            style={{ background: "var(--bg)" }}
            nodesDraggable
            elementsSelectable
          >
            <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="var(--border)" />
            <Controls style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }} showInteractive={false} />
            <MiniMap
              style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
              nodeColor={(n) => {
                if (n.type === "run")   return "var(--accent)";
                if (n.type === "trend") return SEVERITY_COLOR[(n.data as TrendNodeData).severity] ?? "var(--blue)";
                if (n.type === "lead")  return TYPE_COLOR[(n.data as LeadNodeData).leadType] ?? "var(--text-muted)";
                return "var(--border)";
              }}
              maskColor="rgba(248,247,242,0.7)"
            />
            <Legend />
          </ReactFlow>
        )}

        {/* Slide-in panels */}
        {selectedTrend && (
          <TrendPanel
            trend={selectedTrend}
            allLeads={result?.leads ?? []}
            onClose={() => setSelection(null)}
          />
        )}
        {selectedLead && (
          <LeadPanel
            lead={selectedLead}
            leadIdx={selectedIdx}
            onClose={() => setSelection(null)}
          />
        )}
      </div>
    </div>
  );
}
