"use client";

import { useEffect, useState, useCallback, useRef, use } from "react";
import Link from "next/link";
import {
  ReactFlow, Node, Edge, Background, Controls, MiniMap,
  Handle, Position, useNodesState, useEdgesState,
  MarkerType, BackgroundVariant,
  type NodeProps, type ReactFlowInstance,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import Dagre from "@dagrejs/dagre";
import {
  ArrowLeft, CheckCircle, XCircle, Loader2, Clock,
  TrendingUp, Users, Building2, AlertCircle, Layers,
  Zap, X, ChevronRight, ChevronDown, ExternalLink, Mail, User, Square,
} from "lucide-react";
import { api } from "@/lib/api";
import { CompanyLogo } from "@/components/ui/company-logo";
import { PanelSection } from "@/components/ui/detail-section";
import type { PipelineRunResult, TrendData, LeadRecord } from "@/lib/types";

// ── Node data types ────────────────────────────────────────────────────

type RunNodeData     = { label: string; status: string; duration: string; trends: number; leads: number; companies: number };
type TrendNodeData   = { title: string; severity: string; articleCount: number; industries: string[]; oss: number; index: number; isExpanded: boolean; companyCount: number };
type CompanyNodeData = { companyName: string; domain: string; leadCount: number; trendId: string; isExpanded: boolean };
type LeadNodeData    = { company: string; contactName: string; leadType: string; confidence: number; role: string; leadId?: number; leadIdx: number };
type PlusNodeData    = { count: number };

type AppNode =
  | Node<RunNodeData,     "run">
  | Node<TrendNodeData,   "trend">
  | Node<CompanyNodeData, "company">
  | Node<LeadNodeData,    "lead">
  | Node<PlusNodeData,    "plus">;

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

function slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function groupLeadsByCompany(leads: LeadRecord[]): Map<string, LeadRecord[]> {
  const map = new Map<string, LeadRecord[]>();
  for (const l of leads) {
    const key = l.company_name || "(unknown)";
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(l);
  }
  return map;
}

/**
 * Build a mapping from each trend index to its associated leads.
 * Uses multiple strategies because lead.trend_title is often the full event
 * summary (paragraph), while trend.title is the short headline.
 *
 *  1. Exact match:  lead.trend_title === trend.title
 *  2. Company match: lead.company_name ∈ trend.affected_companies (fuzzy)
 *  3. Keyword match: significant words from trend.title appear in lead.trend_title
 */
function buildTrendLeadMap(trends: TrendData[], leads: LeadRecord[]): Map<number, LeadRecord[]> {
  const map = new Map<number, LeadRecord[]>();
  const assigned = new Set<number>(); // lead index → already assigned

  // Pass 1: exact match
  for (let ti = 0; ti < trends.length; ti++) {
    const t = trends[ti];
    const matched: LeadRecord[] = [];
    leads.forEach((l, li) => {
      if (!assigned.has(li) && l.trend_title === t.title) {
        matched.push(l);
        assigned.add(li);
      }
    });
    if (matched.length > 0) map.set(ti, matched);
  }

  // Pass 2: affected_companies fuzzy match
  for (let ti = 0; ti < trends.length; ti++) {
    const ac = (trends[ti].affected_companies ?? []).map((c) => c.toLowerCase());
    if (ac.length === 0) continue;
    leads.forEach((l, li) => {
      if (assigned.has(li)) return;
      const cn = l.company_name.toLowerCase();
      if (ac.some((a) => cn.includes(a) || a.includes(cn))) {
        if (!map.has(ti)) map.set(ti, []);
        map.get(ti)!.push(l);
        assigned.add(li);
      }
    });
  }

  // Pass 3: keyword overlap — extract 3+ char words from trend title, check lead.trend_title
  for (let ti = 0; ti < trends.length; ti++) {
    const words = trends[ti].title
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((w) => w.length >= 4);
    if (words.length === 0) continue;
    const threshold = Math.max(2, Math.floor(words.length * 0.4));
    leads.forEach((l, li) => {
      if (assigned.has(li)) return;
      const lt = (l.trend_title ?? "").toLowerCase();
      const hits = words.filter((w) => lt.includes(w)).length;
      if (hits >= threshold) {
        if (!map.has(ti)) map.set(ti, []);
        map.get(ti)!.push(l);
        assigned.add(li);
      }
    });
  }

  return map;
}

// ── Node sizes ────────────────────────────────────────────────────────

const NODE_DIM: Record<string, { w: number; h: number }> = {
  run:     { w: 310, h: 115 },
  trend:   { w: 245, h: 120 },
  company: { w: 220, h: 75  },
  lead:    { w: 205, h: 85  },
  plus:    { w: 120, h: 44  },
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

// ── Build graph (4-tier collapsible) ──────────────────────────────────

const MAX_LEADS_PER_COMPANY = 5;
const MAX_TRENDS            = 18;
const MAX_COMPANIES         = 12;

function buildGraph(
  result: PipelineRunResult,
  expandedTrends: Set<string>,
  expandedCompanies: Set<string>,
): { nodes: AppNode[]; edges: Edge[] } {
  const nodes: AppNode[] = [];
  const edges: Edge[]    = [];

  // Run node
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

  // Pre-compute trend→lead mapping using multi-strategy matching
  const trendLeadMap = buildTrendLeadMap(result.trends, result.leads);

  const shownTrends  = result.trends.slice(0, MAX_TRENDS);
  const hiddenTrends = result.trends.length - shownTrends.length;

  shownTrends.forEach((t, i) => {
    const tid = `trend-${i}`;
    const trendLeads  = trendLeadMap.get(i) ?? [];
    const companyMap  = groupLeadsByCompany(trendLeads);
    const isTrendExpanded = expandedTrends.has(tid);

    // Trend node
    nodes.push({
      id: tid, type: "trend", position: { x: 0, y: 0 },
      data: {
        title:        t.title,
        severity:     t.severity,
        articleCount: t.article_count,
        industries:   t.industries ?? [],
        oss:          t.oss_score,
        index:        i,
        isExpanded:   isTrendExpanded,
        companyCount: companyMap.size,
      },
    });
    edges.push({
      id: `e-run-${tid}`, source: "run", target: tid,
      type: "smoothstep",
      style: { stroke: "var(--border-strong)", strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "var(--border-strong)", width: 12, height: 12 },
    });

    // Expanded: show company children
    if (isTrendExpanded) {
      const companies = Array.from(companyMap.entries()).slice(0, MAX_COMPANIES);
      const hiddenCompanies = companyMap.size - companies.length;

      for (const [companyName, companyLeads] of companies) {
        const cid = `company-${tid}-${slugify(companyName)}`;
        const domain = companyLeads[0]?.company_domain ?? "";
        const isCompanyExpanded = expandedCompanies.has(cid);

        nodes.push({
          id: cid, type: "company", position: { x: 0, y: 0 },
          data: { companyName, domain, leadCount: companyLeads.length, trendId: tid, isExpanded: isCompanyExpanded },
        });
        edges.push({
          id: `e-${tid}-${cid}`, source: tid, target: cid,
          type: "smoothstep",
          style: { stroke: "var(--accent)", strokeWidth: 1.5, opacity: 0.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: "var(--accent)", width: 10, height: 10 },
        });

        // Expanded company: show lead children
        if (isCompanyExpanded) {
          const shown = companyLeads.slice(0, MAX_LEADS_PER_COMPANY);
          const overflow = companyLeads.length - shown.length;

          shown.forEach((l, li) => {
            const lid   = `lead-${cid}-${li}`;
            const color = TYPE_COLOR[l.lead_type] ?? "var(--text-muted)";
            nodes.push({
              id: lid, type: "lead", position: { x: 0, y: 0 },
              data: {
                company:     l.company_name,
                contactName: l.contact_name || "",
                leadType:    l.lead_type,
                confidence:  l.confidence,
                role:        l.contact_role || "",
                leadId:      l.id,
                leadIdx:     result.leads.indexOf(l),
              },
            });
            edges.push({
              id: `e-${cid}-${lid}`, source: cid, target: lid,
              type: "smoothstep",
              style: { stroke: color, strokeWidth: 1.5, opacity: 0.4 },
              markerEnd: { type: MarkerType.ArrowClosed, color, width: 10, height: 10 },
            });
          });

          if (overflow > 0) {
            const pid = `plus-${cid}`;
            nodes.push({ id: pid, type: "plus", position: { x: 0, y: 0 }, data: { count: overflow } });
            edges.push({
              id: `e-${cid}-${pid}`, source: cid, target: pid,
              type: "smoothstep",
              style: { stroke: "var(--border)", strokeWidth: 1, strokeDasharray: "4 2" },
            });
          }
        }
      }

      if (hiddenCompanies > 0) {
        const pid = `plus-companies-${tid}`;
        nodes.push({ id: pid, type: "plus", position: { x: 0, y: 0 }, data: { count: hiddenCompanies } });
        edges.push({
          id: `e-${tid}-${pid}`, source: tid, target: pid,
          type: "smoothstep",
          style: { stroke: "var(--border)", strokeWidth: 1, strokeDasharray: "4 2" },
        });
      }
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
        </div>
        {/* Expand indicator */}
        {data.companyCount > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 7, paddingTop: 7, borderTop: "1px solid var(--border)" }}>
            <span style={{
              fontSize: 9, display: "flex", alignItems: "center", gap: 3, fontWeight: 600,
              color: data.isExpanded ? sc : "var(--text-muted)",
            }}>
              {data.isExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              {data.companyCount} compan{data.companyCount === 1 ? "y" : "ies"}
            </span>
            {(hov || selected) && (
              <span style={{ marginLeft: "auto", fontSize: 9, color: sc, display: "flex", alignItems: "center", gap: 2, fontWeight: 600 }}>
                <ChevronRight size={9} />details
              </span>
            )}
          </div>
        )}
        {data.companyCount === 0 && (hov || selected) && (
          <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 7, paddingTop: 7, borderTop: "1px solid var(--border)" }}>
            <span style={{ marginLeft: "auto", fontSize: 9, color: sc, display: "flex", alignItems: "center", gap: 2, fontWeight: 600 }}>
              <ChevronRight size={9} />details
            </span>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: sc, width: 7, height: 7, border: "2px solid var(--surface)" }} />
    </div>
  );
}

// ── Custom Node: Company ──────────────────────────────────────────────

function CompanyNode({ data, selected }: NodeProps & { data: CompanyNodeData }) {
  const [hov, setHov] = useState(false);
  const sc = "var(--accent)";
  return (
    <div
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      className="node-enter"
      style={{
        width: NODE_DIM.company.w, background: "var(--surface)",
        border: selected ? `2px solid ${sc}` : `1px solid ${hov ? "var(--border-strong)" : "var(--border)"}`,
        borderLeft: `3px solid ${sc}`, borderRadius: 10,
        boxShadow: selected ? `0 0 0 4px ${sc}1A, var(--shadow-md)` : hov ? "var(--shadow-md)" : "var(--shadow-sm)",
        overflow: "hidden", cursor: "pointer",
        transform: hov && !selected ? "translateY(-2px)" : "none",
        transition: "all 180ms ease",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: sc, width: 7, height: 7, border: "2px solid var(--surface)" }} />
      <div style={{ padding: "10px 12px", display: "flex", alignItems: "center", gap: 9 }}>
        <CompanyLogo domain={data.domain} size={28} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 12, fontWeight: 700, color: "var(--text)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {data.companyName}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
            <span style={{ fontSize: 9, color: "var(--text-muted)" }}>
              {data.leadCount} lead{data.leadCount !== 1 ? "s" : ""}
            </span>
            {data.domain && (
              <span style={{ fontSize: 9, color: "var(--text-xmuted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 80 }}>
                {data.domain}
              </span>
            )}
          </div>
        </div>
        {data.isExpanded
          ? <ChevronDown size={12} style={{ color: sc, flexShrink: 0 }} />
          : <ChevronRight size={12} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
        }
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
      className="node-enter"
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
      <div style={{ height: 2, background: `linear-gradient(90deg, ${color}, ${color}44)`, width: `${conf}%` }} />
      <div style={{ padding: "9px 11px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6, marginBottom: 4 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {data.contactName || data.company}
            </div>
            <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {data.contactName ? (data.role || data.company) : data.role}
            </div>
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

const NODE_TYPES = { run: RunNode, trend: TrendNode, company: CompanyNode, lead: LeadNode, plus: PlusNode } as const;

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
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 18px" }}>
        {trend.summary && (
          <PanelSection title="Summary">
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0 }}>{trend.summary}</p>
          </PanelSection>
        )}
        <PanelSection title="Signal Scores">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <ScoreRow label="OSS Score"       value={trend.oss_score}           color="var(--accent)" />
            <ScoreRow label="Trend Score"     value={trend.trend_score}         color="var(--blue)"   />
            <ScoreRow label="Actionability"   value={trend.actionability_score} color="var(--green)"  />
            <ScoreRow label="Council Confid." value={trend.council_confidence}  color={sc}            />
          </div>
        </PanelSection>
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
        {trend.actionable_insight && (
          <PanelSection title="Actionable Insight">
            <p style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6, margin: 0, padding: "10px 13px", background: sc + "0C", borderLeft: `3px solid ${sc}`, borderRadius: "0 7px 7px 0" }}>
              {trend.actionable_insight}
            </p>
          </PanelSection>
        )}
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
        <div style={{ marginTop: 10, height: 4, borderRadius: 2, background: "var(--surface-raised)", overflow: "hidden" }}>
          <div style={{ width: `${conf}%`, height: "100%", background: color, borderRadius: 2, transition: "width 500ms ease" }} />
        </div>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 18px" }}>
        {(lead.contact_name || lead.contact_role || lead.contact_email) && (
          <PanelSection title={lead.contact_name ? "Contact" : "Target Role"}>
            <div style={{ padding: "11px 13px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 9, display: "flex", flexDirection: "column", gap: 7 }}>
              {lead.contact_name ? (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <User size={11} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{lead.contact_name}</span>
                </div>
              ) : lead.contact_role ? (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <User size={11} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{lead.contact_role}</span>
                </div>
              ) : null}
              {lead.contact_name && lead.contact_role && (
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
        {lead.opening_line && (
          <PanelSection title="Opening Line">
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0, padding: "11px 13px", background: color + "0C", borderLeft: `3px solid ${color}`, borderRadius: "0 7px 7px 0", fontStyle: "italic" }}>
              &ldquo;{lead.opening_line}&rdquo;
            </p>
          </PanelSection>
        )}
        {lead.pain_point && (
          <PanelSection title="Pain Point">
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0 }}>{lead.pain_point}</p>
          </PanelSection>
        )}
        {lead.trigger_event && (
          <PanelSection title="Trigger Event">
            <div style={{ display: "flex", gap: 9, alignItems: "flex-start", padding: "9px 12px", background: "var(--amber-light)", borderRadius: 7, border: "1px solid var(--amber)33" }}>
              <Zap size={11} style={{ color: "var(--amber)", flexShrink: 0, marginTop: 2 }} />
              <p style={{ fontSize: 11, color: "var(--text-secondary)", margin: 0, lineHeight: 1.55 }}>{lead.trigger_event}</p>
            </div>
          </PanelSection>
        )}
        {lead.email_subject && (
          <PanelSection title="Email Subject">
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", padding: "9px 12px", background: "var(--bg)", borderRadius: 7, border: "1px solid var(--border)" }}>
              {lead.email_subject}
            </div>
          </PanelSection>
        )}
      </div>
      <div style={{ padding: "12px 18px", borderTop: "1px solid var(--border)", flexShrink: 0 }}>
        <Link href={href} style={{ textDecoration: "none", display: "block" }}>
          <div
            style={{ padding: "12px 16px", borderRadius: 9, background: color, color: "var(--bg)", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, fontSize: 13, fontWeight: 700, cursor: "pointer", transition: "opacity 150ms" }}
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

// ── Company detail panel ─────────────────────────────────────────────

function CompanyPanel({ name, trend, leads, onClose }: {
  name: string; trend: TrendData | null; leads: LeadRecord[]; onClose: () => void;
}) {
  const domain = leads[0]?.company_domain ?? "";
  const sc = "var(--accent)";

  return (
    <div style={{
      position: "absolute", top: 0, right: 0, bottom: 0, width: 390,
      background: "var(--surface)", borderLeft: "1px solid var(--border)",
      boxShadow: "var(--shadow-lg)", zIndex: 20, display: "flex", flexDirection: "column",
      animation: "slide-in-right 220ms ease forwards",
    }}>
      <div style={{ padding: "15px 18px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
          <CompanyLogo domain={domain} size={36} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)" }}>{name}</div>
            {domain && (
              <a href={`https://${domain}`} target="_blank" rel="noopener noreferrer"
                style={{ fontSize: 11, color: "var(--blue)", textDecoration: "none", display: "flex", alignItems: "center", gap: 3, marginTop: 2 }}>
                {domain} <ExternalLink size={9} />
              </a>
            )}
          </div>
          <button onClick={onClose} style={{ padding: "6px 7px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--bg)", cursor: "pointer", color: "var(--text-muted)", display: "flex", flexShrink: 0 }}>
            <X size={13} />
          </button>
        </div>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 18px" }}>
        {trend && (
          <PanelSection title="From Trend">
            <div style={{ padding: "9px 12px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 8, display: "flex", alignItems: "flex-start", gap: 8 }}>
              <TrendingUp size={11} style={{ color: SEVERITY_COLOR[trend.severity] ?? sc, flexShrink: 0, marginTop: 2 }} />
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", lineHeight: 1.4 }}>{trend.title}</div>
                <span style={{ fontSize: 9, fontWeight: 700, color: SEVERITY_COLOR[trend.severity] ?? "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  {trend.severity}
                </span>
              </div>
            </div>
          </PanelSection>
        )}
        <PanelSection title={`Leads (${leads.length})`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {leads.map((l, i) => {
              const c    = TYPE_COLOR[l.lead_type] ?? "var(--text-muted)";
              const bg   = TYPE_BG[l.lead_type] ?? "var(--surface-raised)";
              const conf = Math.round(l.confidence * 100);
              return (
                <Link key={i} href={`/leads/${l.id ?? i}`} style={{ textDecoration: "none" }}>
                  <div
                    style={{ padding: "10px 12px", borderRadius: 8, border: `1px solid ${c}33`, background: "var(--bg)", transition: "all 140ms", display: "flex", alignItems: "center", gap: 8 }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = c + "0D"; e.currentTarget.style.borderColor = c + "77"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "var(--bg)"; e.currentTarget.style.borderColor = c + "33"; }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                        <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 99, background: bg, color: c, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                          {l.lead_type}
                        </span>
                        {l.contact_name && (
                          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)" }}>{l.contact_name}</span>
                        )}
                      </div>
                      {l.contact_role && <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{l.contact_role}</div>}
                      {l.contact_email && (
                        <div style={{ fontSize: 10, color: "var(--blue)", marginTop: 2 }}>{l.contact_email}</div>
                      )}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: c }}>{conf}</span>
                      <ChevronRight size={11} style={{ color: "var(--text-xmuted)" }} />
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </PanelSection>
      </div>
      <div style={{ padding: "12px 18px", borderTop: "1px solid var(--border)", flexShrink: 0 }}>
        <Link href={`/companies?search=${encodeURIComponent(name)}`} style={{ textDecoration: "none", display: "block" }}>
          <div
            style={{ padding: "12px 16px", borderRadius: 9, background: sc, color: "var(--bg)", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, fontSize: 13, fontWeight: 700, cursor: "pointer", transition: "opacity 150ms" }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.88")}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
          >
            <Building2 size={14} /> View Full Company Page
          </div>
        </Link>
      </div>
    </div>
  );
}

// ── Legend (visible on load → fades to resting → full on hover) ──────

const LEGEND_ITEMS = [
  { color: "var(--red)",    label: "High severity" },
  { color: "var(--amber)",  label: "Medium severity" },
  { color: "var(--green)",  label: "Opportunity" },
  { color: "var(--accent)", label: "Company" },
  { color: "var(--blue)",   label: "Intelligence" },
];

function Legend() {
  const [hov, setHov] = useState(false);
  const [fresh, setFresh] = useState(true); // fully visible on first render

  // Stay fully visible for 8s, then gently fade to resting state
  useEffect(() => {
    const t = setTimeout(() => setFresh(false), 8000);
    return () => clearTimeout(t);
  }, []);

  const active = hov || fresh;

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        position: "absolute", top: 14, left: 14, zIndex: 10,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 9,
        padding: "10px 12px",
        boxShadow: active ? "var(--shadow-sm)" : "none",
        opacity: active ? 0.95 : 0.35,
        transition: "opacity 500ms ease, box-shadow 300ms ease",
        cursor: "default",
        pointerEvents: "auto",
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: "5px 14px", alignItems: "center" }}>
        {LEGEND_ITEMS.map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, flexShrink: 0 }} />
            <span style={{ fontSize: 9, color: "var(--text-muted)", fontWeight: 500 }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Stat bar ──────────────────────────────────────────────────────────

function StatBar({ result }: { result: PipelineRunResult }) {
  return (
    <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
      {[
        { icon: TrendingUp, v: result.trends_detected,                   label: "trends",    c: "var(--blue)"       },
        { icon: Building2,  v: result.companies_found,                   label: "companies", c: "var(--accent)"     },
        { icon: Users,      v: result.leads_generated,                   label: "leads",     c: "var(--green)"      },
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

type Selection =
  | { type: "trend"; index: number }
  | { type: "lead"; index: number }
  | { type: "company"; companyName: string; trendIndex: number }
  | null;

export default function PipelineTreePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [result, setResult]     = useState<PipelineRunResult | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>(null);

  // Expand/collapse state
  const [expandedTrends, setExpandedTrends]       = useState<Set<string>>(new Set());
  const [expandedCompanies, setExpandedCompanies] = useState<Set<string>>(new Set());

  const [nodes, setNodes, onNodesChange] = useNodesState<AppNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const rfInstance = useRef<ReactFlowInstance<AppNode, Edge> | null>(null);
  // "full" = fitView entire graph, "node:xyz" = zoom to subtree, "none" = keep viewport
  const fitMode = useRef<string>("full");

  // Toggle helpers — accordion: only one trend open, only one company per trend open
  const toggleTrend = useCallback((trendId: string) => {
    setExpandedTrends(prev => {
      if (prev.has(trendId)) {
        setExpandedCompanies(prevC => {
          const nextC = new Set(prevC);
          for (const key of nextC) {
            if (key.startsWith(`company-${trendId}-`)) nextC.delete(key);
          }
          return nextC;
        });
        fitMode.current = "full"; // collapsing → show whole tree
        return new Set<string>();
      }
      setExpandedCompanies(new Set<string>());
      fitMode.current = `node:${trendId}`; // zoom toward expanded trend subtree
      return new Set([trendId]);
    });
  }, []);

  const toggleCompany = useCallback((companyId: string) => {
    fitMode.current = "none"; // don't move viewport for company toggles
    setExpandedCompanies(prev => {
      if (prev.has(companyId)) {
        const next = new Set(prev);
        next.delete(companyId);
        return next;
      }
      const trendPrefix = companyId.replace(/^(company-trend-\d+)-.+$/, "$1");
      const next = new Set<string>();
      for (const key of prev) {
        if (!key.startsWith(trendPrefix)) next.add(key);
      }
      next.add(companyId);
      return next;
    });
  }, []);

  // Load result + enrich leads with contact names from DB
  useEffect(() => {
    (async () => {
      try {
        const r = await api.getPipelineResult(id);
        // Pipeline result leads often lack contact_name. Fetch enriched leads from DB.
        try {
          const { leads: dbLeads } = await api.getLeads({ run_id: id, limit: 200 });
          const byCompany = new Map<string, LeadRecord>();
          for (const dl of dbLeads) {
            if (dl.contact_name) byCompany.set(dl.company_name, dl);
          }
          // Merge enriched contact info into result leads
          for (const lead of r.leads) {
            if (!lead.contact_name) {
              const enriched = byCompany.get(lead.company_name);
              if (enriched) {
                lead.contact_name  = enriched.contact_name;
                lead.contact_email = enriched.contact_email || lead.contact_email;
                lead.contact_role  = enriched.contact_role  || lead.contact_role;
              }
            }
          }
        } catch { /* DB leads not available — use result as-is */ }
        setResult(r);
      } catch {
        setError("Failed to load pipeline result.");
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  // Reactive graph rebuild when result or expanded state changes
  useEffect(() => {
    if (!result || (result.trends.length === 0 && result.leads.length === 0)) return;
    const { nodes: n, edges: e } = buildGraph(result, expandedTrends, expandedCompanies);
    setNodes(n as AppNode[]);
    setEdges(e);

    const mode = fitMode.current;
    fitMode.current = "full"; // reset default

    if (mode === "none") return; // company toggle — keep viewport

    requestAnimationFrame(() => {
      const rf = rfInstance.current;
      if (!rf) return;

      if (mode.startsWith("node:")) {
        // Zoom toward the expanded trend's subtree
        const targetId = mode.slice(5);
        const subtreeIds = n.filter((nd) => nd.id === targetId || nd.id.includes(targetId)).map((nd) => nd.id);
        if (subtreeIds.length > 0) {
          rf.fitView({ nodes: subtreeIds.map((id) => ({ id })), padding: 0.25, duration: 400, maxZoom: 1 });
          return;
        }
      }
      rf.fitView({ padding: 0.12, duration: 300, maxZoom: 1.2 });
    });
  }, [result, expandedTrends, expandedCompanies, setNodes, setEdges]);

  // Click handler — trend/company toggle + sidebar, lead opens sidebar
  const handleNodeClick = useCallback((_evt: React.MouseEvent, node: Node) => {
    if (node.type === "trend") {
      toggleTrend(node.id);
      setSelection({ type: "trend", index: (node.data as TrendNodeData).index });
    } else if (node.type === "company") {
      toggleCompany(node.id);
      const d = node.data as CompanyNodeData;
      const trendIdx = parseInt(d.trendId.replace("trend-", ""), 10);
      setSelection({ type: "company", companyName: d.companyName, trendIndex: trendIdx });
    } else if (node.type === "lead") {
      setSelection({ type: "lead", index: (node.data as LeadNodeData).leadIdx });
    } else {
      setSelection(null);
    }
  }, [toggleTrend, toggleCompany]);

  const selectedTrend = selection?.type === "trend" && result ? (result.trends[selection.index] ?? null) : null;
  const selectedLead  = selection?.type === "lead"  && result ? (result.leads[selection.index]  ?? null) : null;
  const selectedIdx   = selection?.type === "lead" ? selection.index : 0;
  const selectedCompany = selection?.type === "company" && result ? {
    name: selection.companyName,
    trend: result.trends[selection.trendIndex] ?? null,
    leads: result.leads.filter((l) => l.company_name === selection.companyName),
  } : null;

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
            Pipeline intelligence tree · click trend to expand
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 14 }}>
          {result && <StatBar result={result} />}
          {result?.status === "running" && (
            <button
              onClick={async () => {
                await api.cancelPipeline();
                setResult(r => r ? { ...r, status: "failed" } : r);
              }}
              style={{
                display: "flex", alignItems: "center", gap: 5,
                padding: "6px 12px", borderRadius: 7,
                border: "1px solid var(--red)",
                background: "var(--red-light)",
                color: "var(--red)",
                fontSize: 11, fontWeight: 700,
                cursor: "pointer", flexShrink: 0,
              }}
            >
              <Square size={11} strokeWidth={2.5} />
              Stop Pipeline
            </button>
          )}
        </div>
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
            onInit={(instance) => { rfInstance.current = instance; }}
            fitView
            fitViewOptions={{ padding: 0.12, maxZoom: 1.2 }}
            minZoom={0.12}
            maxZoom={2.5}
            proOptions={{ hideAttribution: true }}
            style={{ background: "var(--bg)" }}
            nodesDraggable
            elementsSelectable
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1.5} color="var(--border-strong)" />
            <Controls
              style={{ background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 8, boxShadow: "var(--shadow-xs)" }}
              showInteractive={false}
            />
            <MiniMap
              style={{ background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 8, opacity: 0.85 }}
              nodeColor={(n) => {
                if (n.type === "run")     return "var(--accent)";
                if (n.type === "trend")   return SEVERITY_COLOR[(n.data as TrendNodeData).severity] ?? "var(--blue)";
                if (n.type === "company") return "var(--accent)";
                if (n.type === "lead")    return TYPE_COLOR[(n.data as LeadNodeData).leadType] ?? "var(--text-muted)";
                return "var(--border)";
              }}
              maskColor="var(--bg)"
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
        {selectedCompany && (
          <CompanyPanel
            name={selectedCompany.name}
            trend={selectedCompany.trend}
            leads={selectedCompany.leads}
            onClose={() => setSelection(null)}
          />
        )}
      </div>
    </div>
  );
}
