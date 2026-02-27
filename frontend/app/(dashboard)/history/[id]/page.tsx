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
  Zap, GitBranch,
} from "lucide-react";
import { api } from "@/lib/api";
import type { PipelineRunResult } from "@/lib/types";

// ── Types ──────────────────────────────────────────────────────────────

type RunData   = { label: string; status: string; duration: string; trends: number; leads: number; companies: number };
type TrendData = { title: string; severity: string; articleCount: number; industries: string[]; oss: number; index: number };
type LeadData  = { company: string; leadType: string; confidence: number; role: string; leadId?: number };
type PlusData  = { count: number };

type AppNode =
  | Node<RunData,   "run">
  | Node<TrendData, "trend">
  | Node<LeadData,  "lead">
  | Node<PlusData,  "plus">;

// ── Helpers ────────────────────────────────────────────────────────────

const SEVERITY_COLOR: Record<string, string> = {
  high:        "var(--red)",
  medium:      "var(--amber)",
  low:         "var(--green)",
  negligible:  "var(--text-muted)",
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

function formatDuration(s: number) {
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

// ── Node sizes for Dagre layout ────────────────────────────────────────

const NODE_DIM: Record<string, { w: number; h: number }> = {
  run:   { w: 280, h: 100 },
  trend: { w: 230, h: 95  },
  lead:  { w: 190, h: 75  },
  plus:  { w: 110, h: 40  },
};

// ── Dagre layout ───────────────────────────────────────────────────────

function applyLayout(nodes: AppNode[], edges: Edge[]): AppNode[] {
  const g = new Dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", ranksep: 80, nodesep: 30, marginx: 40, marginy: 40 });

  nodes.forEach((n) => {
    const dim = NODE_DIM[n.type ?? "trend"] ?? { w: 200, h: 72 };
    g.setNode(n.id, { width: dim.w, height: dim.h });
  });
  edges.forEach((e) => g.setEdge(e.source, e.target));
  Dagre.layout(g);

  return nodes.map((n) => {
    const { x, y } = g.node(n.id);
    const dim = NODE_DIM[n.type ?? "trend"] ?? { w: 200, h: 72 };
    return { ...n, position: { x: x - dim.w / 2, y: y - dim.h / 2 } };
  });
}

// ── Build graph from pipeline result ──────────────────────────────────

const MAX_LEADS_PER_TREND = 5;
const MAX_TRENDS          = 18;

function buildGraph(result: PipelineRunResult): { nodes: AppNode[]; edges: Edge[] } {
  const nodes: AppNode[] = [];
  const edges: Edge[]    = [];

  // Root: run node
  nodes.push({
    id: "run",
    type: "run",
    position: { x: 0, y: 0 },
    data: {
      label:    result.run_id,
      status:   result.status,
      duration: formatDuration(result.run_time_seconds),
      trends:   result.trends_detected,
      leads:    result.leads_generated,
      companies: result.companies_found,
    },
  });

  // Trend nodes (cap at MAX_TRENDS)
  const shownTrends = result.trends.slice(0, MAX_TRENDS);
  const hiddenTrends = result.trends.length - shownTrends.length;

  shownTrends.forEach((t, i) => {
    const id = `trend-${i}`;
    nodes.push({
      id, type: "trend", position: { x: 0, y: 0 },
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
      id: `e-run-${id}`, source: "run", target: id,
      type: "smoothstep", animated: false,
      style: { stroke: "var(--border-strong)", strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "var(--border-strong)", width: 12, height: 12 },
    });

    // Leads for this trend (cap at MAX_LEADS_PER_TREND)
    const trendLeads = result.leads.filter((l) => l.trend_title === t.title);
    const shown      = trendLeads.slice(0, MAX_LEADS_PER_TREND);
    const overflow   = trendLeads.length - shown.length;

    shown.forEach((l, li) => {
      const lid = `lead-${id}-${li}`;
      const color = TYPE_COLOR[l.lead_type] ?? "var(--text-muted)";
      nodes.push({
        id: lid, type: "lead", position: { x: 0, y: 0 },
        data: {
          company:    l.company_name,
          leadType:   l.lead_type,
          confidence: l.confidence,
          role:       l.contact_role || "",
          leadId:     l.id,
        },
      });
      edges.push({
        id: `e-${id}-${lid}`, source: id, target: lid,
        type: "smoothstep", animated: false,
        style: { stroke: color + "66", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 10, height: 10 },
      });
    });

    if (overflow > 0) {
      const pid = `plus-${id}`;
      nodes.push({ id: pid, type: "plus", position: { x: 0, y: 0 }, data: { count: overflow } });
      edges.push({
        id: `e-${id}-${pid}`, source: id, target: pid,
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

// ── Custom Node: Run ───────────────────────────────────────────────────

function RunNode({ data }: NodeProps & { data: RunData }) {
  const statusColor = data.status === "completed" ? "var(--green)" : data.status === "failed" ? "var(--red)" : "var(--amber)";
  return (
    <div style={{
      width: NODE_DIM.run.w, background: "var(--surface)", border: `2px solid ${statusColor}`,
      borderRadius: 14, boxShadow: "var(--shadow-md)", overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{ padding: "10px 14px", background: statusColor + "18", borderBottom: `1px solid ${statusColor}33`, display: "flex", alignItems: "center", gap: 8 }}>
        {data.status === "completed" && <CheckCircle size={14} style={{ color: statusColor }} />}
        {data.status === "failed"    && <XCircle     size={14} style={{ color: statusColor }} />}
        {data.status === "running"   && <Loader2     size={14} style={{ color: statusColor, animation: "spin 1s linear infinite" }} />}
        <span style={{ fontSize: 11, fontWeight: 800, color: "var(--text)", letterSpacing: "0.01em", fontFamily: "monospace" }}>{data.label}</span>
      </div>
      {/* Stats */}
      <div style={{ padding: "10px 14px", display: "flex", gap: 14, justifyContent: "space-between" }}>
        <NodeStat icon={TrendingUp} value={data.trends}    label="trends"    color="var(--blue)"  />
        <NodeStat icon={Building2}  value={data.companies} label="companies" color="var(--green)" />
        <NodeStat icon={Users}      value={data.leads}     label="leads"     color="var(--green)" />
        <NodeStat icon={Clock}      value={data.duration}  label="runtime"   color="var(--text-muted)" />
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: statusColor, width: 8, height: 8 }} />
    </div>
  );
}

// ── Custom Node: Trend ─────────────────────────────────────────────────

function TrendNode({ data }: NodeProps & { data: TrendData }) {
  const sevColor = SEVERITY_COLOR[data.severity] ?? "var(--text-muted)";
  const sevBg    = sevColor + "18";
  const ind      = data.industries.slice(0, 2).join(" · ");
  return (
    <div style={{
      width: NODE_DIM.trend.w, background: "var(--surface)",
      border: "1px solid var(--border)", borderRadius: 12,
      boxShadow: "var(--shadow-sm)", overflow: "hidden",
      borderTop: `3px solid ${sevColor}`,
    }}>
      <Handle type="target" position={Position.Top}    style={{ background: sevColor, width: 6, height: 6 }} />
      <div style={{ padding: "10px 13px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
          <TrendingUp size={11} style={{ color: sevColor, flexShrink: 0 }} />
          <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", lineHeight: 1.35, flex: 1 }}>
            {data.title.length > 60 ? data.title.slice(0, 58) + "…" : data.title}
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 99, background: sevBg, color: sevColor, textTransform: "uppercase", letterSpacing: "0.06em" }}>
            {data.severity}
          </span>
          {data.articleCount > 0 && (
            <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>{data.articleCount} articles</span>
          )}
          {ind && <span style={{ fontSize: 9, color: "var(--text-xmuted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 110 }}>{ind}</span>}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: sevColor, width: 6, height: 6 }} />
    </div>
  );
}

// ── Custom Node: Lead ──────────────────────────────────────────────────

function LeadNode({ data }: NodeProps & { data: LeadData }) {
  const color = TYPE_COLOR[data.leadType] ?? "var(--text-muted)";
  const bg    = TYPE_BG[data.leadType]   ?? "var(--surface-raised)";
  const conf  = Math.round(data.confidence * 100);
  return (
    <div style={{
      width: NODE_DIM.lead.w, background: "var(--surface)",
      border: `1px solid ${color}55`, borderRadius: 10,
      boxShadow: "var(--shadow-xs)", overflow: "hidden",
    }}>
      <Handle type="target" position={Position.Top} style={{ background: color, width: 6, height: 6 }} />
      <div style={{ padding: "8px 11px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6, marginBottom: 4 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
            {data.company}
          </div>
          <div style={{ fontSize: 13, fontWeight: 800, color, flexShrink: 0 }}>{conf}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 99, background: bg, color, textTransform: "uppercase", letterSpacing: "0.04em", flexShrink: 0 }}>
            {data.leadType}
          </span>
          {data.role && (
            <span style={{ fontSize: 9, color: "var(--text-xmuted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {data.role}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Custom Node: Plus (overflow) ───────────────────────────────────────

function PlusNode({ data }: NodeProps & { data: PlusData }) {
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

const NODE_TYPES = {
  run:   RunNode,
  trend: TrendNode,
  lead:  LeadNode,
  plus:  PlusNode,
} as const;

// ── Helper sub-component ───────────────────────────────────────────────

function NodeStat({ icon: Icon, value, label, color }: { icon: React.ElementType; value: string | number; label: string; color: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
      <Icon size={11} style={{ color }} />
      <span style={{ fontSize: 13, fontWeight: 700, color, lineHeight: 1 }}>{value}</span>
      <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>{label}</span>
    </div>
  );
}

// ── Legend ─────────────────────────────────────────────────────────────

function Legend() {
  return (
    <div style={{ position: "absolute", bottom: 16, left: 16, zIndex: 10, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px", boxShadow: "var(--shadow-md)" }}>
      <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.07em", marginBottom: 8 }}>LEGEND</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {[
          { color: "var(--green)", label: "Completed run",      shape: "circle" },
          { color: "var(--red)",   label: "High severity trend", shape: "line"  },
          { color: "var(--amber)", label: "Med severity trend",  shape: "line"  },
          { color: "var(--green)", label: "Opportunity lead",    shape: "badge" },
          { color: "var(--red)",   label: "Pain lead",           shape: "badge" },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, flexShrink: 0 }} />
            <span style={{ fontSize: 10, color: "var(--text-secondary)" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Stat bar ───────────────────────────────────────────────────────────

function StatBar({ result }: { result: PipelineRunResult }) {
  return (
    <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
      {[
        { icon: TrendingUp, v: result.trends_detected,  label: "trends",    c: "var(--blue)"  },
        { icon: Building2,  v: result.companies_found,  label: "companies", c: "var(--accent)" },
        { icon: Users,      v: result.leads_generated,  label: "leads",     c: "var(--green)" },
        { icon: Clock,      v: formatDuration(result.run_time_seconds), label: "runtime", c: "var(--text-muted)" },
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

// ── Pipeline steps strip ───────────────────────────────────────────────

const STEPS = ["source_intel","analysis","impact","quality","causal_council","lead_crystallize","lead_gen","learning_update"];
const STEP_LABELS: Record<string, string> = {
  source_intel: "Sources", analysis: "Analysis", impact: "Impact", quality: "Quality",
  causal_council: "Council", lead_crystallize: "Leads", lead_gen: "Outreach", learning_update: "Learning",
};

// ── Main page ──────────────────────────────────────────────────────────

export default function PipelineTreePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [result, setResult]   = useState<PipelineRunResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

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

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "var(--bg)" }}>

      {/* ── Header ── */}
      <div style={{ padding: "12px 22px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0, display: "flex", alignItems: "center", gap: 14 }}>
        <Link href="/history" style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--text-secondary)", textDecoration: "none", padding: "5px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", flexShrink: 0 }}>
          <ArrowLeft size={11} /> History
        </Link>
        <div style={{ width: 1, height: 18, background: "var(--border)" }} />
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text)", fontFamily: "monospace" }}>{id}</div>
          {result && <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>Pipeline intelligence tree</div>}
        </div>
        {result && <div style={{ marginLeft: "auto" }}><StatBar result={result} /></div>}
      </div>

      {/* ── Pipeline step strip ── */}
      <div style={{ padding: "10px 22px", borderBottom: "1px solid var(--border)", background: "var(--surface-raised)", flexShrink: 0, display: "flex", gap: 4, alignItems: "center" }}>
        {STEPS.map((step, i) => (
          <div key={step} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
              <div style={{ height: 3, width: 56, borderRadius: 2, background: result ? "var(--green)" : "var(--border)" }} />
              <span style={{ fontSize: 9, color: result ? "var(--text-secondary)" : "var(--text-xmuted)", fontWeight: 700, letterSpacing: "0.04em" }}>
                {STEP_LABELS[step]}
              </span>
            </div>
            {i < STEPS.length - 1 && <div style={{ width: 8, height: 1, background: "var(--border-strong)", flexShrink: 0 }} />}
          </div>
        ))}
      </div>

      {/* ── React Flow canvas ── */}
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
            <p style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 600, marginBottom: 6 }}>No graph data available</p>
            <p style={{ fontSize: 12, color: "var(--text-xmuted)" }}>This run may not have produced trends or leads yet.</p>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={{ padding: 0.12, maxZoom: 1.2 }}
            minZoom={0.15}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
            style={{ background: "var(--bg)" }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={24}
              size={1}
              color="var(--border)"
            />
            <Controls
              style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
              showInteractive={false}
            />
            <MiniMap
              style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
              nodeColor={(n) => {
                if (n.type === "run")   return "var(--accent)";
                if (n.type === "trend") return "var(--blue)";
                if (n.type === "lead")  return TYPE_COLOR[(n.data as LeadData).leadType] ?? "var(--text-muted)";
                return "var(--border)";
              }}
              maskColor="rgba(248,247,242,0.7)"
            />
            <Legend />
          </ReactFlow>
        )}
      </div>
    </div>
  );
}
