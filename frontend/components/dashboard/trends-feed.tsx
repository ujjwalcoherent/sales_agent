"use client";

import { useState } from "react";
import { ChevronDown, AlertTriangle, Info, Zap } from "lucide-react";
import type { TrendData, Severity } from "@/lib/types";

// ── Severity helpers ───────────────────────────────

const SEVERITY_CONFIG: Record<
  Severity,
  { label: string; className: string; icon: React.ElementType }
> = {
  high:        { label: "High",        className: "badge-red",   icon: AlertTriangle },
  medium:      { label: "Medium",      className: "badge-amber", icon: Zap },
  low:         { label: "Low",         className: "badge-blue",  icon: Info },
  negligible:  { label: "Negligible",  className: "badge-muted", icon: Info },
};

function getSeverity(s: string): Severity {
  return (["high", "medium", "low", "negligible"].includes(s) ? s : "medium") as Severity;
}

// ── Individual trend card ──────────────────────────

function TrendCard({
  trend,
  selected,
  onSelect,
}: {
  trend: TrendData;
  selected: boolean;
  onSelect: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const sev = getSeverity(trend.severity);
  const cfg = SEVERITY_CONFIG[sev];
  const Icon = cfg.icon;

  return (
    <div
      className="card card-hover"
      style={{
        padding: "14px 16px",
        cursor: "pointer",
        outline: selected ? "2px solid var(--accent)" : "none",
        outlineOffset: -2,
        transition: "outline 150ms",
      }}
      onClick={onSelect}
    >
      {/* Top row */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        {/* Severity dot + badge */}
        <div style={{ paddingTop: 3, flexShrink: 0 }}>
          <span className={`badge ${cfg.className}`}>
            <Icon size={10} />
            {cfg.label}
          </span>
        </div>

        {/* Content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--text)",
              lineHeight: 1.4,
              marginBottom: 4,
            }}
          >
            {trend.title}
          </h3>
          <p
            style={{
              fontSize: 12,
              color: "var(--text-secondary)",
              lineHeight: 1.5,
              display: "-webkit-box",
              WebkitLineClamp: expanded ? "none" : 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {trend.summary}
          </p>
        </div>

        {/* Score pill */}
        <div
          style={{
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: 4,
          }}
        >
          <span
            className="num"
            style={{
              fontSize: 18,
              color: "var(--accent)",
              lineHeight: 1,
              letterSpacing: "-0.02em",
            }}
          >
            {(trend.trend_score * 100).toFixed(0)}
          </span>
          <span style={{ fontSize: 9, color: "var(--text-xmuted)", letterSpacing: "0.06em" }}>
            SCORE
          </span>
        </div>
      </div>

      {/* Meta row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginTop: 10,
          flexWrap: "wrap",
        }}
      >
        {[...new Set(trend.industries)].slice(0, 3).map((ind, idx) => (
          <span key={`${ind}-${idx}`} className="badge badge-muted">
            {ind}
          </span>
        ))}
        {trend.industries.length > 3 && (
          <span style={{ fontSize: 11, color: "var(--text-xmuted)" }}>
            +{trend.industries.length - 3}
          </span>
        )}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: "var(--text-xmuted)" }}>
            {trend.article_count} articles
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setExpanded((v) => !v);
            }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 2,
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "var(--text-muted)",
              fontSize: 11,
              padding: 0,
            }}
          >
            <ChevronDown
              size={13}
              style={{
                transition: "transform 200ms",
                transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
              }}
            />
          </button>
        </div>
      </div>

      {/* Expanded: actionable insight + keywords + companies */}
      {expanded && (
        <div
          className="animate-slide-up"
          style={{
            marginTop: 12,
            paddingTop: 12,
            borderTop: "1px solid var(--border)",
          }}
        >
          {trend.actionable_insight && (
            <div
              style={{
                fontSize: 12,
                color: "var(--text-secondary)",
                lineHeight: 1.6,
                marginBottom: 10,
                paddingLeft: 10,
                borderLeft: "2px solid var(--accent-mid)",
              }}
            >
              {trend.actionable_insight}
            </div>
          )}

          {trend.affected_companies.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Companies:</span>
              {trend.affected_companies.map((c) => (
                <span key={c} className="badge badge-blue">
                  {c}
                </span>
              ))}
            </div>
          )}

          {trend.keywords.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {trend.keywords.slice(0, 8).map((kw) => (
                <span key={kw} className="badge badge-muted">{kw}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Trends Feed ────────────────────────────────────

interface TrendsFeedProps {
  trends: TrendData[];
  loading?: boolean;
  selectedId?: string;
  onSelect?: (trend: TrendData) => void;
}

export function TrendsFeed({ trends, loading, selectedId, onSelect }: TrendsFeedProps) {
  if (loading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {[0, 1, 2].map((i) => (
          <div key={i} className="card" style={{ padding: "14px 16px" }}>
            <div className="skeleton" style={{ height: 14, width: "70%", marginBottom: 8 }} />
            <div className="skeleton" style={{ height: 12, width: "90%", marginBottom: 4 }} />
            <div className="skeleton" style={{ height: 12, width: "60%" }} />
          </div>
        ))}
      </div>
    );
  }

  if (trends.length === 0) {
    return (
      <div
        style={{
          padding: 40,
          textAlign: "center",
          color: "var(--text-muted)",
          fontSize: 13,
        }}
      >
        <TrendsFeedEmpty />
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {trends.map((trend) => (
        <TrendCard
          key={trend.id}
          trend={trend}
          selected={trend.id === selectedId}
          onSelect={() => onSelect?.(trend)}
        />
      ))}
    </div>
  );
}

function TrendsFeedEmpty() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 8,
        padding: "40px 20px",
      }}
    >
      <div
        style={{
          width: 44,
          height: 44,
          borderRadius: 10,
          background: "var(--surface-raised)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Zap size={20} style={{ color: "var(--text-xmuted)" }} />
      </div>
      <p style={{ fontSize: 13, color: "var(--text-muted)" }}>No trends yet</p>
      <p style={{ fontSize: 12, color: "var(--text-xmuted)" }}>
        Run the pipeline to detect market signals
      </p>
    </div>
  );
}
