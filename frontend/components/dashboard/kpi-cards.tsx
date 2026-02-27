"use client";

import { TrendingUp, Users, Building2, Activity } from "lucide-react";
import type { DashboardStats } from "@/lib/types";

interface KpiCardsProps {
  stats: DashboardStats;
  loading?: boolean;
}

interface CardConfig {
  key: keyof DashboardStats;
  label: string;
  icon: React.ElementType;
  format: (v: number) => string;
  accent?: string;
}

const CARDS: CardConfig[] = [
  {
    key: "trendsDetected",
    label: "Trends Detected",
    icon: TrendingUp,
    format: (v) => String(v),
    accent: "var(--blue)",
  },
  {
    key: "companiesFound",
    label: "Companies Found",
    icon: Building2,
    format: (v) => String(v),
    accent: "var(--green)",
  },
  {
    key: "leadsGenerated",
    label: "Leads Generated",
    icon: Users,
    format: (v) => String(v),
    accent: "var(--accent)",
  },
  {
    key: "pipelineRuns",
    label: "Pipeline Runs",
    icon: Activity,
    format: (v) => String(v),
    accent: "var(--amber)",
  },
];

export function KpiCards({ stats, loading }: KpiCardsProps) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 12,
      }}
    >
      {CARDS.map(({ key, label, icon: Icon, format, accent }) => (
        <div
          key={key}
          className="card"
          style={{
            padding: "18px 20px",
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {/* Header row */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "var(--text-muted)",
                letterSpacing: "0.06em",
                textTransform: "uppercase",
              }}
            >
              {label}
            </span>
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: 7,
                background: "var(--surface-raised)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Icon size={14} style={{ color: accent }} />
            </div>
          </div>

          {/* Value */}
          {loading ? (
            <div className="skeleton" style={{ height: 36, width: "60%", borderRadius: 6 }} />
          ) : (
            <span
              className="num"
              style={{
                fontSize: 32,
                lineHeight: 1,
                color: "var(--text)",
                letterSpacing: "-0.03em",
              }}
            >
              {format(stats[key])}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
