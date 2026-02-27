"use client";

import Link from "next/link";
import { ArrowRight, ChevronRight } from "lucide-react";
import type { LeadRecord } from "@/lib/types";

/** Confidence 0–1 → display color */
function confidenceColor(c: number) {
  if (c >= 0.75) return { text: "var(--green)",  bg: "var(--green-light)"  };
  if (c >= 0.50) return { text: "var(--accent)", bg: "var(--amber-light)"  };
  return               { text: "var(--text-muted)", bg: "var(--surface-raised)" };
}

const HOP_LABELS: Record<number, string> = { 1: "H1", 2: "H2", 3: "H3" };
const TYPE_CLASSES: Record<string, string> = {
  pain: "badge-red",
  opportunity: "badge-green",
  risk: "badge-amber",
  intelligence: "badge-blue",
};

interface LeadsCompactProps {
  leads: LeadRecord[];
  maxVisible?: number;
  onSelect?: (lead: LeadRecord) => void;
}

export function LeadsCompact({ leads, maxVisible = 6, onSelect }: LeadsCompactProps) {
  const sorted = [...leads].sort((a, b) => b.confidence - a.confidence);
  const visible = sorted.slice(0, maxVisible);

  if (leads.length === 0) {
    return (
      <div style={{ padding: "32px 16px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
        No leads yet — run the pipeline to generate.
      </div>
    );
  }

  const rowBaseStyle = {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "10px 14px",
    borderBottom: "1px solid var(--border)",
    textDecoration: "none" as const,
    transition: "background 150ms",
    cursor: "pointer" as const,
  };

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {visible.map((lead, i) => {
        const { text, bg } = confidenceColor(lead.confidence);
        const key = `${lead.company_name}-${lead.id ?? i}`;

        const inner = (
          <>
            <div style={{ width: 36, height: 36, borderRadius: 8, background: bg, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
              <span className="num" style={{ fontSize: 13, color: text, fontWeight: 600, lineHeight: 1 }}>
                {Math.round(lead.confidence * 100)}
              </span>
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {lead.company_name}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {lead.trend_title}
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3, flexShrink: 0 }}>
              <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`} style={{ fontSize: 9 }}>
                {lead.lead_type}
              </span>
              <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>
                {HOP_LABELS[lead.hop] ?? `H${lead.hop}`} · {lead.urgency_weeks}w
              </span>
            </div>
            <ChevronRight size={12} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
          </>
        );

        if (onSelect) {
          return (
            <div
              key={key}
              style={rowBaseStyle}
              onClick={() => onSelect(lead)}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              {inner}
            </div>
          );
        }

        return (
          <Link
            key={key}
            href={`/leads/${lead.id ?? i}`}
            style={rowBaseStyle}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
          >
            {inner}
          </Link>
        );
      })}

      {/* View all */}
      <Link
        href="/leads"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 5,
          padding: "10px 14px",
          fontSize: 12,
          color: "var(--text-secondary)",
          textDecoration: "none",
          fontWeight: 500,
          transition: "color 150ms",
        }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = "var(--text)"; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)"; }}
      >
        View all {leads.length} leads
        <ArrowRight size={12} />
      </Link>
    </div>
  );
}
