"use client";

import { useState, useEffect } from "react";
import {
  ChevronRight, Star,
  Target, Clock, Layers, MessageSquare, Sparkles, Newspaper, ExternalLink,
} from "lucide-react";
import { CompanyLogo } from "@/components/ui/company-logo";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { confidenceColor, TYPE_CLASSES, cleanTriggerEvent, cleanOpeningLine, extractDomain } from "@/lib/utils";
import { DetailSection } from "@/components/ui/detail-section";
import type { LeadRecord, TrendData } from "@/lib/types";

// ── Lead row ───────────────────────────────────────

function LeadRow({
  lead,
  selected,
  onSelect,
}: {
  lead: LeadRecord;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      style={{
        padding: "12px 14px",
        borderBottom: "1px solid var(--border)",
        cursor: "pointer",
        background: selected ? "var(--accent-light)" : "transparent",
        transition: "background 150ms",
        display: "flex",
        alignItems: "center",
        gap: 12,
      }}
      onMouseEnter={(e) => {
        if (!selected) e.currentTarget.style.background = "var(--surface-raised)";
      }}
      onMouseLeave={(e) => {
        if (!selected) e.currentTarget.style.background = "transparent";
      }}
    >
      {/* Confidence badge */}
      <div
        style={{
          width: 40,
          height: 40,
          borderRadius: 8,
          background: confidenceColor(lead.confidence).bg,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <span
          className="num"
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: confidenceColor(lead.confidence).text,
            lineHeight: 1,
          }}
        >
          {Math.round(lead.confidence * 100)}
        </span>
      </div>

      {/* Company + trend */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: "var(--text)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {lead.company_name}
        </div>
        <div
          style={{
            fontSize: 11,
            color: "var(--text-muted)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {lead.trend_title}
        </div>
      </div>

      {/* Meta */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2, flexShrink: 0 }}>
        <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`} style={{ fontSize: 10 }}>
          {lead.lead_type}
        </span>
        <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>
          H{lead.hop} · {lead.urgency_weeks}w
        </span>
      </div>

      <ChevronRight size={13} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
    </div>
  );
}

// ── Lead Detail Pane ──────────────────────────────


function LeadDetail({ lead }: { lead: LeadRecord }) {
  const { trends } = usePipelineContext();
  const matchedTrend = trends.find((t) => t.title === lead.trend_title);
  const location = [lead.company_city, lead.company_state].filter(Boolean).join(", ");
  const triggerEvent = cleanTriggerEvent(lead);
  const openingLine = cleanOpeningLine(lead.opening_line);
  // Effective contact data — fallback to people[0]
  const _p0 = lead.people?.[0];
  const effRole = lead.contact_role || _p0?.role || "";
  const effName = lead.contact_name || _p0?.person_name || "";
  const effEmail = lead.contact_email || _p0?.email || "";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Company header */}
      <div
        style={{
          padding: "16px 16px 14px",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
          <CompanyLogo domain={lead.company_domain} size={36} />
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)" }}>
              {lead.company_name}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {lead.company_size_band}{location ? ` · ${location}` : ""}
            </div>
          </div>
          <div
            style={{
              marginLeft: "auto",
              background: confidenceColor(lead.confidence).bg,
              borderRadius: 8,
              padding: "4px 10px",
            }}
          >
            <span
              className="num"
              style={{ fontSize: 20, color: confidenceColor(lead.confidence).text, lineHeight: 1 }}
            >
              {Math.round(lead.confidence * 100)}
            </span>
          </div>
        </div>

        {/* Badges + Full profile link */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
            <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`}>{lead.lead_type}</span>
            <span className="badge badge-blue">Hop {lead.hop}</span>
            <span className="badge badge-muted">{lead.event_type}</span>
            {effRole && <span className="badge badge-muted">{effRole}</span>}
          </div>
          <a
            href={`/leads/${lead.id ?? 0}`}
            style={{
              display: "inline-flex", alignItems: "center", gap: 4,
              fontSize: 11, fontWeight: 500, color: "var(--accent)",
              textDecoration: "none", padding: "3px 8px", borderRadius: 5,
              border: "1px solid var(--accent)33", background: "var(--accent-light)",
              transition: "opacity 150ms",
            }}
          >
            <ExternalLink size={10} /> Full Profile
          </a>
        </div>
      </div>

      {/* Scrollable detail sections */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {/* Trigger */}
        <DetailSection label="TRIGGER EVENT" icon={Sparkles}>
          <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)", marginBottom: triggerEvent ? 6 : 0 }}>
            {lead.trend_title}
          </div>
          {triggerEvent && (
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
              {triggerEvent}
            </p>
          )}
        </DetailSection>

        {/* Pain point / Opportunity */}
        {lead.pain_point && (
          <DetailSection label={lead.lead_type === "opportunity" ? "OPPORTUNITY" : lead.lead_type === "risk" ? "RISK FACTOR" : "PAIN POINT"} icon={Target}>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, paddingLeft: 10, borderLeft: `2px solid ${lead.lead_type === "opportunity" ? "var(--green)" : lead.lead_type === "risk" ? "var(--amber)" : "var(--red)"}`, padding: "8px 10px", borderRadius: "0 7px 7px 0", background: lead.lead_type === "opportunity" ? "var(--green-light)" : lead.lead_type === "risk" ? "var(--amber-light)" : "var(--red-light)" }}>
              {lead.pain_point}
            </div>
          </DetailSection>
        )}

        {/* Service pitch */}
        {lead.service_pitch && (
          <DetailSection label="SERVICE PITCH" icon={Layers}>
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65 }}>
              {lead.service_pitch}
            </p>
          </DetailSection>
        )}

        {/* Opening line */}
        {openingLine && (
          <DetailSection label="SUGGESTED OPENING" icon={MessageSquare}>
            <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.7, fontStyle: "italic", padding: "10px 14px", background: "var(--accent-light)", borderRadius: 8, borderLeft: "2px solid var(--accent)" }}>
              &ldquo;{openingLine}&rdquo;
            </div>
          </DetailSection>
        )}

        {/* Contact preview */}
        {(effName || effEmail) && (
          <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)" }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.06em", marginBottom: 6 }}>CONTACT</div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 28, height: 28, borderRadius: 7, background: "var(--green-light)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: "var(--green)" }}>{(effName || "?").slice(0, 2).toUpperCase()}</span>
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                {effName && <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{effName}</div>}
                {effRole && <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{effRole}</div>}
              </div>
              {effEmail && <span style={{ fontSize: 10, color: "var(--blue)", flexShrink: 0 }}>{effEmail}</span>}
            </div>
          </div>
        )}

        {/* Scores */}
        <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", gap: 12 }}>
          {[
            { label: "Confidence", value: `${Math.round(lead.confidence * 100)}%`, color: confidenceColor(lead.confidence).text },
            { label: "OSS Score", value: lead.oss_score > 0 ? lead.oss_score.toFixed(2) : matchedTrend?.oss_score ? matchedTrend.oss_score.toFixed(2) : "—", color: "var(--blue)" },
            { label: "Urgency", value: `${lead.urgency_weeks}w`, color: "var(--amber)" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ flex: 1, padding: "8px 0", textAlign: "center" }}>
              <div className="num" style={{ fontSize: 20, color, lineHeight: 1 }}>{value}</div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3 }}>{label}</div>
            </div>
          ))}
        </div>

        {/* Source articles from parent trend */}
        <InlineSourceArticles matchedTrend={matchedTrend} />

        {/* Data sources */}
        {lead.data_sources.length > 0 && (
          <div style={{ padding: "12px 16px" }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.06em", marginBottom: 6 }}>DATA SOURCES</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {lead.data_sources.map((src) => (
                <span key={src} className="badge badge-muted">{src}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


/** Compact source articles for the inline detail pane */
function InlineSourceArticles({ matchedTrend }: { matchedTrend?: TrendData | null }) {
  if (!matchedTrend) return null;
  const snippets = matchedTrend.article_snippets;
  const links = matchedTrend.source_links;
  if ((!snippets || snippets.length === 0) && (!links || links.length === 0)) return null;

  const count = Math.max(snippets?.length ?? 0, links?.length ?? 0);

  return (
    <DetailSection label={`SOURCE ARTICLES (${count})`} icon={Newspaper}>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {snippets?.map((snippet, i) => {
          const colonIdx = snippet.indexOf(":");
          const title = colonIdx > 0 && colonIdx < 120 ? snippet.substring(0, colonIdx).trim() : snippet.substring(0, 80);
          const link = links?.[i];
          const domain = link ? extractDomain(link) : null;
          return (
            <a
              key={i}
              href={link || "#"}
              target={link ? "_blank" : undefined}
              rel={link ? "noopener noreferrer" : undefined}
              style={{
                display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6,
                padding: "7px 10px", background: "var(--surface-raised)", borderRadius: 6,
                borderLeft: "2px solid var(--blue)", textDecoration: "none",
                transition: "background 150ms", cursor: link ? "pointer" : "default",
              }}
              onMouseEnter={(e) => { if (link) (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
            >
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {title}
              </span>
              {domain && (
                <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--blue)", flexShrink: 0 }}>
                  {domain} <ExternalLink size={9} />
                </span>
              )}
            </a>
          );
        })}
      </div>
    </DetailSection>
  );
}

// ── Leads Panel ────────────────────────────────────

interface LeadsPanelProps {
  leads: LeadRecord[];
  loading?: boolean;
  selectedId?: string;
}

export function LeadsPanel({ leads, loading, selectedId }: LeadsPanelProps) {
  const sorted = [...leads].sort((a, b) => b.confidence - a.confidence);

  const initialIdx = (() => {
    if (sorted.length === 0) return null;
    if (selectedId) {
      const idx = sorted.findIndex((l) => String(l.id) === selectedId);
      if (idx !== -1) return idx;
    }
    return 0;
  })();

  const [selectedIdx, setSelectedIdx] = useState<number | null>(initialIdx);

  useEffect(() => {
    if (selectedId) {
      const idx = sorted.findIndex((l) => String(l.id) === selectedId);
      if (idx !== -1) setSelectedIdx(idx);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  const selected = selectedIdx !== null ? sorted[selectedIdx] : null;

  if (loading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {[0, 1, 2, 3].map((i) => (
          <div key={i} style={{ padding: "12px 14px", borderBottom: "1px solid var(--border)" }}>
            <div className="skeleton" style={{ height: 13, width: "50%", marginBottom: 6 }} />
            <div className="skeleton" style={{ height: 11, width: "35%" }} />
          </div>
        ))}
      </div>
    );
  }

  if (leads.length === 0) {
    return (
      <div
        style={{
          padding: "40px 20px",
          textAlign: "center",
          color: "var(--text-muted)",
          fontSize: 13,
        }}
      >
        <Star size={28} style={{ color: "var(--text-xmuted)", margin: "0 auto 10px" }} />
        <p>No leads generated yet</p>
        <p style={{ fontSize: 12, color: "var(--text-xmuted)", marginTop: 4 }}>
          Run the pipeline to start
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      {/* Lead list */}
      <div
        style={{
          width: 280,
          minWidth: 280,
          borderRight: "1px solid var(--border)",
          overflow: "auto",
          flexShrink: 0,
        }}
      >
        {sorted.map((lead, i) => (
          <LeadRow
            key={`${lead.company_name}-${lead.id ?? i}`}
            lead={lead}
            selected={selectedIdx === i}
            onSelect={() => setSelectedIdx(i)}
          />
        ))}
      </div>

      {/* Detail pane */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {selected ? (
          <LeadDetail lead={selected} />
        ) : (
          <div
            style={{
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--text-muted)",
              fontSize: 13,
            }}
          >
            Select a lead to view details
          </div>
        )}
      </div>
    </div>
  );
}
