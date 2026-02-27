"use client";

import { useState, useEffect } from "react";
import {
  Building2, ChevronRight, Star,
  Target, Clock, Layers, MessageSquare, Sparkles, Newspaper, ExternalLink,
} from "lucide-react";
import { usePipelineContext } from "@/contexts/pipeline-context";
import type { LeadRecord, TrendData } from "@/lib/types";

// ── Score color ────────────────────────────────────

function confidenceColor(c: number): string {
  if (c >= 0.75) return "var(--green)";
  if (c >= 0.50) return "var(--accent)";
  return "var(--text-muted)";
}

function confidenceBg(c: number): string {
  if (c >= 0.75) return "var(--green-light)";
  if (c >= 0.50) return "var(--amber-light)";
  return "var(--surface-raised)";
}

const TYPE_CLASSES: Record<string, string> = {
  pain: "badge-red",
  opportunity: "badge-green",
  risk: "badge-amber",
  intelligence: "badge-blue",
};

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
          background: confidenceBg(lead.confidence),
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
            color: confidenceColor(lead.confidence),
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

/** Clean up trigger_event: remove if it's just the trend_title repeated/truncated */
function cleanTriggerEvent(lead: LeadRecord): string | null {
  if (!lead.trigger_event) return null;
  const trigger = lead.trigger_event.trim();
  const title = lead.trend_title.trim();
  // Skip if trigger is just the title repeated (possibly with " — " separator and truncation)
  if (trigger === title) return null;
  if (trigger.startsWith(title + " — " + title.substring(0, 20))) return null;
  if (trigger.startsWith(title + " —")) return null;
  return trigger;
}

/** Clean up opening_line: fix common template issues */
function cleanOpeningLine(line: string): string {
  if (!line) return "";
  let cleaned = line;
  // Fix "Your" mid-sentence capitalization
  cleaned = cleaned.replace(/ for Your /g, " for your ");
  // Fix double periods
  cleaned = cleaned.replace(/\.\./g, ".");
  // Fix lowercase trend title at start — capitalize first letter after "The recent"
  cleaned = cleaned.replace(/^"?The recent (.+?) creates/, (match, title) => {
    const capitalized = title.charAt(0).toUpperCase() + title.slice(1);
    return match.replace(title, capitalized);
  });
  return cleaned;
}

function LeadDetail({ lead }: { lead: LeadRecord }) {
  const { trends } = usePipelineContext();
  const matchedTrend = trends.find((t) => t.title === lead.trend_title);
  const location = [lead.company_city, lead.company_state].filter(Boolean).join(", ");
  const triggerEvent = cleanTriggerEvent(lead);
  const openingLine = cleanOpeningLine(lead.opening_line);

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
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 8,
              background: "var(--surface-raised)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Building2 size={16} style={{ color: "var(--text-secondary)" }} />
          </div>
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
              background: confidenceBg(lead.confidence),
              borderRadius: 8,
              padding: "4px 10px",
            }}
          >
            <span
              className="num"
              style={{ fontSize: 20, color: confidenceColor(lead.confidence), lineHeight: 1 }}
            >
              {Math.round(lead.confidence * 100)}
            </span>
          </div>
        </div>

        {/* Badges */}
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`}>{lead.lead_type}</span>
          <span className="badge badge-blue">Hop {lead.hop}</span>
          <span className="badge badge-muted">{lead.event_type}</span>
          {lead.contact_role && <span className="badge badge-muted">{lead.contact_role}</span>}
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

        {/* Scores */}
        <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", gap: 12 }}>
          {[
            { label: "Confidence", value: `${Math.round(lead.confidence * 100)}%`, color: confidenceColor(lead.confidence) },
            { label: "OSS Score", value: lead.oss_score > 0 ? lead.oss_score.toFixed(2) : "—", color: "var(--blue)" },
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

function DetailSection({ label, icon: Icon, children }: { label: string; icon: React.ElementType; children: React.ReactNode }) {
  return (
    <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 8 }}>
        <Icon size={11} style={{ color: "var(--text-xmuted)" }} />
        <span style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.06em" }}>{label}</span>
      </div>
      {children}
    </div>
  );
}

/** Extract domain from URL */
function extractDomain(url: string): string {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch { return url.substring(0, 40); }
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
