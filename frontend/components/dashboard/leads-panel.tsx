"use client";

import { useState, useEffect } from "react";
import {
  Star, ExternalLink, Newspaper,
  Target, Layers, MessageSquare, Sparkles,
} from "lucide-react";
import { CompanyLogo } from "@/components/ui/company-logo";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { confidenceColor, cleanTriggerEvent, cleanOpeningLine, extractDomain } from "@/lib/utils";
import { DetailSection } from "@/components/ui/detail-section";
import type { LeadRecord, TrendData } from "@/lib/types";

// ── Helpers ────────────────────────────────────────

function hashCode(str: string): number {
  let hash = 0
  for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash)
  return Math.abs(hash)
}

function HopBadge({ hop }: { hop?: number }) {
  const colors: Record<number, string> = { 1: "#3b82f6", 2: "#8b5cf6", 3: "#ec4899" }
  const c = colors[hop ?? 0] ?? "var(--fg-muted)"
  return (
    <span style={{ fontSize: "0.68rem", padding: "2px 6px", borderRadius: 4, background: c + "22", color: c, fontWeight: 600 }}>
      Hop {hop ?? "?"}
    </span>
  )
}

function LeadTypeBadge({ type }: { type?: string }) {
  const map: Record<string, [string, string]> = {
    pain:         ["#ef444422", "#ef4444"],
    opportunity:  ["#22c55e22", "#22c55e"],
    risk:         ["#f59e0b22", "#f59e0b"],
    intelligence: ["#3b82f622", "#3b82f6"],
  }
  const [bg, fg] = map[type ?? ""] ?? ["var(--surface-2)", "var(--fg-muted)"]
  return (
    <span style={{ fontSize: "0.68rem", padding: "2px 6px", borderRadius: 4, background: bg, color: fg, textTransform: "capitalize" as const }}>
      {type}
    </span>
  )
}

// ── Lead Card (people-first) ────────────────────────

function LeadCard({
  lead,
  selected,
  onSelect,
}: {
  lead: LeadRecord;
  selected: boolean;
  onSelect: () => void;
}) {
  const location = [lead.company_city, lead.company_state].filter(Boolean).join(", ");
  const confPct = Math.round(lead.confidence * 100);

  // Person display — prefer contact fields, fall back to people[0]
  const p0 = lead.people?.[0];
  const personName = lead.contact_name || p0?.person_name || "";
  const personRole = lead.contact_role || p0?.role || "";
  const personEmail = lead.contact_email || p0?.email || "";
  const emailConf = lead.email_confidence || p0?.email_confidence || 0;

  // Multi-contact count (beyond the primary contact shown)
  const extraPeopleCount = (lead.people?.length ?? 0) > 1 ? lead.people!.length - 1 : 0;

  // Avatar color from name hash
  const AVATAR_COLORS = [
    ["#3b82f622", "#3b82f6"],
    ["#8b5cf622", "#8b5cf6"],
    ["#ec489922", "#ec4899"],
    ["#22c55e22", "#22c55e"],
    ["#f59e0b22", "#f59e0b"],
    ["#14b8a622", "#14b8a6"],
  ]
  const [avatarBg, avatarFg] = AVATAR_COLORS[hashCode(personName || lead.company_name) % AVATAR_COLORS.length]

  // Urgency color
  const urgencyColor =
    lead.urgency_weeks <= 2 ? "var(--red)" :
    lead.urgency_weeks <= 4 ? "var(--amber)" :
    "var(--text-muted)"

  return (
    <div
      className="card card-hover"
      onClick={onSelect}
      style={{
        padding: "14px 16px",
        cursor: "pointer",
        outline: selected ? "2px solid var(--accent)" : "none",
        outlineOffset: -1,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      {/* Top row: avatar + name/role · company + badges */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {/* Avatar circle */}
        <div style={{
          width: 36, height: 36, borderRadius: "50%",
          background: avatarBg, flexShrink: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: avatarFg, lineHeight: 1 }}>
            {(personName || lead.company_name).slice(0, 2).toUpperCase()}
          </span>
        </div>

        {/* Name + role · company */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 13, fontWeight: 600, color: "var(--text)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {personName || <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>No contact</span>}
            {personRole && (
              <span style={{ fontWeight: 400, color: "var(--text-muted)", fontSize: 12 }}>
                {" "}&middot; {personRole}
              </span>
            )}
          </div>
          <div style={{
            fontSize: 11, color: "var(--text-secondary)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {lead.company_name}
          </div>
        </div>

        {/* Hop + type + extra-contacts badges */}
        <div style={{ display: "flex", gap: 4, flexShrink: 0, alignItems: "center" }}>
          <HopBadge hop={lead.hop} />
          <LeadTypeBadge type={lead.lead_type} />
          {extraPeopleCount > 0 && (
            <span style={{ fontSize: "0.65rem", padding: "2px 5px", borderRadius: 4, background: "#14b8a622", color: "#14b8a6", fontWeight: 600 }}>
              +{extraPeopleCount}
            </span>
          )}
        </div>
      </div>

      {/* Trigger row */}
      {lead.trend_title && (
        <div style={{
          fontSize: 11, color: "var(--text-secondary)",
          background: "var(--surface-raised)",
          borderLeft: "2px solid var(--accent)",
          padding: "5px 8px",
          borderRadius: "0 5px 5px 0",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          {lead.trend_title}
        </div>
      )}

      {/* Location + urgency row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11 }}>
        <span style={{ color: "var(--text-muted)" }}>
          {location || <span style={{ color: "var(--text-xmuted)" }}>—</span>}
        </span>
        <span style={{ color: urgencyColor, fontWeight: 600 }}>
          {lead.urgency_weeks}w
        </span>
      </div>

      {/* Score bar */}
      <div>
        <div style={{
          height: 3, borderRadius: 2,
          background: "var(--border)",
          overflow: "hidden",
          marginBottom: 4,
        }}>
          <div style={{
            height: "100%",
            width: `${confPct}%`,
            background: confidenceColor(lead.confidence).text,
            borderRadius: 2,
            transition: "width 400ms ease",
          }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "var(--text-muted)", fontWeight: 500 }}>
            {confPct}% confidence
          </span>
          {lead.oss_score > 0 && (
            <span style={{ fontSize: 10, color: "var(--blue)", fontWeight: 600 }}>
              OSS {lead.oss_score.toFixed(2)}
            </span>
          )}
        </div>
      </div>

      {/* Email row */}
      {personEmail && (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          gap: 6, paddingTop: 4, borderTop: "1px solid var(--border)",
        }}>
          <span style={{
            fontFamily: "monospace", fontSize: 11, color: "var(--text-secondary)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {personEmail}
          </span>
          {emailConf > 0 && (
            <span style={{
              fontSize: 10, padding: "1px 6px", borderRadius: 4, fontWeight: 600, flexShrink: 0,
              background: emailConf > 80 ? "#22c55e22" : "#f59e0b22",
              color: emailConf > 80 ? "#22c55e" : (emailConf > 50 ? "#f59e0b" : "var(--text-muted)"),
            }}>
              {Math.round(emailConf)}%
            </span>
          )}
        </div>
      )}
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
  const _p0 = lead.people?.[0];
  const effRole = lead.contact_role || _p0?.role || "";
  const effName = lead.contact_name || _p0?.person_name || "";
  const effEmail = lead.contact_email || _p0?.email || "";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Company header */}
      <div style={{ padding: "16px 16px 14px", borderBottom: "1px solid var(--border)" }}>
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
          <div style={{
            marginLeft: "auto",
            background: confidenceColor(lead.confidence).bg,
            borderRadius: 8, padding: "4px 10px",
          }}>
            <span className="num" style={{ fontSize: 20, color: confidenceColor(lead.confidence).text, lineHeight: 1 }}>
              {Math.round(lead.confidence * 100)}
            </span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
            <LeadTypeBadge type={lead.lead_type} />
            <HopBadge hop={lead.hop} />
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

        {lead.pain_point && (
          <DetailSection label={lead.lead_type === "opportunity" ? "OPPORTUNITY" : lead.lead_type === "risk" ? "RISK FACTOR" : "PAIN POINT"} icon={Target}>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, paddingLeft: 10, borderLeft: `2px solid ${lead.lead_type === "opportunity" ? "var(--green)" : lead.lead_type === "risk" ? "var(--amber)" : "var(--red)"}`, padding: "8px 10px", borderRadius: "0 7px 7px 0", background: lead.lead_type === "opportunity" ? "var(--green-light)" : lead.lead_type === "risk" ? "var(--amber-light)" : "var(--red-light)" }}>
              {lead.pain_point}
            </div>
          </DetailSection>
        )}

        {lead.service_pitch && (
          <DetailSection label="SERVICE PITCH" icon={Layers}>
            <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65 }}>
              {lead.service_pitch}
            </p>
          </DetailSection>
        )}

        {openingLine && (
          <DetailSection label="SUGGESTED OPENING" icon={MessageSquare}>
            <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.7, fontStyle: "italic", padding: "10px 14px", background: "var(--accent-light)", borderRadius: 8, borderLeft: "2px solid var(--accent)" }}>
              &ldquo;{openingLine}&rdquo;
            </div>
          </DetailSection>
        )}

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

        <InlineSourceArticles matchedTrend={matchedTrend} />

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
  // Use leads as-is — sorting is handled by the parent page (leads/page.tsx)
  const sorted = leads;

  const initialIdx = (() => {
    if (sorted.length === 0) return null;
    if (selectedId) {
      const idx = sorted.findIndex((l) => String(l.id) === selectedId);
      if (idx !== -1) return idx;
    }
    return null;
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
      <div style={{ padding: "18px 24px", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "1rem" }}>
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="card" style={{ padding: "14px 16px" }}>
            <div className="skeleton" style={{ height: 14, width: "50%", marginBottom: 8 }} />
            <div className="skeleton" style={{ height: 12, width: "80%", marginBottom: 4 }} />
            <div className="skeleton" style={{ height: 12, width: "40%" }} />
          </div>
        ))}
      </div>
    );
  }

  if (leads.length === 0) {
    return (
      <div style={{ padding: "40px 20px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
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
      {/* Card grid */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
          gap: "1rem",
        }}>
          {sorted.map((lead, i) => (
            <LeadCard
              key={`${lead.company_name}-${lead.id ?? i}`}
              lead={lead}
              selected={selectedIdx === i}
              onSelect={() => setSelectedIdx(selectedIdx === i ? null : i)}
            />
          ))}
        </div>
      </div>

      {/* Detail pane (slide in when a card is selected) */}
      {selected && (
        <div style={{
          width: 360,
          minWidth: 320,
          maxWidth: 400,
          borderLeft: "1px solid var(--border)",
          overflow: "auto",
          flexShrink: 0,
          background: "var(--surface)",
        }}>
          <LeadDetail lead={selected} />
        </div>
      )}
    </div>
  );
}
