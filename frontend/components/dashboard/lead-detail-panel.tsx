"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  X, Building2, ArrowRight, MessageSquare,
  Target, Clock, Layers, Sparkles, Mail, User, Globe, ExternalLink, Newspaper,
} from "lucide-react";
import { usePipelineContext } from "@/contexts/pipeline-context";
import type { LeadRecord } from "@/lib/types";

/** Clean up trigger_event: remove if it's just the trend_title repeated/truncated */
function cleanTriggerEvent(lead: LeadRecord): string | null {
  if (!lead.trigger_event) return null;
  const trigger = lead.trigger_event.trim();
  const title = lead.trend_title.trim();
  if (trigger === title) return null;
  if (trigger.startsWith(title + " — " + title.substring(0, 20))) return null;
  if (trigger.startsWith(title + " —")) return null;
  return trigger;
}

/** Clean up opening_line: fix common template issues */
function cleanOpeningLine(line: string): string {
  if (!line) return "";
  let cleaned = line;
  cleaned = cleaned.replace(/ for Your /g, " for your ");
  cleaned = cleaned.replace(/\.\./g, ".");
  cleaned = cleaned.replace(/^"?The recent (.+?) creates/, (match, title) => {
    const capitalized = title.charAt(0).toUpperCase() + title.slice(1);
    return match.replace(title, capitalized);
  });
  return cleaned;
}

function confidenceColor(c: number) {
  if (c >= 0.75) return "var(--green)";
  if (c >= 0.50) return "var(--accent)";
  return "var(--text-muted)";
}

function confidenceBg(c: number) {
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

interface LeadDetailPanelProps {
  lead: LeadRecord | null;
  onClose: () => void;
  showViewFull?: boolean;
}

export function LeadDetailPanel({ lead, onClose, showViewFull = true }: LeadDetailPanelProps) {
  const router = useRouter();
  const { trends } = usePipelineContext();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(!!lead);
  }, [lead]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  if (!lead) return null;

  const viewFull = () => {
    router.push(`/leads/${lead.id ?? 0}`);
    onClose();
  };

  const location = [lead.company_city, lead.company_state].filter(Boolean).join(", ");
  const triggerEvent = cleanTriggerEvent(lead);
  const openingLine = cleanOpeningLine(lead.opening_line);

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.25)",
          zIndex: 200,
          opacity: visible ? 1 : 0,
          transition: "opacity 250ms ease",
        }}
      />

      {/* Drawer */}
      <div
        style={{
          position: "fixed",
          top: 0,
          right: 0,
          bottom: 0,
          width: 520,
          maxWidth: "90vw",
          background: "var(--surface)",
          borderLeft: "1px solid var(--border)",
          zIndex: 201,
          display: "flex",
          flexDirection: "column",
          boxShadow: "var(--shadow-lg)",
          transform: visible ? "translateX(0)" : "translateX(100%)",
          transition: "transform 280ms cubic-bezier(0.23, 1, 0.32, 1)",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "14px 16px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            gap: 10,
            flexShrink: 0,
          }}
        >
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 8,
              background: "var(--surface-raised)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Building2 size={15} style={{ color: "var(--text-secondary)" }} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {lead.company_name}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {lead.company_size_band}{location ? ` · ${location}` : ""}
            </div>
          </div>

          {/* Confidence */}
          <div style={{ background: confidenceBg(lead.confidence), borderRadius: 8, padding: "4px 10px", flexShrink: 0 }}>
            <span className="num" style={{ fontSize: 18, color: confidenceColor(lead.confidence), lineHeight: 1 }}>
              {Math.round(lead.confidence * 100)}
            </span>
          </div>

          {/* Close */}
          <button
            onClick={onClose}
            style={{
              width: 28,
              height: 28,
              borderRadius: 6,
              border: "1px solid var(--border)",
              background: "var(--surface)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--text-muted)",
              flexShrink: 0,
              transition: "background 150ms",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface)"; }}
          >
            <X size={13} />
          </button>
        </div>

        {/* Scrollable body */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          {/* Badges row */}
          <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)", display: "flex", gap: 6, flexWrap: "wrap" }}>
            <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`}>{lead.lead_type}</span>
            <span className="badge badge-blue">Hop {lead.hop}</span>
            <span className="badge badge-muted">{lead.event_type}</span>
            {lead.contact_role && <span className="badge badge-muted">{lead.contact_role}</span>}
            <span className="badge badge-amber" style={{ marginLeft: "auto" }}>
              <Clock size={9} /> {lead.urgency_weeks}w urgency
            </span>
          </div>

          {/* Trigger event */}
          <PanelSection label="TRIGGER EVENT" icon={Sparkles}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: triggerEvent ? 6 : 0 }}>
              {lead.trend_title}
            </div>
            {triggerEvent && (
              <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
                {triggerEvent}
              </p>
            )}
          </PanelSection>

          {/* Pain point / Opportunity */}
          {lead.pain_point && (
            <PanelSection label={lead.lead_type === "opportunity" ? "OPPORTUNITY" : lead.lead_type === "risk" ? "RISK FACTOR" : "PAIN POINT"} icon={Target}>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, paddingLeft: 10, borderLeft: `2px solid ${lead.lead_type === "opportunity" ? "var(--green)" : lead.lead_type === "risk" ? "var(--amber)" : "var(--red)"}`, background: lead.lead_type === "opportunity" ? "var(--green-light)" : lead.lead_type === "risk" ? "var(--amber-light)" : "var(--red-light)", padding: "8px 10px", borderRadius: "0 7px 7px 0" }}>
                {lead.pain_point}
              </div>
            </PanelSection>
          )}

          {/* Service pitch */}
          {lead.service_pitch && (
            <PanelSection label="SERVICE PITCH" icon={Layers}>
              <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65 }}>
                {lead.service_pitch}
              </p>
            </PanelSection>
          )}

          {/* Contact info */}
          {(lead.contact_name || lead.contact_email) && (
            <PanelSection label="CONTACT" icon={User}>
              <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                {lead.contact_name && (
                  <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{lead.contact_name}</div>
                )}
                {lead.contact_role && (
                  <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{lead.contact_role}</div>
                )}
                {lead.contact_email && (
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <Mail size={11} style={{ color: "var(--text-muted)" }} />
                    <a href={`mailto:${lead.contact_email}`} style={{ fontSize: 12, color: "var(--blue)" }}>{lead.contact_email}</a>
                    {lead.email_confidence > 0 && (
                      <span className="badge badge-muted" style={{ fontSize: 9 }}>{lead.email_confidence}%</span>
                    )}
                  </div>
                )}
                {lead.contact_linkedin && (
                  <a href={lead.contact_linkedin} target="_blank" rel="noopener noreferrer"
                    style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--blue)" }}>
                    <ExternalLink size={10} /> LinkedIn
                  </a>
                )}
              </div>
            </PanelSection>
          )}

          {/* Company website */}
          {lead.company_domain && (
            <div style={{ padding: "6px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 5 }}>
              <Globe size={10} style={{ color: "var(--text-muted)" }} />
              <a href={lead.company_website || `https://${lead.company_domain}`} target="_blank" rel="noopener noreferrer"
                style={{ fontSize: 11, color: "var(--blue)" }}>{lead.company_domain}</a>
            </div>
          )}

          {/* Opening line */}
          {openingLine && (
            <PanelSection label="SUGGESTED OPENING" icon={MessageSquare}>
              <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.7, fontStyle: "italic", padding: "10px 14px", background: "var(--accent-light)", borderRadius: 8, borderLeft: "2px solid var(--accent)" }}>
                &ldquo;{openingLine}&rdquo;
              </div>
            </PanelSection>
          )}

          {/* Email preview */}
          {lead.email_subject && (
            <PanelSection label="EMAIL PREVIEW" icon={Mail}>
              <div style={{ background: "var(--surface-raised)", borderRadius: 7, padding: "10px 12px" }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                  <strong>Subject:</strong> <span style={{ color: "var(--text)" }}>{lead.email_subject}</span>
                </div>
                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, maxHeight: 120, overflow: "hidden", whiteSpace: "pre-wrap" }}>
                  {lead.email_body}
                </div>
              </div>
            </PanelSection>
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
          <PanelSourceArticles trendTitle={lead.trend_title} trends={trends} />

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

        {/* Footer — View Full */}
        {showViewFull && (
          <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)", flexShrink: 0 }}>
            <button
              onClick={viewFull}
              style={{
                width: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
                padding: "9px 16px",
                borderRadius: 8,
                border: "1px solid var(--border)",
                background: "var(--surface-raised)",
                cursor: "pointer",
                fontSize: 12,
                fontWeight: 500,
                color: "var(--text-secondary)",
                transition: "background 150ms, color 150ms",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)"; (e.currentTarget as HTMLElement).style.color = "var(--text)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)"; }}
            >
              View Full Call Sheet
              <ArrowRight size={13} />
            </button>
          </div>
        )}
      </div>
    </>
  );
}

function PanelSection({ label, icon: Icon, children }: { label: string; icon: React.ElementType; children: React.ReactNode }) {
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

/** Compact source articles for the panel drawer — cross-references via trend_title */
function PanelSourceArticles({ trendTitle, trends }: { trendTitle: string; trends: import("@/lib/types").TrendData[] }) {
  const matchedTrend = trends.find((t) => t.title === trendTitle);
  const snippets = matchedTrend?.article_snippets;
  const links = matchedTrend?.source_links;
  const hasSnippets = snippets && snippets.length > 0;
  const hasLinks = links && links.length > 0;
  if (!hasSnippets && !hasLinks) return null;

  const count = Math.max(snippets?.length ?? 0, links?.length ?? 0);

  return (
    <PanelSection label={`SOURCE ARTICLES (${count})`} icon={Newspaper}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
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
                display: "block", padding: "8px 10px",
                background: "var(--surface-raised)", borderRadius: 7,
                borderLeft: "2px solid var(--blue)",
                textDecoration: "none", transition: "background 150ms",
                cursor: link ? "pointer" : "default",
              }}
              onMouseEnter={(e) => { if (link) (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", lineHeight: 1.4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {title}
                </div>
                {domain && (
                  <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--blue)", flexShrink: 0 }}>
                    {domain} <ExternalLink size={9} />
                  </span>
                )}
              </div>
            </a>
          );
        })}
        {links?.slice(snippets?.length ?? 0).map((link, i) => (
          <a key={`link-${i}`} href={link} target="_blank" rel="noopener noreferrer"
            style={{ display: "flex", alignItems: "center", gap: 5, padding: "6px 10px", fontSize: 11, color: "var(--blue)", textDecoration: "none" }}
          >
            <ExternalLink size={10} /> {extractDomain(link)}
          </a>
        ))}
      </div>
    </PanelSection>
  );
}
