"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  X, ArrowRight, MessageSquare, Building2,
  Target, Clock, Layers, Sparkles, Mail, User, Globe, ExternalLink, Newspaper,
} from "lucide-react";
import { CompanyLogo } from "@/components/ui/company-logo";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { confidenceColor, TYPE_CLASSES, cleanTriggerEvent, cleanOpeningLine, extractDomain } from "@/lib/utils";
import { DetailSection } from "@/components/ui/detail-section";
import type { LeadRecord } from "@/lib/types";

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
          <CompanyLogo domain={lead.company_domain} size={34} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {lead.company_name}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {lead.company_size_band}{location ? ` · ${location}` : ""}
            </div>
          </div>

          {/* Confidence */}
          <div style={{ background: confidenceColor(lead.confidence).bg, borderRadius: 8, padding: "4px 10px", flexShrink: 0 }}>
            <span className="num" style={{ fontSize: 18, color: confidenceColor(lead.confidence).text, lineHeight: 1 }}>
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
          <DetailSection label="TRIGGER EVENT" icon={Sparkles}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: triggerEvent ? 6 : 0 }}>
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
              <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, paddingLeft: 10, borderLeft: `2px solid ${lead.lead_type === "opportunity" ? "var(--green)" : lead.lead_type === "risk" ? "var(--amber)" : "var(--red)"}`, background: lead.lead_type === "opportunity" ? "var(--green-light)" : lead.lead_type === "risk" ? "var(--amber-light)" : "var(--red-light)", padding: "8px 10px", borderRadius: "0 7px 7px 0" }}>
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

          {/* Lead Validation */}
          {lead.validation && (
            <div style={{ padding: "0 16px 0", marginBottom: "0" }}>
              <div style={{ borderBottom: "1px solid var(--border)", paddingBottom: "12px", marginBottom: "0" }}>
                <div style={{ fontSize: "0.72rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: "0.5rem", paddingTop: "10px" }}>
                  Lead Validation
                </div>
                <div style={{ background: "var(--surface-raised)", borderRadius: 8, padding: "0.75rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
                    <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", minWidth: 90 }}>Relevance</span>
                    <div style={{ flex: 1, height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{
                        width: `${Math.round((lead.validation.relevance_score ?? 0) * 100)}%`,
                        height: "100%", borderRadius: 3,
                        background: (lead.validation.relevance_score ?? 0) > 0.7 ? "#22c55e" : (lead.validation.relevance_score ?? 0) > 0.4 ? "#f59e0b" : "#ef4444"
                      }} />
                    </div>
                    <span style={{ fontSize: "0.78rem", color: "var(--text)", fontWeight: 600 }}>
                      {Math.round((lead.validation.relevance_score ?? 0) * 100)}%
                    </span>
                  </div>
                  {lead.validation.recommended_service && (
                    <div style={{ marginBottom: "0.4rem", fontSize: "0.8rem" }}>
                      <span style={{ color: "var(--text-muted)" }}>Recommended: </span>
                      <span style={{ color: "var(--text)", fontWeight: 500 }}>{lead.validation.recommended_service}</span>
                      {lead.validation.recommended_offering && (
                        <span style={{ color: "var(--text-muted)" }}> — {lead.validation.recommended_offering}</span>
                      )}
                    </div>
                  )}
                  {lead.validation.reasoning && (
                    <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", margin: 0, lineHeight: 1.5 }}>
                      {lead.validation.reasoning}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Contact info */}
          {(lead.contact_name || lead.contact_email) && (
            <DetailSection label="CONTACT" icon={User}>
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
            </DetailSection>
          )}

          {/* People (tiered contacts with verified badge + reach score) */}
          {lead.people && lead.people.length > 0 && (
            <DetailSection label={`CONTACTS (${lead.people.length})`} icon={User}>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {lead.people.map((person, i) => (
                  <div key={i} style={{ background: "var(--surface-raised)", borderRadius: 7, padding: "8px 10px" }}>
                    <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", marginBottom: 3 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{person.person_name}</span>
                      {person.verified && (
                        <span style={{ fontSize: "0.65rem", color: "#22c55e", background: "#22c55e22",
                          padding: "1px 5px", borderRadius: 3, fontWeight: 600, marginLeft: 6 }}>✓ Verified</span>
                      )}
                      {person.reach_score != null && (
                        <span style={{ color: "var(--text-muted)", fontSize: "0.72rem", marginLeft: 8 }}>
                          Reach: {"★".repeat(Math.min(Math.round((person.reach_score ?? 0) * 5), 5))}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: person.email ? 4 : 0 }}>{person.role}</div>
                    {person.email && (
                      <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                        <Mail size={11} style={{ color: "var(--text-muted)" }} />
                        <a href={`mailto:${person.email}`} style={{ fontSize: 12, color: "var(--blue)" }}>{person.email}</a>
                        {person.email_confidence > 0 && (
                          <span className="badge badge-muted" style={{ fontSize: 9 }}>{Math.round(person.email_confidence * 100)}%</span>
                        )}
                      </div>
                    )}
                    {person.linkedin_url && (
                      <a href={person.linkedin_url} target="_blank" rel="noopener noreferrer"
                        style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--blue)", marginTop: 3 }}>
                        <ExternalLink size={10} /> LinkedIn
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </DetailSection>
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
            <DetailSection label="SUGGESTED OPENING" icon={MessageSquare}>
              <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.7, fontStyle: "italic", padding: "10px 14px", background: "var(--accent-light)", borderRadius: 8, borderLeft: "2px solid var(--accent)" }}>
                &ldquo;{openingLine}&rdquo;
              </div>
            </DetailSection>
          )}

          {/* Email preview */}
          {lead.email_subject && (
            <DetailSection label="EMAIL PREVIEW" icon={Mail}>
              <div style={{ background: "var(--surface-raised)", borderRadius: 7, padding: "10px 12px" }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                  <strong>Subject:</strong> <span style={{ color: "var(--text)" }}>{lead.email_subject}</span>
                </div>
                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, maxHeight: 120, overflow: "hidden", whiteSpace: "pre-wrap" }}>
                  {lead.email_body}
                </div>
              </div>
            </DetailSection>
          )}

          {/* Scores */}
          <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", gap: 12 }}>
            {[
              { label: "Confidence", value: `${Math.round(lead.confidence * 100)}%`, color: confidenceColor(lead.confidence).text },
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

          {/* Company News */}
          {lead.company_news && lead.company_news.length > 0 && (
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ fontSize: "0.72rem", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: "0.5rem" }}>
                Company News ({lead.company_news.length})
              </div>
              {lead.company_news.slice(0, 4).map((article, i) => (
                <div key={i} style={{ borderLeft: "2px solid var(--border-strong)", paddingLeft: "0.75rem", marginBottom: "0.6rem" }}>
                  <a href={article.url} target="_blank" rel="noopener noreferrer"
                    style={{ color: "var(--text)", fontSize: "0.8rem", textDecoration: "none", fontWeight: 500, display: "block" }}>
                    {article.title}
                  </a>
                  {article.summary && (
                    <p style={{ color: "var(--text-muted)", fontSize: "0.75rem", margin: "0.2rem 0 0", lineHeight: 1.4 }}>
                      {article.summary.slice(0, 150)}…
                    </p>
                  )}
                  <span style={{ color: "var(--text-muted)", fontSize: "0.68rem" }}>
                    {article.source_name}{article.published_at ? ` · ${new Date(article.published_at).toLocaleDateString()}` : ""}
                  </span>
                </div>
              ))}
            </div>
          )}

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
          <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)", flexShrink: 0, display: "flex", gap: 8 }}>
            <button
              onClick={viewFull}
              style={{
                flex: 1,
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
              View Call Sheet
              <ArrowRight size={13} />
            </button>
            <button
              onClick={() => { router.push(`/companies/${encodeURIComponent(lead.company_name)}`); onClose(); }}
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
                padding: "9px 16px",
                borderRadius: 8,
                border: "none",
                background: "var(--text)",
                cursor: "pointer",
                fontSize: 12,
                fontWeight: 600,
                color: "var(--bg)",
                transition: "opacity 150ms",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.opacity = "0.85"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.opacity = "1"; }}
            >
              <Building2 size={12} />
              Company Page
              <ArrowRight size={13} />
            </button>
          </div>
        )}
      </div>
    </>
  );
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
    <DetailSection label={`SOURCE ARTICLES (${count})`} icon={Newspaper}>
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
    </DetailSection>
  );
}
