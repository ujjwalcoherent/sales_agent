"use client";

import { use, useState, useEffect } from "react";
import Link from "next/link";
import {
  ArrowLeft, Building2, TrendingUp, Target, Layers,
  MessageSquare, Clock, Sparkles, ThumbsUp, ThumbsDown,
  Mail, User, ExternalLink, Globe, Newspaper, Copy, Check,
  BarChart3, Zap, MapPin, Hash, Link2, ChevronRight,
  GitBranch, ShieldCheck, Users, BrainCircuit,
} from "lucide-react";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import type { LeadRecord, TrendData } from "@/lib/types";

// ── Helpers ───────────────────────────────────────────────────────────

function confidenceColor(c: number) {
  if (c >= 0.75) return { text: "var(--green)", bg: "var(--green-light)", label: "High" };
  if (c >= 0.50) return { text: "var(--amber)", bg: "var(--amber-light)", label: "Med"  };
  return                { text: "var(--text-muted)", bg: "var(--surface-raised)", label: "Low" };
}

const TYPE_COLORS: Record<string, { badge: string; accent: string }> = {
  pain:         { badge: "badge-red",   accent: "var(--red)"   },
  opportunity:  { badge: "badge-green", accent: "var(--green)" },
  risk:         { badge: "badge-amber", accent: "var(--amber)" },
  intelligence: { badge: "badge-blue",  accent: "var(--blue)"  },
};

// Source chip metadata — trust/meaning per provider
const SOURCE_META: Record<string, { label: string; color: string; bg: string; hint: string }> = {
  apollo:   { label: "Apollo",      color: "var(--blue)",           bg: "var(--blue-light)",    hint: "Contact enrichment"  },
  hunter:   { label: "Hunter.io",   color: "var(--green)",          bg: "var(--green-light)",   hint: "Email verification"  },
  searxng:  { label: "SearXNG",     color: "var(--accent)",         bg: "var(--accent-light)",  hint: "Web intelligence"    },
  rss:      { label: "RSS Feeds",   color: "var(--amber)",          bg: "var(--amber-light)",   hint: "Live news signals"   },
  chromadb: { label: "Vector DB",   color: "var(--blue)",           bg: "var(--blue-light)",    hint: "Semantic memory"     },
  llm:      { label: "AI Analysis", color: "var(--text-secondary)", bg: "var(--surface-raised)", hint: "LLM synthesis"      },
  scraper:  { label: "Web Scraper", color: "var(--text-secondary)", bg: "var(--surface-raised)", hint: "Direct scraping"    },
};

function useCopy() {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const copy = (text: string, key: string) => {
    navigator.clipboard.writeText(text).catch(() => {});
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1800);
  };
  return { copiedKey, copy };
}

type Tab = "briefing" | "outreach" | "intel";

// ── Page entry ────────────────────────────────────────────────────────

export default function LeadDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { leads: contextLeads } = usePipelineContext();
  const [lead, setLead]         = useState<LeadRecord | null>(null);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    const idx = Number(id);
    if (contextLeads.length > 0 && idx >= 0 && idx < contextLeads.length) {
      setLead(contextLeads[idx]);
      setLoading(false);
      return;
    }
    api.getLeads({ limit: 200 })
      .then(({ leads }) => { setLead(leads.find((l) => l.id === idx) ?? leads[idx] ?? null); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [id, contextLeads]);

  if (loading) return <LoadingSkeleton />;

  if (!lead) {
    return (
      <div style={{ padding: 60, textAlign: "center" }}>
        <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 12 }}>Lead not found.</p>
        <Link href="/leads" style={{ color: "var(--accent)", fontSize: 13 }}>← Back to leads</Link>
      </div>
    );
  }

  return <LeadDetail lead={lead} />;
}

// ── Main component ────────────────────────────────────────────────────

function LeadDetail({ lead }: { lead: LeadRecord }) {
  const [tab, setTab] = useState<Tab>("briefing");
  const { trends }    = usePipelineContext();
  const { text: scoreText, bg: scoreBg } = confidenceColor(lead.confidence);
  const location   = [lead.company_city, lead.company_state].filter(Boolean).join(", ");
  const typeColors = TYPE_COLORS[lead.lead_type] ?? { badge: "badge-muted", accent: "var(--text-muted)" };

  const matchedTrend = trends.find((t) => t.title === lead.trend_title);

  const intelCount = matchedTrend
    ? (matchedTrend.causal_chain?.length ?? 0) + (matchedTrend.source_links?.length ?? 0)
    : 0;

  const tabs: { key: Tab; label: string; icon: React.ElementType; count?: number }[] = [
    { key: "briefing", label: "Briefing",       icon: Target        },
    { key: "outreach", label: "Outreach + Email", icon: MessageSquare },
    { key: "intel",    label: "Deep Intel",      icon: BrainCircuit, count: intelCount > 0 ? intelCount : undefined },
  ];

  return (
    <>
      {/* ── Sticky header ─────────────────────────────── */}
      <div style={{ flexShrink: 0, borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>

        {/* Identity bar */}
        <div style={{ padding: "11px 22px", display: "flex", alignItems: "center", gap: 12 }}>
          <Link href="/leads" style={{
            display: "flex", alignItems: "center", gap: 5, fontSize: 12,
            color: "var(--text-secondary)", textDecoration: "none",
            padding: "5px 10px", borderRadius: 6,
            border: "1px solid var(--border)", background: "var(--bg)", flexShrink: 0,
          }}>
            <ArrowLeft size={11} /> Leads
          </Link>
          <div style={{ width: 1, height: 18, background: "var(--border)", flexShrink: 0 }} />

          {/* Company monogram */}
          <div style={{ width: 34, height: 34, borderRadius: 9, background: scoreBg, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, border: `1px solid ${scoreText}22` }}>
            <span style={{ fontSize: 12, fontWeight: 800, color: scoreText, fontFamily: "var(--font-display)" }}>
              {lead.company_name.slice(0, 2).toUpperCase()}
            </span>
          </div>

          {/* Name + badges */}
          <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: 15, fontWeight: 700, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {lead.company_name}
            </span>
            <span className={`badge ${typeColors.badge}`} style={{ fontSize: 10, flexShrink: 0 }}>{lead.lead_type}</span>
            <span className="badge badge-blue" style={{ fontSize: 10, flexShrink: 0 }}>Hop {lead.hop}</span>
            {lead.company_size_band && <span className="badge badge-muted" style={{ fontSize: 10, flexShrink: 0 }}>{lead.company_size_band}</span>}
            {location && (
              <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 11, color: "var(--text-muted)", flexShrink: 0 }}>
                <MapPin size={10} /> {location}
              </span>
            )}
          </div>

          {/* Action shortcuts */}
          <div style={{ display: "flex", gap: 7, flexShrink: 0 }}>
            {lead.company_website && (
              <a href={lead.company_website} target="_blank" rel="noopener noreferrer"
                style={{ display: "flex", alignItems: "center", gap: 4, padding: "5px 10px", fontSize: 11, color: "var(--text-secondary)", border: "1px solid var(--border)", borderRadius: 6, textDecoration: "none", background: "var(--bg)" }}>
                <Globe size={11} /> Website
              </a>
            )}
            {lead.contact_linkedin && (
              <a href={lead.contact_linkedin} target="_blank" rel="noopener noreferrer"
                style={{ display: "flex", alignItems: "center", gap: 4, padding: "5px 10px", fontSize: 11, color: "var(--blue)", border: "1px solid var(--blue)44", borderRadius: 6, textDecoration: "none", background: "var(--blue-light)" }}>
                <ExternalLink size={11} /> LinkedIn
              </a>
            )}
            {lead.contact_email && (
              <a href={`mailto:${lead.contact_email}`}
                style={{ display: "flex", alignItems: "center", gap: 4, padding: "5px 10px", fontSize: 11, color: "var(--green)", border: "1px solid var(--green)44", borderRadius: 6, textDecoration: "none", background: "var(--green-light)" }}>
                <Mail size={11} /> Email
              </a>
            )}
          </div>

          {/* Confidence score */}
          <div style={{ flexShrink: 0, display: "flex", flexDirection: "column", alignItems: "center", background: scoreBg, borderRadius: 10, padding: "6px 16px", border: `1px solid ${scoreText}33` }}>
            <span className="num" style={{ fontSize: 26, color: scoreText, lineHeight: 1 }}>
              {Math.round(lead.confidence * 100)}
            </span>
            <span style={{ fontSize: 9, color: scoreText, opacity: 0.7, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              score
            </span>
          </div>
        </div>

        {/* Quick stats */}
        <div style={{ padding: "7px 22px", background: "var(--bg)", borderTop: "1px solid var(--border)", display: "flex", gap: 22, alignItems: "center", flexWrap: "wrap" }}>
          <QuickStat icon={Clock}    label="Urgency"  value={`${lead.urgency_weeks}w`}        color="var(--amber)"  />
          <QuickStat icon={Zap}      label="Event"    value={lead.event_type}                  color="var(--blue)"   />
          <QuickStat icon={Hash}     label="OSS"      value={lead.oss_score > 0 ? lead.oss_score.toFixed(2) : matchedTrend?.oss_score ? matchedTrend.oss_score.toFixed(2) : "—"}  color="var(--accent)" />
          {lead.contact_role && <QuickStat icon={User} label="Target" value={lead.contact_role} color="var(--text)"   />}
          {matchedTrend && (
            <QuickStat icon={ShieldCheck} label="Council" value={`${Math.round(matchedTrend.council_confidence * 100)}%`} color="var(--green)" />
          )}
        </div>

        {/* Tab strip */}
        <div style={{ display: "flex", padding: "0 22px", background: "var(--surface)", gap: 2 }}>
          {tabs.map(({ key, label, icon: Icon, count }) => {
            const active = tab === key;
            return (
              <button key={key} onClick={() => setTab(key)} style={{
                display: "flex", alignItems: "center", gap: 6, padding: "10px 16px", fontSize: 12,
                fontWeight: active ? 700 : 500,
                color: active ? "var(--text)" : "var(--text-muted)",
                background: "none", border: "none", cursor: "pointer",
                borderBottom: `2px solid ${active ? "var(--accent)" : "transparent"}`,
                marginBottom: -1, transition: "color 150ms", whiteSpace: "nowrap",
              }}>
                <Icon size={12} />
                {label}
                {count != null && (
                  <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 999, background: active ? "var(--accent-light)" : "var(--surface-raised)", color: active ? "var(--accent)" : "var(--text-xmuted)", fontWeight: 700 }}>
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Tab body ──────────────────────────────────── */}
      <div style={{ flex: 1, overflow: "auto", padding: "20px 22px", background: "var(--bg)" }}>
        {tab === "briefing" && <BriefingTab  lead={lead} matchedTrend={matchedTrend} />}
        {tab === "outreach" && <OutreachTab  lead={lead} />}
        {tab === "intel"    && <IntelTab     lead={lead} matchedTrend={matchedTrend} />}
      </div>
    </>
  );
}

// ── Tab 1: Briefing ───────────────────────────────────────────────────

function BriefingTab({ lead, matchedTrend }: { lead: LeadRecord; matchedTrend?: TrendData }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, maxWidth: 1100, alignItems: "start" }}>

      {/* ── Left ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

        <Section title="MARKET SIGNAL" icon={TrendingUp} accent="var(--blue)">
          <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text)", lineHeight: 1.45, marginBottom: 10 }}>
            {lead.trend_title}
          </div>
          {lead.trigger_event && (
            <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.75, margin: 0 }}>
              {lead.trigger_event}
            </p>
          )}
          {matchedTrend?.actionable_insight && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "var(--blue-light)", borderRadius: 8, fontSize: 12, color: "var(--blue)", lineHeight: 1.65 }}>
              {matchedTrend.actionable_insight}
            </div>
          )}
        </Section>

        {/* Pain point */}
        {lead.pain_point && (
          <div style={{ borderRadius: 10, overflow: "hidden", border: "1px solid var(--red)66", background: "var(--red-light)" }}>
            <div style={{ padding: "9px 16px", display: "flex", alignItems: "center", gap: 7 }}>
              <Target size={12} style={{ color: "var(--red)" }} />
              <span style={{ fontSize: 10, fontWeight: 700, color: "var(--red)", letterSpacing: "0.07em" }}>PAIN POINT</span>
            </div>
            <div style={{ padding: "12px 16px 16px", fontSize: 14, color: "var(--red)", lineHeight: 1.75, fontWeight: 500 }}>
              {lead.pain_point}
            </div>
          </div>
        )}

        {/* Causal chain */}
        {matchedTrend?.causal_chain && matchedTrend.causal_chain.length > 0 && (
          <Section title="CAUSAL CHAIN" icon={GitBranch} accent="var(--text-muted)">
            <div style={{ display: "flex", flexDirection: "column" }}>
              {matchedTrend.causal_chain.map((step, i) => (
                <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                    <div style={{ width: 20, height: 20, borderRadius: "50%", background: "var(--surface-raised)", border: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: "var(--text-muted)" }}>
                      {i + 1}
                    </div>
                    {i < matchedTrend.causal_chain.length - 1 && (
                      <div style={{ width: 1, flex: 1, minHeight: 16, background: "var(--border)", margin: "2px 0" }} />
                    )}
                  </div>
                  <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, margin: "0 0 10px" }}>{step}</p>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>

      {/* ── Right ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

        {/* Company */}
        <Section title="COMPANY" icon={Building2} accent="var(--accent)">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
              <div style={{ width: 44, height: 44, borderRadius: 10, background: "var(--surface-raised)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontWeight: 800, fontSize: 15, color: "var(--text-muted)", fontFamily: "var(--font-display)" }}>
                {lead.company_name.slice(0, 2).toUpperCase()}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)", marginBottom: 3 }}>{lead.company_name}</div>
                {lead.company_domain && (
                  <a href={lead.company_website || `https://${lead.company_domain}`} target="_blank" rel="noopener noreferrer"
                    style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 12, color: "var(--blue)", textDecoration: "none" }}>
                    <Link2 size={10} /> {lead.company_domain}
                  </a>
                )}
              </div>
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {lead.company_size_band && <span className="badge badge-muted">{lead.company_size_band}</span>}
              {lead.company_city      && <span className="badge badge-muted">{lead.company_city}</span>}
              {lead.company_state     && <span className="badge badge-muted">{lead.company_state}</span>}
            </div>
            {lead.company_cin && (
              <div style={{ fontSize: 11, color: "var(--text-xmuted)", fontFamily: "monospace" }}>CIN: {lead.company_cin}</div>
            )}
            {lead.reason_relevant && (
              <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, paddingLeft: 10, borderLeft: "2px solid var(--accent-mid)" }}>
                {lead.reason_relevant}
              </div>
            )}
          </div>
        </Section>

        {/* Contact */}
        {(lead.contact_name || lead.contact_email) && (
          <Section title="CONTACT" icon={User} accent="var(--green)">
            <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
              <div style={{ width: 46, height: 46, borderRadius: 12, background: "var(--green-light)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, border: "1px solid var(--green)33" }}>
                <User size={20} style={{ color: "var(--green)" }} />
              </div>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 5 }}>
                {lead.contact_name && <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)" }}>{lead.contact_name}</div>}
                {lead.contact_role && <div style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>{lead.contact_role}</div>}
                {lead.contact_email && (
                  <a href={`mailto:${lead.contact_email}`} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--blue)", textDecoration: "none", marginTop: 2 }}>
                    <Mail size={11} style={{ color: "var(--text-muted)" }} />
                    {lead.contact_email}
                    {lead.email_confidence > 0 && (
                      <span className="badge badge-green" style={{ fontSize: 9 }}>{lead.email_confidence}% verified</span>
                    )}
                  </a>
                )}
                {lead.contact_linkedin && (
                  <a href={lead.contact_linkedin} target="_blank" rel="noopener noreferrer"
                    style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, color: "var(--blue)", textDecoration: "none" }}>
                    <ExternalLink size={10} /> View LinkedIn Profile
                  </a>
                )}
              </div>
            </div>
          </Section>
        )}

        {/* Target roles */}
        {matchedTrend?.target_roles && matchedTrend.target_roles.length > 0 && (
          <Section title="WHO TO TARGET" icon={Users} accent="var(--blue)">
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {matchedTrend.target_roles.map((role) => (
                <span key={role} className="badge badge-blue">{role}</span>
              ))}
            </div>
          </Section>
        )}

        {/* News — fully clickable rows */}
        {lead.company_news?.length > 0 && (
          <Section title="RECENT NEWS" icon={Newspaper} accent="var(--text-muted)">
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {lead.company_news.map((news, i) =>
                news.url ? (
                  <a key={i} href={news.url} target="_blank" rel="noopener noreferrer"
                    style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "10px 12px", borderRadius: 8, textDecoration: "none", background: "var(--bg)", border: "1px solid transparent", transition: "border-color 150ms, background 150ms" }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLElement).style.background = "var(--surface)"; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "transparent"; (e.currentTarget as HTMLElement).style.background = "var(--bg)"; }}
                  >
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent-mid)", flexShrink: 0, marginTop: 5 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>{news.title}</div>
                      {news.date && <div style={{ fontSize: 10, color: "var(--text-xmuted)", marginTop: 3 }}>{news.date}</div>}
                    </div>
                    <ChevronRight size={12} style={{ color: "var(--text-xmuted)", flexShrink: 0, marginTop: 3 }} />
                  </a>
                ) : (
                  <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "10px 12px", borderRadius: 8, background: "var(--bg)" }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--border-strong)", flexShrink: 0, marginTop: 5 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>{news.title}</div>
                      {news.date && <div style={{ fontSize: 10, color: "var(--text-xmuted)", marginTop: 3 }}>{news.date}</div>}
                    </div>
                  </div>
                )
              )}
            </div>
          </Section>
        )}
      </div>
    </div>
  );
}

// ── Tab 2: Outreach + Email ───────────────────────────────────────────

function OutreachTab({ lead }: { lead: LeadRecord }) {
  const { copiedKey, copy } = useCopy();
  const [feedbackSent, setFeedbackSent] = useState<string | null>(null);

  const submitFeedback = async (rating: string) => {
    try {
      await api.submitFeedback({ feedback_type: "lead", item_id: String(lead.id ?? lead.company_name), rating });
      setFeedbackSent(rating);
    } catch { /* silent */ }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, maxWidth: 1100 }}>

      {/* ── Opening line hero ── */}
      {lead.opening_line && (
        <div style={{ borderRadius: 12, background: "var(--surface)", border: "1px solid var(--border)", overflow: "hidden", boxShadow: "var(--shadow-sm)" }}>
          <div style={{ padding: "10px 18px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <MessageSquare size={12} style={{ color: "var(--accent)" }} />
              <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.07em" }}>OPENING LINE</span>
              <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>— start the call or cold email with this</span>
            </div>
            <CopyBtn onCopy={() => copy(lead.opening_line, "opening")} copied={copiedKey === "opening"} />
          </div>
          <div style={{ padding: "22px 24px" }}>
            <div style={{ fontSize: 16, color: "var(--text)", lineHeight: 1.8, fontStyle: "italic", borderLeft: "3px solid var(--accent)", paddingLeft: 18 }}>
              &ldquo;{lead.opening_line}&rdquo;
            </div>
          </div>
        </div>
      )}

      {/* ── Pitch + Email side by side ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.7fr", gap: 14, alignItems: "start" }}>

        {/* Service pitch */}
        {lead.service_pitch && (
          <Section title="SERVICE PITCH" icon={Layers} accent="var(--accent)">
            <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8, margin: 0 }}>
              {lead.service_pitch}
            </p>
          </Section>
        )}

        {/* ── Personalized Email Draft ── */}
        <div style={{ borderRadius: 12, background: "var(--surface)", border: "1px solid var(--border)", overflow: "hidden", boxShadow: "var(--shadow-sm)" }}>

          {/* Email chrome header */}
          <div style={{ padding: "10px 18px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between", background: "var(--surface-raised)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <Mail size={12} style={{ color: "var(--blue)" }} />
              <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.07em" }}>
                PERSONALIZED EMAIL
              </span>
              {lead.email_confidence > 0 && (
                <span className="badge badge-green" style={{ fontSize: 9 }}>
                  {lead.email_confidence}% email confidence
                </span>
              )}
            </div>
            {lead.email_subject && (
              <CopyBtn
                onCopy={() => copy(`Subject: ${lead.email_subject}\n\n${lead.email_body ?? ""}`, "email-full")}
                copied={copiedKey === "email-full"}
                label="Copy full email"
              />
            )}
          </div>

          {lead.email_subject ? (
            <>
              {/* Headers */}
              <div style={{ padding: "0 20px" }}>
                <EmailRow label="To">
                  <span style={{ fontSize: 13, color: "var(--text)" }}>
                    {lead.contact_name ? `${lead.contact_name} ` : ""}
                    {lead.contact_email
                      ? <><span style={{ color: "var(--text-muted)" }}>&lt;</span><a href={`mailto:${lead.contact_email}`} style={{ color: "var(--blue)", textDecoration: "none" }}>{lead.contact_email}</a><span style={{ color: "var(--text-muted)" }}>&gt;</span></>
                      : <span style={{ color: "var(--text-xmuted)", fontStyle: "italic" }}>no contact found</span>
                    }
                  </span>
                </EmailRow>

                <div style={{ display: "flex", padding: "10px 0", borderBottom: "1px solid var(--border)", alignItems: "baseline", gap: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--text-xmuted)", width: 64, flexShrink: 0, fontWeight: 600 }}>Subject:</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)", flex: 1, lineHeight: 1.4 }}>{lead.email_subject}</span>
                  <button onClick={() => copy(lead.email_subject, "subj")} style={{ display: "flex", alignItems: "center", gap: 3, padding: "2px 8px", fontSize: 9, color: copiedKey === "subj" ? "var(--green)" : "var(--text-xmuted)", background: copiedKey === "subj" ? "var(--green-light)" : "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 4, cursor: "pointer", flexShrink: 0 }}>
                    {copiedKey === "subj" ? <Check size={9} /> : <Copy size={9} />}
                    {copiedKey === "subj" ? "Copied" : "Copy"}
                  </button>
                </div>
              </div>

              {/* Body */}
              <div style={{ padding: "16px 20px 20px" }}>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.85, whiteSpace: "pre-wrap", borderLeft: "2px solid var(--border)", paddingLeft: 14 }}>
                  {lead.email_body}
                </div>
              </div>
            </>
          ) : (
            <div style={{ padding: "30px 20px", textAlign: "center" }}>
              <Mail size={20} style={{ color: "var(--text-xmuted)", marginBottom: 8 }} />
              <p style={{ fontSize: 12, color: "var(--text-xmuted)", margin: 0 }}>No personalized email generated</p>
              <p style={{ fontSize: 11, color: "var(--text-xmuted)", marginTop: 4, opacity: 0.7 }}>
                {!lead.contact_email ? "Contact email not found — domain resolution may have failed" : "Email generation was skipped during pipeline run"}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Feedback ── */}
      <div style={{ borderRadius: 12, background: "var(--surface)", border: "1px solid var(--border)", padding: "18px 20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
          <Sparkles size={12} style={{ color: "var(--text-xmuted)" }} />
          <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.07em" }}>
            RATE THIS LEAD — TRAINS THE SELF-LEARNING SYSTEM
          </span>
        </div>
        {feedbackSent ? (
          <div style={{ padding: "12px 16px", background: "var(--green-light)", borderRadius: 8, fontSize: 13, color: "var(--green)", fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}>
            <Check size={14} />
            Recorded: <span style={{ textTransform: "capitalize" }}>{feedbackSent}</span> — updates the weight learner + adaptive thresholds.
          </div>
        ) : (
          <div style={{ display: "flex", gap: 10 }}>
            {[
              { rating: "good",  label: "Good lead",    sub: "Worth pursuing",  icon: ThumbsUp,   color: "var(--green)", bg: "var(--green-light)"   },
              { rating: "bad",   label: "Bad lead",     sub: "Not a fit",       icon: ThumbsDown, color: "var(--red)",   bg: "var(--red-light)"     },
              { rating: "known", label: "Already knew", sub: "In our pipeline", icon: Sparkles,   color: "var(--text-muted)", bg: "var(--surface-raised)" },
            ].map(({ rating, label, sub, icon: Icon, color, bg }) => (
              <button key={rating} onClick={() => submitFeedback(rating)} style={{
                flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
                padding: "14px 12px", borderRadius: 10, border: "1px solid var(--border)",
                background: "var(--bg)", cursor: "pointer", transition: "all 180ms",
              }}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = bg; (e.currentTarget as HTMLElement).style.borderColor = color; }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "var(--bg)"; (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; }}>
                <Icon size={18} style={{ color }} />
                <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text)" }}>{label}</span>
                <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{sub}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Tab 3: Deep Intel ─────────────────────────────────────────────────

function IntelTab({ lead, matchedTrend }: { lead: LeadRecord; matchedTrend?: TrendData }) {
  const { text: confColor } = confidenceColor(lead.confidence);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, maxWidth: 1100, alignItems: "start" }}>

      {/* ── Left: Scores + Sources ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

        <Section title="SCORING" icon={BarChart3} accent="var(--accent)">
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {[
              { label: "Confidence Score",   value: lead.confidence,                     color: confColor,        fmt: (v: number) => `${Math.round(v * 100)}%` },
              { label: "OSS Score",          value: Math.min(lead.oss_score || matchedTrend?.oss_score || 0, 1),  color: "var(--blue)",    fmt: (v: number) => v > 0 ? v.toFixed(2) : "—" },
              { label: "Urgency Window",     value: Math.min(lead.urgency_weeks / 12, 1), color: "var(--amber)",   fmt: () => `${lead.urgency_weeks}w` },
              ...(matchedTrend ? [{ label: "Council Confidence", value: matchedTrend.council_confidence, color: "var(--green)", fmt: (v: number) => `${Math.round(v * 100)}%` }] : []),
            ].map(({ label, value, color, fmt }) => (
              <div key={label}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 7 }}>
                  <span style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 600 }}>{label}</span>
                  <span className="num" style={{ fontSize: 20, color, fontWeight: 700 }}>{fmt(value)}</span>
                </div>
                <div style={{ height: 6, borderRadius: 3, background: "var(--surface-raised)", overflow: "hidden" }}>
                  <div style={{ height: "100%", borderRadius: 3, background: color, width: `${Math.min(value, 1) * 100}%`, transition: "width 800ms cubic-bezier(0.23,1,0.32,1)" }} />
                </div>
              </div>
            ))}
            <div style={{ display: "flex", gap: 10, paddingTop: 6, borderTop: "1px solid var(--border)" }}>
              <MetaTile label="Hop Level"  value={`Hop ${lead.hop}`}  color="var(--text-secondary)" />
              <MetaTile label="Lead Type"  value={lead.lead_type}     color={TYPE_COLORS[lead.lead_type]?.accent ?? "var(--text-muted)"} />
              <MetaTile label="Event"      value={lead.event_type}    color="var(--blue)" />
            </div>
          </div>
        </Section>

        {/* Data sources — rich chips */}
        {lead.data_sources.length > 0 && (
          <Section title="DATA SOURCES" icon={Sparkles} accent="var(--text-muted)">
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {lead.data_sources.map((src) => {
                const meta = SOURCE_META[src.toLowerCase()] ?? { label: src, color: "var(--text-muted)", bg: "var(--surface-raised)", hint: "Data source" };
                return (
                  <div key={src} title={meta.hint} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 8, background: meta.bg, border: `1px solid ${meta.color}33` }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: meta.color, flexShrink: 0 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: meta.color }}>{meta.label}</div>
                      <div style={{ fontSize: 10, color: meta.color, opacity: 0.7 }}>{meta.hint}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Section>
        )}

        {/* Run metadata */}
        {lead.run_id && (
          <Section title="RUN METADATA" icon={Hash} accent="var(--text-xmuted)">
            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
              {[
                { label: "Run ID",  value: lead.run_id,    mono: true  },
                ...(lead.id != null ? [{ label: "Lead ID", value: `#${lead.id}`, mono: true }] : []),
                { label: "Event",   value: lead.event_type, mono: false },
              ].map(({ label, value, mono }) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
                  <span style={{ fontSize: 11, color: "var(--text-xmuted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</span>
                  <span style={{ fontSize: 12, color: "var(--text-secondary)", fontFamily: mono ? "monospace" : undefined }}>{value}</span>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>

      {/* ── Right: Trend deep intel ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {matchedTrend ? (
          <>
            {/* Buying intent signals */}
            {matchedTrend.buying_intent && Object.keys(matchedTrend.buying_intent).length > 0 && (
              <Section title="BUYING INTENT SIGNALS" icon={ShieldCheck} accent="var(--green)">
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {Object.entries(matchedTrend.buying_intent).map(([signal, detail]) => (
                    <div key={signal} style={{ padding: "10px 12px", background: "var(--green-light)", borderRadius: 8, border: "1px solid var(--green)22" }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: "var(--green)", marginBottom: 3 }}>{signal}</div>
                      <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.55 }}>{detail}</div>
                    </div>
                  ))}
                </div>
              </Section>
            )}

            {/* Evidence snippets */}
            {matchedTrend.article_snippets && matchedTrend.article_snippets.length > 0 && (
              <Section title="EVIDENCE SNIPPETS" icon={Newspaper} accent="var(--text-muted)">
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {matchedTrend.article_snippets.slice(0, 4).map((snippet, i) => (
                    <div key={i} style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65, padding: "8px 12px", background: "var(--bg)", borderRadius: 6, borderLeft: "2px solid var(--border-strong)", fontStyle: "italic" }}>
                      &ldquo;{snippet}&rdquo;
                    </div>
                  ))}
                </div>
              </Section>
            )}

            {/* Source links — fully clickable */}
            {matchedTrend.source_links && matchedTrend.source_links.length > 0 && (
              <Section title="SOURCE ARTICLES" icon={Link2} accent="var(--blue)">
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {matchedTrend.source_links.map((url, i) => {
                    let domain = url;
                    try { domain = new URL(url).hostname.replace("www.", ""); } catch {}
                    return (
                      <a key={i} href={url} target="_blank" rel="noopener noreferrer" style={{
                        display: "flex", alignItems: "center", gap: 10, padding: "9px 12px",
                        borderRadius: 8, textDecoration: "none",
                        background: "var(--bg)", border: "1px solid transparent",
                        transition: "border-color 150ms, background 150ms",
                      }}
                        onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLElement).style.background = "var(--surface)"; }}
                        onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "transparent"; (e.currentTarget as HTMLElement).style.background = "var(--bg)"; }}
                      >
                        <Globe size={12} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--blue)", marginBottom: 1 }}>{domain}</div>
                          <div style={{ fontSize: 10, color: "var(--text-xmuted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{url}</div>
                        </div>
                        <ExternalLink size={11} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
                      </a>
                    );
                  })}
                </div>
              </Section>
            )}

            {/* Pitch angle */}
            {matchedTrend.pitch_angle && (
              <Section title="TREND PITCH ANGLE" icon={Target} accent="var(--accent)">
                <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.75, margin: 0 }}>
                  {matchedTrend.pitch_angle}
                </p>
              </Section>
            )}
          </>
        ) : (
          <div style={{ padding: "48px 24px", background: "var(--surface)", borderRadius: 12, border: "1px solid var(--border)", textAlign: "center" }}>
            <BrainCircuit size={30} style={{ color: "var(--text-xmuted)", margin: "0 auto 12px", display: "block" }} />
            <p style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 600, marginBottom: 6 }}>No trend cross-reference</p>
            <p style={{ fontSize: 12, color: "var(--text-xmuted)" }}>
              Full trend intel is available when the pipeline is run from the dashboard.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Shared components ─────────────────────────────────────────────────

function Section({ title, icon: Icon, accent = "var(--text-xmuted)", children }: {
  title: string; icon: React.ElementType; accent?: string; children: React.ReactNode;
}) {
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden", boxShadow: "var(--shadow-xs)" }}>
      <div style={{ padding: "10px 18px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 7 }}>
        <Icon size={12} style={{ color: accent }} />
        <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.07em" }}>{title}</span>
      </div>
      <div style={{ padding: "16px 18px" }}>{children}</div>
    </div>
  );
}

function QuickStat({ icon: Icon, label, value, color }: { icon: React.ElementType; label: string; value: string; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <Icon size={11} style={{ color: "var(--text-xmuted)" }} />
      <span style={{ fontSize: 11, color: "var(--text-xmuted)" }}>{label}:</span>
      <span style={{ fontSize: 12, color, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function MetaTile({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ flex: 1, padding: "10px 12px", background: "var(--bg)", borderRadius: 8, border: "1px solid var(--border)", textAlign: "center" }}>
      <div style={{ fontSize: 13, fontWeight: 700, color, lineHeight: 1.2, marginBottom: 4 }}>{value}</div>
      <div style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{label}</div>
    </div>
  );
}

function EmailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", padding: "9px 0", borderBottom: "1px solid var(--border)", alignItems: "baseline", gap: 10 }}>
      <span style={{ fontSize: 11, color: "var(--text-xmuted)", width: 64, flexShrink: 0, fontWeight: 600 }}>{label}:</span>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  );
}

function CopyBtn({ onCopy, copied, label = "Copy" }: { onCopy: () => void; copied: boolean; label?: string }) {
  return (
    <button onClick={onCopy} style={{
      display: "flex", alignItems: "center", gap: 5, padding: "5px 12px", fontSize: 11,
      color: copied ? "var(--green)" : "var(--text-secondary)",
      background: copied ? "var(--green-light)" : "var(--surface-raised)",
      border: `1px solid ${copied ? "var(--green)" : "var(--border)"}`,
      borderRadius: 6, cursor: "pointer", fontWeight: 600, transition: "all 200ms",
    }}>
      {copied ? <Check size={11} /> : <Copy size={11} />}
      {copied ? "Copied!" : label}
    </button>
  );
}

function LoadingSkeleton() {
  return (
    <div style={{ padding: "20px 22px" }}>
      <div className="skeleton" style={{ height: 64, borderRadius: 12, marginBottom: 10 }} />
      <div className="skeleton" style={{ height: 38, borderRadius: 8, marginBottom: 16 }} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[0, 1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 140, borderRadius: 10 }} />)}
      </div>
    </div>
  );
}
