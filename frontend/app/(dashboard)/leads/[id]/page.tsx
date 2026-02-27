"use client";

import { use, useState, useEffect } from "react";
import Link from "next/link";
import {
  ArrowLeft, Building2, TrendingUp, Target,
  Layers, MessageSquare, Clock, Sparkles,
  ThumbsUp, ThumbsDown, Mail, User, ExternalLink,
  Globe, Newspaper,
} from "lucide-react";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import type { LeadRecord } from "@/lib/types";

function confidenceColor(c: number) {
  if (c >= 0.75) return { text: "var(--green)",  bg: "var(--green-light)"  };
  if (c >= 0.50) return { text: "var(--accent)", bg: "var(--amber-light)"  };
  return               { text: "var(--text-muted)", bg: "var(--surface-raised)" };
}

const TYPE_CLASSES: Record<string, string> = {
  pain: "badge-red",
  opportunity: "badge-green",
  risk: "badge-amber",
  intelligence: "badge-blue",
};

type Tab = "call-sheet" | "feedback";

export default function LeadDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { leads: contextLeads } = usePipelineContext();
  const [lead, setLead] = useState<LeadRecord | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Try context first
    const idx = Number(id);
    if (contextLeads.length > 0 && idx >= 0 && idx < contextLeads.length) {
      setLead(contextLeads[idx]);
      setLoading(false);
      return;
    }
    // Fall back to API
    api.getLeads({ limit: 200 })
      .then(({ leads }) => {
        const found = leads.find((l) => l.id === idx) ?? leads[idx] ?? null;
        setLead(found);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [id, contextLeads]);

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: "center" }}>
        <div className="skeleton" style={{ height: 20, width: 200, margin: "0 auto 12px" }} />
        <div className="skeleton" style={{ height: 14, width: 140, margin: "0 auto" }} />
      </div>
    );
  }

  if (!lead) {
    return (
      <div style={{ padding: 40, textAlign: "center" }}>
        <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Lead not found.</p>
        <Link href="/leads" style={{ color: "var(--accent)", fontSize: 13, marginTop: 8, display: "inline-block" }}>← Back to leads</Link>
      </div>
    );
  }

  return <LeadDetail lead={lead} />;
}

function LeadDetail({ lead }: { lead: LeadRecord }) {
  const [tab, setTab] = useState<Tab>("call-sheet");
  const { text: scoreText, bg: scoreBg } = confidenceColor(lead.confidence);
  const location = [lead.company_city, lead.company_state].filter(Boolean).join(", ");

  const tabs: { key: Tab; label: string }[] = [
    { key: "call-sheet", label: "Call Sheet" },
    { key: "feedback",   label: "Feedback" },
  ];

  return (
    <>
      {/* ── Compact sticky header ───────────────────────── */}
      <div style={{ padding: "10px 20px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0, display: "flex", alignItems: "center", gap: 10 }}>
        <Link href="/leads" style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--text-secondary)", textDecoration: "none", padding: "4px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
          <ArrowLeft size={11} /> All leads
        </Link>
        <div style={{ width: 1, height: 14, background: "var(--border)", flexShrink: 0 }} />
        <div style={{ width: 26, height: 26, borderRadius: 7, background: "var(--surface-raised)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <Building2 size={13} style={{ color: "var(--text-secondary)" }} />
        </div>
        <span style={{ fontSize: 13, color: "var(--text)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{lead.company_name}</span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>·</span>
        <span style={{ fontSize: 12, color: "var(--text-secondary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {lead.company_size_band}{location ? ` · ${location}` : ""}
        </span>

        {/* Badges */}
        <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`} style={{ fontSize: 9, flexShrink: 0 }}>{lead.lead_type}</span>
        <span className="badge badge-blue" style={{ fontSize: 9, flexShrink: 0 }}>H{lead.hop}</span>

        <div style={{ marginLeft: "auto", background: scoreBg, borderRadius: 8, padding: "4px 14px", display: "flex", alignItems: "center", gap: 5, flexShrink: 0 }}>
          <span className="num" style={{ fontSize: 20, color: scoreText, lineHeight: 1 }}>{Math.round(lead.confidence * 100)}</span>
          <span style={{ fontSize: 9, color: scoreText, opacity: 0.7 }}>score</span>
        </div>
      </div>

      {/* ── Info bar ───────────────────────────────────── */}
      <div style={{ padding: "8px 20px", borderBottom: "1px solid var(--border)", background: "var(--bg)", flexShrink: 0, display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Clock size={11} style={{ color: "var(--text-muted)" }} />
          <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{lead.urgency_weeks}w urgency</span>
        </div>
        {lead.contact_role && (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Target:</span>
            <span style={{ fontSize: 11, color: "var(--text)", fontWeight: 500 }}>{lead.contact_role}</span>
          </div>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Event:</span>
          <span style={{ fontSize: 11, color: "var(--text)" }}>{lead.event_type}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>OSS:</span>
          <span className="num" style={{ fontSize: 12, color: "var(--blue)" }}>{lead.oss_score.toFixed(2)}</span>
        </div>
      </div>

      {/* ── Tab strip ───────────────────────────────────── */}
      <div style={{ padding: "0 20px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0, display: "flex" }}>
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: "10px 16px",
              fontSize: 12,
              fontWeight: tab === key ? 600 : 400,
              color: tab === key ? "var(--text)" : "var(--text-muted)",
              background: "none",
              border: "none",
              borderBottom: `2px solid ${tab === key ? "var(--accent)" : "transparent"}`,
              cursor: "pointer",
              transition: "color 150ms",
              marginBottom: -1,
              whiteSpace: "nowrap",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Scrollable tab body ──────────────────────────── */}
      <div style={{ flex: 1, overflow: "auto", padding: "20px", background: "var(--bg)" }}>
        {tab === "call-sheet" && <CallSheetTab lead={lead} />}
        {tab === "feedback" && <FeedbackTab lead={lead} />}
      </div>
    </>
  );
}

// ── Call Sheet Tab ─────────────────────────────────────────────────────

function CallSheetTab({ lead }: { lead: LeadRecord }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, maxWidth: 980 }}>

      {/* Trigger Trend */}
      <Section title="TRIGGER TREND" icon={TrendingUp}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", lineHeight: 1.4, marginBottom: 6 }}>
          {lead.trend_title}
        </div>
        {lead.trigger_event && (
          <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65 }}>{lead.trigger_event}</p>
        )}
      </Section>

      {/* Company Info */}
      <Section title="COMPANY" icon={Building2}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{lead.company_name}</div>
            {lead.company_website && (
              <a href={lead.company_website} target="_blank" rel="noopener noreferrer" style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 11, color: "var(--blue)" }}>
                <Globe size={10} /> {lead.company_domain || "website"}
              </a>
            )}
          </div>
          {lead.company_cin && (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>CIN: {lead.company_cin}</div>
          )}
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
            {lead.company_size_band && <span className="badge badge-muted">{lead.company_size_band}</span>}
            {lead.company_city && <span className="badge badge-muted">{lead.company_city}</span>}
            {lead.company_state && <span className="badge badge-muted">{lead.company_state}</span>}
          </div>
          {lead.reason_relevant && (
            <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5, marginTop: 4 }}>
              {lead.reason_relevant}
            </div>
          )}
        </div>
      </Section>

      {/* Contact Info */}
      {(lead.contact_name || lead.contact_email) && (
        <Section title="CONTACT" icon={User}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {lead.contact_name && (
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{lead.contact_name}</div>
            )}
            <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{lead.contact_role}</div>
            {lead.contact_email && (
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Mail size={11} style={{ color: "var(--text-muted)" }} />
                <a href={`mailto:${lead.contact_email}`} style={{ fontSize: 12, color: "var(--blue)" }}>
                  {lead.contact_email}
                </a>
                {lead.email_confidence > 0 && (
                  <span className="badge badge-muted" style={{ fontSize: 9 }}>
                    {lead.email_confidence}% verified
                  </span>
                )}
              </div>
            )}
            {lead.contact_linkedin && (
              <a href={lead.contact_linkedin} target="_blank" rel="noopener noreferrer"
                style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "var(--blue)" }}>
                <ExternalLink size={10} /> LinkedIn Profile
              </a>
            )}
          </div>
        </Section>
      )}

      {/* Pain Point — full width */}
      {lead.pain_point && (
        <div style={{ gridColumn: "1 / -1" }}>
          <Section title="PAIN POINT" icon={Target}>
            <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7, paddingLeft: 10, borderLeft: "2px solid var(--red)", background: "var(--red-light)", padding: "10px 14px", borderRadius: "0 8px 8px 0" }}>
              {lead.pain_point}
            </div>
          </Section>
        </div>
      )}

      {/* Service Pitch */}
      {lead.service_pitch && (
        <Section title="SERVICE PITCH" icon={Layers}>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65 }}>
            {lead.service_pitch}
          </p>
        </Section>
      )}

      {/* Opening Line */}
      {lead.opening_line && (
        <Section title="OPENING LINE" icon={MessageSquare}>
          <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.7, fontStyle: "italic", padding: "12px 16px", background: "var(--accent-light)", borderRadius: 8, borderLeft: "2px solid var(--accent)" }}>
            &ldquo;{lead.opening_line}&rdquo;
          </div>
        </Section>
      )}

      {/* Email Preview — full width */}
      {lead.email_subject && (
        <div style={{ gridColumn: "1 / -1" }}>
          <Section title="PERSONALIZED EMAIL" icon={Mail}>
            <div style={{ background: "var(--surface-raised)", borderRadius: 8, padding: "14px 16px" }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>
                <span style={{ fontWeight: 600 }}>Subject: </span>
                <span style={{ color: "var(--text)" }}>{lead.email_subject}</span>
              </div>
              {lead.contact_email && (
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 10 }}>
                  To: {lead.contact_name ? `${lead.contact_name} <${lead.contact_email}>` : lead.contact_email}
                </div>
              )}
              <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10, fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
                {lead.email_body}
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* Company News */}
      {lead.company_news?.length > 0 && (
        <div style={{ gridColumn: "1 / -1" }}>
          <Section title="RECENT COMPANY NEWS" icon={Newspaper}>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {lead.company_news.map((news, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8, padding: "6px 10px", background: "var(--surface-raised)", borderRadius: 6 }}>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.4 }}>
                    {news.title}
                    {news.date && <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: 6 }}>{news.date}</span>}
                  </div>
                  {news.url && (
                    <a href={news.url} target="_blank" rel="noopener noreferrer" style={{ flexShrink: 0, color: "var(--blue)" }}>
                      <ExternalLink size={10} />
                    </a>
                  )}
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* Scores — full width */}
      <div style={{ gridColumn: "1 / -1" }}>
        <Section title="SCORES" icon={Sparkles}>
          <div style={{ display: "flex", gap: 16 }}>
            {[
              { label: "Confidence", value: `${Math.round(lead.confidence * 100)}%`, color: confidenceColor(lead.confidence).text },
              { label: "OSS Score", value: lead.oss_score.toFixed(2), color: "var(--blue)" },
              { label: "Urgency", value: `${lead.urgency_weeks} weeks`, color: "var(--amber)" },
              { label: "Hop Level", value: `Hop ${lead.hop}`, color: "var(--text-secondary)" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ padding: "10px 20px", background: "var(--surface-raised)", borderRadius: 8, textAlign: "center", minWidth: 80 }}>
                <div className="num" style={{ fontSize: 22, color, lineHeight: 1 }}>{value}</div>
                <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>
        </Section>
      </div>

      {/* Data Sources */}
      {lead.data_sources.length > 0 && (
        <div style={{ gridColumn: "1 / -1" }}>
          <Section title="DATA SOURCES" icon={Sparkles}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {lead.data_sources.map((src) => (
                <span key={src} className="badge badge-muted">{src}</span>
              ))}
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}

// ── Feedback Tab ──────────────────────────────────────────────────────

function FeedbackTab({ lead }: { lead: LeadRecord }) {
  const [submitted, setSubmitted] = useState<string | null>(null);

  const submit = async (rating: string) => {
    try {
      await api.submitFeedback({
        feedback_type: "lead",
        item_id: String(lead.id ?? lead.company_name),
        rating,
      });
      setSubmitted(rating);
    } catch {
      // silently fail
    }
  };

  if (submitted) {
    return (
      <div style={{ maxWidth: 480, padding: "40px 0", textAlign: "center" }}>
        <div style={{ fontSize: 14, color: "var(--green)", fontWeight: 600, marginBottom: 6 }}>
          Feedback recorded: {submitted}
        </div>
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          Your feedback helps the self-learning system improve lead quality.
        </p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 480 }}>
      <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16, lineHeight: 1.6 }}>
        Rate this lead to improve future pipeline quality. Feedback is used by the weight learner and adaptive threshold systems.
      </p>
      <div style={{ display: "flex", gap: 10 }}>
        {[
          { rating: "good", label: "Good Lead", icon: ThumbsUp, color: "var(--green)", bg: "var(--green-light)" },
          { rating: "bad", label: "Bad Lead", icon: ThumbsDown, color: "var(--red)", bg: "var(--red-light)" },
          { rating: "known", label: "Already Knew", icon: Sparkles, color: "var(--text-muted)", bg: "var(--surface-raised)" },
        ].map(({ rating, label, icon: Icon, color, bg }) => (
          <button
            key={rating}
            onClick={() => submit(rating)}
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 8,
              padding: "20px 16px",
              borderRadius: 10,
              border: "1px solid var(--border)",
              background: "var(--surface)",
              cursor: "pointer",
              transition: "background 150ms, border-color 150ms",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = bg; (e.currentTarget as HTMLElement).style.borderColor = color; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface)"; (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; }}
          >
            <Icon size={20} style={{ color }} />
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Section card ───────────────────────────────────────────────────────

function Section({ title, icon: Icon, children }: { title: string; icon: React.ElementType; children: React.ReactNode }) {
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden", boxShadow: "var(--shadow-xs)" }}>
      {title && (
        <div style={{ padding: "9px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 6 }}>
          <Icon size={11} style={{ color: "var(--text-xmuted)" }} />
          <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-xmuted)", letterSpacing: "0.07em" }}>{title}</span>
        </div>
      )}
      <div style={{ padding: "16px" }}>{children}</div>
    </div>
  );
}
