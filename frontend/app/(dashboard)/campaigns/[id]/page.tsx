"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft, Building2, Users, Mail, Loader2,
  CheckCircle, XCircle, Clock, RefreshCw,
  ChevronDown, ChevronRight, Copy, Check, Download,
  User, Briefcase, Shield, FileText, Play, Trash2,
  Globe, AlertCircle, ExternalLink, Send,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import type {
  Campaign, CampaignCompanyStatus, CampaignStreamEvent,
  CampaignContact, CampaignEmail,
} from "@/lib/types";

/* ── Status maps ─────────────────────────────────────────── */

const STATUS_DOT: Record<string, string> = {
  pending: "var(--text-muted)",
  enriching: "var(--blue)", enriched: "var(--blue)",
  contacts: "var(--amber)", contacts_done: "var(--amber)",
  outreach: "var(--accent)", outreach_done: "var(--accent)",
  done: "var(--green)",
  failed: "var(--red)",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "Queued",
  enriching: "Enriching...", enriched: "Enriched",
  contacts: "Finding contacts...", contacts_done: "Contacts found",
  outreach: "Generating outreach...", outreach_done: "Outreach ready",
  done: "Complete",
  failed: "Failed",
};

const TYPE_META: Record<string, { label: string; icon: React.ReactNode; badgeClass: string }> = {
  company_first: { label: "Company-First", icon: <Building2 size={12} />, badgeClass: "badge-blue" },
  industry_first: { label: "Industry-First", icon: <Globe size={12} />, badgeClass: "badge-green" },
  report_driven: { label: "Report-Driven", icon: <FileText size={12} />, badgeClass: "badge-muted" },
};

const CAMPAIGN_STATUS_BADGE: Record<string, string> = {
  draft: "badge-blue",
  running: "badge-amber",
  completed: "badge-green",
  failed: "badge-red",
};

const SENIORITY_META: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  decision_maker: { icon: <Shield size={11} />, color: "var(--green)", label: "Decision Maker" },
  influencer: { icon: <Briefcase size={11} />, color: "var(--accent)", label: "Influencer" },
  gatekeeper: { icon: <User size={11} />, color: "var(--amber)", label: "Gatekeeper" },
};

/* ── Sub-components ──────────────────────────────────────── */

function CopyButton({ text, label, size = "sm" }: { text: string; label?: string; size?: "sm" | "md" }) {
  const [copied, setCopied] = useState(false);
  const fontSize = size === "md" ? 12 : 11;

  async function handleCopy(e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* noop */ }
  }

  return (
    <button
      onClick={handleCopy}
      title={label || "Copy"}
      style={{
        display: "inline-flex", alignItems: "center", gap: 4,
        background: copied ? "var(--green-light)" : "var(--surface-raised)",
        color: copied ? "var(--green)" : "var(--text-muted)",
        border: "1px solid var(--border)", borderRadius: 6,
        padding: size === "md" ? "5px 10px" : "3px 7px",
        fontSize, cursor: "pointer", transition: "all 150ms",
      }}
    >
      {copied ? <Check size={fontSize} /> : <Copy size={fontSize} />}
      {label && <span>{copied ? "Copied!" : label}</span>}
    </button>
  );
}

function StatusDot({ status, pulse = false }: { status: string; pulse?: boolean }) {
  const color = STATUS_DOT[status] ?? "var(--text-muted)";
  return (
    <span style={{
      width: 7, height: 7, borderRadius: "50%", display: "inline-block",
      background: color, flexShrink: 0,
      ...(pulse ? { animation: "pulse-dot 1.5s ease-in-out infinite" } : {}),
    }} />
  );
}

/* ── Contact row ──────────────────────────────────────────── */

function ContactRow({ contact, outreach }: { contact: CampaignContact; outreach?: CampaignEmail }) {
  const sm = SENIORITY_META[contact.seniority] ?? { icon: <User size={11} />, color: "var(--text-muted)", label: contact.seniority };
  const conf = contact.email_confidence ?? 0;  // 0-100 integer from backend
  const confColor = conf >= 75 ? "var(--green)" : conf >= 50 ? "var(--accent)" : "var(--text-muted)";

  const mailtoHref = outreach && contact.email
    ? `mailto:${contact.email}?subject=${encodeURIComponent(outreach.subject)}&body=${encodeURIComponent(outreach.body)}`
    : contact.email ? `mailto:${contact.email}` : undefined;

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "180px 1fr 160px auto",
      gap: 10, alignItems: "center",
      padding: "9px 12px",
      borderBottom: "1px solid var(--border)",
      fontSize: 12,
    }}>
      {/* Name + seniority */}
      <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
        <div style={{
          width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
          background: "var(--surface-raised)", color: sm.color,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          {sm.icon}
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 600, color: "var(--text)", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {contact.full_name || "—"}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{contact.role || sm.label}</div>
        </div>
      </div>

      {/* Email */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
        {contact.email ? (
          <>
            <span style={{ color: "var(--text-secondary)", fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {contact.email}
            </span>
            <span style={{
              fontSize: 10, fontWeight: 600, color: confColor, flexShrink: 0,
              padding: "1px 5px", borderRadius: 4,
              background: conf >= 75 ? "var(--green-light)" : conf >= 50 ? "var(--amber-light)" : "var(--surface-raised)",
            }}>
              {Math.round(conf)}%
            </span>
          </>
        ) : (
          <span style={{ color: "var(--text-muted)", fontSize: 11 }}>No email</span>
        )}
      </div>

      {/* Outreach subject preview */}
      <div style={{ color: "var(--text-muted)", fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {outreach?.subject || "—"}
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
        {contact.email && <CopyButton text={contact.email} />}
        {contact.linkedin_url && (
          <a
            href={contact.linkedin_url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={e => e.stopPropagation()}
            style={{
              display: "inline-flex", alignItems: "center", gap: 3,
              padding: "3px 7px", borderRadius: 6, fontSize: 11,
              background: "var(--surface-raised)", color: "var(--text-muted)",
              border: "1px solid var(--border)", textDecoration: "none",
            }}
          >
            <Globe size={10} />
          </a>
        )}
        {mailtoHref && (
          <a
            href={mailtoHref}
            onClick={e => e.stopPropagation()}
            title="Open in email client"
            style={{
              display: "inline-flex", alignItems: "center", gap: 3,
              padding: "3px 7px", borderRadius: 6, fontSize: 11,
              background: "var(--accent-light)", color: "var(--accent)",
              border: "1px solid var(--border)", textDecoration: "none",
            }}
          >
            <Send size={10} />
          </a>
        )}
      </div>
    </div>
  );
}

/* ── Company card ─────────────────────────────────────────── */

function CompanyCard({
  company,
  expanded,
  onToggle,
}: {
  company: CampaignCompanyStatus;
  expanded: boolean;
  onToggle: () => void;
}) {
  const isActive = ["enriching", "contacts", "outreach"].includes(company.status);
  const contacts = company.contacts ?? [];
  const emails = company.emails ?? [];

  // Match each contact to their outreach email by name similarity
  function findEmail(contact: CampaignContact): CampaignEmail | undefined {
    return emails.find(e =>
      e.recipient_name?.toLowerCase() === contact.full_name?.toLowerCase() ||
      e.recipient_role?.toLowerCase() === contact.role?.toLowerCase()
    );
  }

  return (
    <div style={{
      border: `1px solid ${company.status === "failed" ? "var(--red)" : expanded ? "var(--border-strong)" : "var(--border)"}`,
      borderRadius: 10, overflow: "hidden",
      background: "var(--surface)",
      transition: "border-color 200ms",
    }}>
      {/* Header row — always visible */}
      <div
        onClick={onToggle}
        style={{
          display: "flex", alignItems: "center", gap: 12, padding: "12px 16px",
          cursor: "pointer", userSelect: "none",
          background: expanded ? "var(--surface-raised)" : "var(--surface)",
          transition: "background 150ms",
        }}
      >
        {/* Status dot */}
        <StatusDot status={company.status} pulse={isActive} />

        {/* Name + industry */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>
              {company.company_name}
            </span>
            {company.domain && (
              <a
                href={`https://${company.domain}`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={e => e.stopPropagation()}
                style={{ fontSize: 10, color: "var(--blue)", textDecoration: "none" }}
              >
                {company.domain}
              </a>
            )}
          </div>
          {company.industry && (
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1 }}>
              {company.industry}
            </div>
          )}
        </div>

        {/* Stats + actions */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          {contacts.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 11, color: "var(--text-secondary)" }}>
              <Users size={10} style={{ color: "var(--text-muted)" }} />
              <span className="num" style={{ fontWeight: 600 }}>{contacts.length}</span>
            </div>
          )}
          {emails.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 11, color: "var(--text-secondary)" }}>
              <Mail size={10} style={{ color: "var(--text-muted)" }} />
              <span className="num" style={{ fontWeight: 600 }}>{emails.length}</span>
            </div>
          )}

          {/* View Lead link */}
          <Link
            href={`/leads?company=${encodeURIComponent(company.company_name)}`}
            onClick={e => e.stopPropagation()}
            title="View in Leads"
            style={{
              display: "inline-flex", alignItems: "center", gap: 4,
              padding: "3px 8px", borderRadius: 6, fontSize: 11, fontWeight: 500,
              border: "1px solid var(--border)",
              background: "var(--surface-raised)", color: "var(--text-muted)",
              textDecoration: "none",
            }}
          >
            <ExternalLink size={10} /> Lead
          </Link>

          {/* Status label */}
          <span style={{
            fontSize: 10, padding: "2px 8px", borderRadius: 999, fontWeight: 500,
            background: company.status === "done" ? "var(--green-light)" :
              company.status === "failed" ? "var(--red-light)" :
              isActive ? "var(--blue-light)" : "var(--surface-raised)",
            color: company.status === "done" ? "var(--green)" :
              company.status === "failed" ? "var(--red)" :
              isActive ? "var(--blue)" : "var(--text-muted)",
          }}>
            {STATUS_LABEL[company.status] ?? company.status}
          </span>
          <ChevronDown
            size={14}
            style={{
              color: "var(--text-muted)", flexShrink: 0,
              transform: expanded ? "rotate(0deg)" : "rotate(-90deg)",
              transition: "transform 200ms",
            }}
          />
        </div>
      </div>

      {/* Error bar */}
      {company.error && (
        <div style={{
          padding: "6px 16px", background: "var(--red-light)",
          borderTop: "1px solid var(--border)",
          fontSize: 11, color: "var(--red)", display: "flex", alignItems: "center", gap: 6,
        }}>
          <AlertCircle size={11} /> {company.error}
        </div>
      )}

      {/* Expanded content */}
      {expanded && (
        <div style={{ borderTop: "1px solid var(--border)" }}>
          {/* Description */}
          {company.description && (
            <div style={{
              padding: "12px 16px",
              fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6,
              borderBottom: contacts.length > 0 || emails.length > 0 ? "1px solid var(--border)" : "none",
            }}>
              {company.description}
            </div>
          )}

          {/* Contacts table — outreach subject shown inline per row */}
          {contacts.length > 0 && (
            <div>
              <div style={{
                padding: "7px 12px 5px",
                fontSize: 10, fontWeight: 700, color: "var(--text-muted)",
                textTransform: "uppercase", letterSpacing: "0.06em",
                borderBottom: "1px solid var(--border)",
                display: "grid",
                gridTemplateColumns: "180px 1fr 160px auto",
                gap: 10,
              }}>
                <span>Contact</span>
                <span>Email</span>
                <span>Subject</span>
                <span />
              </div>
              {contacts.map((ct, i) => (
                <ContactRow key={i} contact={ct} outreach={findEmail(ct)} />
              ))}
            </div>
          )}

          {/* Empty state */}
          {!company.description && contacts.length === 0 && emails.length === 0 && !company.error && (
            <div style={{
              padding: "20px 16px", textAlign: "center",
              fontSize: 12, color: "var(--text-muted)",
            }}>
              {isActive ? "Processing in progress..." : "No data collected"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Progress header strip ───────────────────────────────── */

function ProgressStrip({ campaign, progressText }: { campaign: Campaign; progressText: string }) {
  const pct = campaign.total_companies > 0
    ? Math.round((campaign.completed_companies / campaign.total_companies) * 100)
    : 0;
  const isRunning = campaign.status === "running";

  return (
    <div style={{
      padding: "10px 24px",
      background: isRunning ? "var(--blue-light)" : "var(--surface-raised)",
      borderBottom: "1px solid var(--border)",
      display: "flex", alignItems: "center", gap: 12,
    }}>
      {isRunning && <Loader2 size={13} style={{ color: "var(--blue)", animation: "spin 1s linear infinite", flexShrink: 0 }} />}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: isRunning ? "var(--blue)" : "var(--text-secondary)" }}>
            {progressText || (isRunning ? "Running..." : `${pct}% complete`)}
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {campaign.completed_companies}/{campaign.total_companies} companies
          </span>
        </div>
        <div style={{ height: 5, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
          <div style={{
            height: "100%", borderRadius: 3,
            width: isRunning && pct === 0 ? "100%" : `${pct}%`,
            background: campaign.status === "failed" ? "var(--red)" :
              campaign.status === "completed" ? "var(--green)" : "var(--blue)",
            transition: "width 400ms ease",
            ...(isRunning && pct === 0 ? { animation: "progress-indeterminate 1.4s ease-in-out infinite" } : {}),
          }} />
        </div>
      </div>
    </div>
  );
}

/* ── Delete confirmation modal ───────────────────────────── */

function DeleteModal({ name, onConfirm, onCancel }: {
  name: string; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)",
      }}
      onClick={onCancel}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: "var(--surface)", borderRadius: 14, padding: "24px 28px",
          width: "min(420px, 90vw)", border: "1px solid var(--border)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
          <div style={{
            width: 36, height: 36, borderRadius: "50%",
            background: "var(--red-light)", color: "var(--red)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Trash2 size={17} />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)" }}>Delete Campaign</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>This action cannot be undone</div>
          </div>
        </div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 20 }}>
          Are you sure you want to delete <strong>{name}</strong>?
          All contacts, emails, and results will be permanently removed.
        </div>
        <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
          <button onClick={onCancel} style={{
            padding: "8px 18px", fontSize: 13, fontWeight: 500, borderRadius: 8,
            border: "1px solid var(--border)", background: "var(--surface)",
            color: "var(--text-secondary)", cursor: "pointer",
          }}>
            Cancel
          </button>
          <button onClick={onConfirm} style={{
            padding: "8px 18px", fontSize: 13, fontWeight: 600, borderRadius: 8,
            border: "none", background: "var(--red)", color: "#fff", cursor: "pointer",
          }}>
            Delete Campaign
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Main page ────────────────────────────────────────────── */

export default function CampaignDetailPage() {
  const params = useParams();
  const router = useRouter();
  const campaignId = params.id as string;

  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [progressText, setProgressText] = useState("");
  const [expandedCompanies, setExpandedCompanies] = useState<Set<string>>(new Set());
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  /* ── Load ── */

  const loadCampaign = useCallback(async () => {
    try {
      const data = await api.getCampaign(campaignId);
      setCampaign(data);
      if (data.status === "completed") {
        // Auto-expand the first successfully completed company
        const firstDone = data.companies.find(
          c => c.status === "done" && ((c.contacts?.length ?? 0) > 0 || c.contacts_found > 0),
        );
        if (firstDone) setExpandedCompanies(new Set([firstDone.company_name]));
      }
      if (data.status === "running") startStream(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load campaign");
    } finally {
      setLoading(false);
    }
  }, [campaignId]);

  useEffect(() => {
    loadCampaign();
    return () => { cleanupRef.current?.(); };
  }, [loadCampaign]);

  /* ── Streaming ── */

  function startStream(_campaign: Campaign) {
    cleanupRef.current?.();
    setStreaming(true);

    const cleanup = api.streamCampaign(
      campaignId,
      (event: CampaignStreamEvent) => {
        setCampaign(prev => {
          if (!prev) return prev;
          const updated = { ...prev, companies: [...prev.companies] };

          switch (event.event) {
            case "campaign_start":
              updated.status = "running";
              break;
            case "company_start":
              if (event.company) {
                const idx = updated.companies.findIndex(c => c.company_name === event.company);
                if (idx >= 0) updated.companies[idx] = { ...updated.companies[idx], status: "enriching" };
              }
              if (event.index !== undefined && event.total !== undefined)
                setProgressText(`Processing ${event.index + 1} of ${event.total}...`);
              break;
            case "company_enriched":
              if (event.company) {
                const idx = updated.companies.findIndex(c => c.company_name === event.company);
                if (idx >= 0) updated.companies[idx] = {
                  ...updated.companies[idx], status: "contacts",
                  domain: event.domain ?? updated.companies[idx].domain,
                  industry: event.industry ?? updated.companies[idx].industry,
                };
              }
              break;
            case "company_contacts":
              if (event.company) {
                const idx = updated.companies.findIndex(c => c.company_name === event.company);
                if (idx >= 0) updated.companies[idx] = {
                  ...updated.companies[idx], status: "outreach",
                  contacts_found: event.contacts_found ?? updated.companies[idx].contacts_found,
                };
              }
              break;
            case "company_done":
              if (event.company) {
                const idx = updated.companies.findIndex(c => c.company_name === event.company);
                if (idx >= 0) updated.companies[idx] = { ...updated.companies[idx], status: "done" };
                updated.completed_companies = updated.companies.filter(
                  c => c.status === "done" || c.status === "failed",
                ).length;
              }
              break;
            case "company_error":
              if (event.company) {
                const idx = updated.companies.findIndex(c => c.company_name === event.company);
                if (idx >= 0) updated.companies[idx] = {
                  ...updated.companies[idx], status: "failed",
                  error: event.error ?? "Unknown error",
                };
                updated.completed_companies = updated.companies.filter(
                  c => c.status === "done" || c.status === "failed",
                ).length;
              }
              break;
            case "campaign_done":
              updated.status = "completed";
              updated.total_contacts = event.total_contacts ?? updated.total_contacts;
              updated.total_outreach = event.total_outreach ?? updated.total_outreach;
              break;
          }
          return updated;
        });
      },
      () => {
        setStreaming(false);
        setRunning(false);
        setProgressText("");
        loadCampaign();
      },
      (errMsg: string) => {
        setStreaming(false);
        setRunning(false);
        setError(errMsg);
      },
    );
    cleanupRef.current = cleanup;
  }

  /* ── Actions ── */

  async function handleRun() {
    if (!campaign) return;
    setRunning(true);
    setError(null);
    try {
      await api.runCampaign(campaignId);
      setCampaign(prev => prev ? { ...prev, status: "running" } : prev);
      startStream(campaign);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start campaign");
      setRunning(false);
    }
  }

  async function handleDelete() {
    try {
      await api.deleteCampaign(campaignId);
      router.push("/campaigns");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete campaign");
      setShowDeleteModal(false);
    }
  }

  function toggleCompany(name: string) {
    setExpandedCompanies(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  }

  function expandAll() {
    if (campaign) setExpandedCompanies(new Set(campaign.companies.map(c => c.company_name)));
  }

  function collapseAll() {
    setExpandedCompanies(new Set());
  }

  /* ── Derived ── */

  const name = campaign?.name || (campaign ? `Campaign ${campaign.id.slice(0, 8)}` : "Campaign");
  const typeMeta = campaign ? (TYPE_META[campaign.campaign_type] ?? TYPE_META.company_first) : null;
  const allContacts = campaign?.companies.flatMap(c => c.contacts ?? []) ?? [];
  const allEmails = campaign?.companies.flatMap(c => c.emails ?? []) ?? [];
  const isRunning = campaign?.status === "running";
  const isDraft = campaign?.status === "draft";
  const isFailed = campaign?.status === "failed";
  const isCompleted = campaign?.status === "completed";

  const showProgress = isRunning || (campaign && campaign.total_companies > 0);

  /* ── Render ── */

  if (loading) {
    return (
      <>
        <div style={{ padding: "14px 24px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
          <div className="skeleton" style={{ height: 18, width: 220 }} />
        </div>
        <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 12 }}>
          {[0, 1, 2].map(i => (
            <div key={i} className="card" style={{ padding: "14px 16px" }}>
              <div className="skeleton" style={{ height: 14, width: "50%", marginBottom: 8 }} />
              <div className="skeleton" style={{ height: 11, width: "30%" }} />
            </div>
          ))}
        </div>
      </>
    );
  }

  if (!campaign) {
    return (
      <div style={{ padding: "48px 24px", textAlign: "center" }}>
        <div style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 12 }}>Campaign not found</div>
        <Link href="/campaigns" style={{ fontSize: 13, color: "var(--accent)", textDecoration: "none" }}>
          Back to Campaigns
        </Link>
      </div>
    );
  }

  return (
    <>
      {showDeleteModal && (
        <DeleteModal
          name={name}
          onConfirm={handleDelete}
          onCancel={() => setShowDeleteModal(false)}
        />
      )}

      {/* ── Header ── */}
      <div style={{
        padding: "13px 24px",
        borderBottom: "1px solid var(--border)",
        background: "var(--surface)", flexShrink: 0,
        display: "flex", alignItems: "center", gap: 12,
      }}>
        <Link
          href="/campaigns"
          style={{
            display: "flex", alignItems: "center", gap: 5,
            fontSize: 12, color: "var(--text-muted)", textDecoration: "none",
            padding: "5px 8px", borderRadius: 6,
            border: "1px solid var(--border)", background: "var(--surface-raised)",
          }}
        >
          <ArrowLeft size={12} /> Campaigns
        </Link>

        <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center", gap: 10 }}>
          <h1
            className="font-display"
            style={{
              fontSize: 17, color: "var(--text)", letterSpacing: "-0.02em",
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}
          >
            {name}
          </h1>
          {typeMeta && (
            <span
              className={`badge ${typeMeta.badgeClass}`}
              style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 4, flexShrink: 0 }}
            >
              {typeMeta.icon} {typeMeta.label}
            </span>
          )}
          <span
            className={`badge ${CAMPAIGN_STATUS_BADGE[campaign.status] ?? "badge-muted"}`}
            style={{ fontSize: 10, flexShrink: 0 }}
          >
            {campaign.status}
          </span>
        </div>

        {/* Header actions */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          {/* Aggregate stats */}
          {isCompleted && (
            <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 12, color: "var(--text-secondary)" }}>
              <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <Users size={11} style={{ color: "var(--text-muted)" }} />
                <span className="num" style={{ fontWeight: 600 }}>{campaign.total_contacts}</span> contacts
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <Mail size={11} style={{ color: "var(--text-muted)" }} />
                <span className="num" style={{ fontWeight: 600 }}>{campaign.total_outreach}</span> emails
              </span>
            </div>
          )}

          {/* Export */}
          {isCompleted && (
            <a
              href={api.getCampaignExportUrl(campaignId)}
              download
              style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                padding: "7px 13px", borderRadius: 7, fontSize: 12, fontWeight: 500,
                border: "1px solid var(--border)", background: "var(--surface)",
                color: "var(--text-secondary)", textDecoration: "none",
                cursor: "pointer",
              }}
            >
              <Download size={12} /> Export CSV
            </a>
          )}

          {/* Run / Refresh */}
          {(isDraft || isFailed) && (
            <button
              onClick={handleRun}
              disabled={running}
              style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                padding: "7px 14px", borderRadius: 7, fontSize: 12, fontWeight: 600,
                background: running ? "var(--surface-raised)" : "var(--green)",
                color: running ? "var(--text-muted)" : "#fff",
                border: "none", cursor: running ? "not-allowed" : "pointer",
              }}
            >
              {running
                ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} />
                : <Play size={12} />
              }
              {running ? "Starting..." : isDraft ? "Run Campaign" : "Re-run"}
            </button>
          )}

          {isRunning && (
            <button
              onClick={() => loadCampaign()}
              style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "7px 12px", borderRadius: 7, fontSize: 12,
                border: "1px solid var(--border)", background: "var(--surface)",
                color: "var(--text-secondary)", cursor: "pointer",
              }}
            >
              <RefreshCw size={11} /> Refresh
            </button>
          )}

          {/* Delete */}
          <button
            onClick={() => setShowDeleteModal(true)}
            title="Delete campaign"
            style={{
              padding: "7px 9px", borderRadius: 7, fontSize: 12,
              border: "1px solid var(--border)", background: "var(--surface)",
              color: "var(--text-muted)", cursor: "pointer", display: "flex",
            }}
          >
            <Trash2 size={13} />
          </button>
        </div>
      </div>

      {/* ── Progress strip (visible when running or has companies) ── */}
      {showProgress && (
        <ProgressStrip campaign={campaign} progressText={progressText} />
      )}

      {/* ── Body ── */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>

        {/* Error banner */}
        {error && (
          <div style={{
            padding: "10px 14px", background: "var(--red-light)", color: "var(--red)",
            borderRadius: 8, fontSize: 12, marginBottom: 16,
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <AlertCircle size={13} /> {error}
            </span>
            <button onClick={() => setError(null)} style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 16 }}>×</button>
          </div>
        )}

        {/* Campaign error */}
        {campaign.error && (
          <div style={{
            padding: "10px 14px", background: "var(--red-light)", color: "var(--red)",
            borderRadius: 8, fontSize: 12, marginBottom: 16,
            display: "flex", alignItems: "center", gap: 6,
          }}>
            <AlertCircle size={13} /> {campaign.error}
          </div>
        )}

        {/* Companies section */}
        {campaign.companies.length > 0 && (
          <>
            {/* Section header */}
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{
                  fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
                  textTransform: "uppercase", letterSpacing: "0.07em",
                }}>
                  Companies
                </span>
                <span style={{
                  fontSize: 10, padding: "2px 7px", borderRadius: 999,
                  background: "var(--surface-raised)", color: "var(--text-muted)",
                  border: "1px solid var(--border)",
                }}>
                  {campaign.companies.length}
                </span>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={expandAll}
                  style={{
                    fontSize: 11, color: "var(--text-muted)", background: "none", border: "none",
                    cursor: "pointer", padding: "3px 6px", borderRadius: 4,
                  }}
                >
                  Expand all
                </button>
                <button
                  onClick={collapseAll}
                  style={{
                    fontSize: 11, color: "var(--text-muted)", background: "none", border: "none",
                    cursor: "pointer", padding: "3px 6px", borderRadius: 4,
                  }}
                >
                  Collapse all
                </button>
              </div>
            </div>

            {/* Company cards */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
              {campaign.companies.map(company => (
                <CompanyCard
                  key={company.company_name}
                  company={company}
                  expanded={expandedCompanies.has(company.company_name)}
                  onToggle={() => toggleCompany(company.company_name)}
                />
              ))}
            </div>
          </>
        )}

        {/* Empty state for draft with no companies */}
        {campaign.companies.length === 0 && isDraft && (
          <div style={{
            padding: "56px 24px", textAlign: "center",
            background: "var(--surface-raised)", borderRadius: 12,
            border: "1px dashed var(--border)",
          }}>
            <Building2 size={28} style={{ margin: "0 auto 12px", color: "var(--text-muted)", opacity: 0.4 }} />
            <div style={{ fontSize: 14, fontWeight: 500, color: "var(--text-secondary)", marginBottom: 6 }}>
              Ready to run
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 18, lineHeight: 1.6 }}>
              Click Run Campaign to start enriching companies and generating outreach.
            </div>
            <button
              onClick={handleRun}
              disabled={running}
              style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                padding: "9px 20px", borderRadius: 8, fontSize: 13, fontWeight: 600,
                background: "var(--green)", color: "#fff",
                border: "none", cursor: "pointer",
              }}
            >
              <Play size={13} /> Run Campaign
            </button>
          </div>
        )}

        {/* Campaign metadata footer */}
        <div style={{
          display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap",
          fontSize: 11, color: "var(--text-muted)", paddingTop: 8,
          borderTop: campaign.companies.length > 0 ? "1px solid var(--border)" : "none",
        }}>
          <span>Created {formatDate(campaign.created_at, "medium")}</span>
          {campaign.completed_at && (
            <span>Completed {formatDate(campaign.completed_at, "medium")}</span>
          )}
          <span style={{ marginLeft: "auto", fontFamily: "monospace", fontSize: 10, color: "var(--text-xmuted)" }}>
            {campaign.id}
          </span>
        </div>
      </div>
    </>
  );
}
