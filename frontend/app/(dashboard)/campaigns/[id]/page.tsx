"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  Play, ArrowLeft, Building2, Users, Mail, Loader2,
  CheckCircle, XCircle, Clock, RefreshCw, Globe, AlertCircle,
  ChevronDown, ChevronRight, Copy, Check, Download, ExternalLink,
  User, Briefcase, Shield, FileText, Eye, Send, Pencil, Trash2,
  X, Save, Plus, AlertTriangle,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import type {
  Campaign, CampaignCompanyStatus, CampaignStreamEvent,
  CampaignContact, CampaignEmail, CampaignCompanyInput, LeadRecord,
} from "@/lib/types";
/* ── Status maps ─────────────────────────────────────── */

const STATUS_DOT: Record<string, string> = {
  pending: "var(--text-muted)", enriching: "var(--blue)", enriched: "var(--blue)",
  contacts: "var(--amber)", contacts_done: "var(--amber)",
  outreach: "var(--accent)", outreach_done: "var(--accent)",
  done: "var(--green)", failed: "var(--red)",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "Queued", enriching: "Enriching...", enriched: "Enriched",
  contacts: "Finding contacts...", contacts_done: "Contacts found",
  outreach: "Generating outreach...", outreach_done: "Outreach ready",
  done: "Complete", failed: "Failed",
};

const SENIORITY_COLORS: Record<string, { icon: React.ReactNode; color: string }> = {
  decision_maker: { icon: <Shield size={11} />, color: "var(--green)" },
  influencer: { icon: <Briefcase size={11} />, color: "var(--accent)" },
  gatekeeper: { icon: <User size={11} />, color: "var(--amber)" },
};

/* ── Helpers ──────────────────────────────────────────── */

function companyHash(name: string): string {
  // Simple hash matching Python: hashlib.md5(name.lower().encode()).hexdigest()[:12]
  // Uses a JS MD5 equivalent via simple hash for URL generation
  let hash = 0;
  const str = name.toLowerCase();
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  // Return the URL-encoded company name for lookup (backend resolves both hash and name)
  return encodeURIComponent(name);
}

function CopyButton({ text, label, size = "sm" }: { text: string; label?: string; size?: "sm" | "md" }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* noop */ }
  };
  const pad = size === "md" ? "6px 12px" : "4px 8px";
  const fs = size === "md" ? 12 : 11;
  return (
    <button onClick={handleCopy} title={label || "Copy"} style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      background: copied ? "var(--green-light)" : "var(--surface-raised)",
      color: copied ? "var(--green)" : "var(--text-muted)",
      border: "1px solid var(--border)", borderRadius: 6,
      padding: pad, fontSize: fs, cursor: "pointer", transition: "all 150ms ease",
    }}>
      {copied ? <Check size={fs} /> : <Copy size={fs} />}
      {label && <span>{copied ? "Copied!" : label}</span>}
    </button>
  );
}

/* ── Delete Confirmation Modal ────────────────────────── */

function DeleteModal({ campaignName, onConfirm, onCancel }: {
  campaignName: string; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      display: "flex", alignItems: "center", justifyContent: "center",
      background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)",
    }} onClick={onCancel}>
      <div onClick={e => e.stopPropagation()} style={{
        background: "var(--surface)", borderRadius: 14, padding: "24px 28px",
        width: "min(420px, 90vw)", border: "1px solid var(--border)",
        boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
          <div style={{
            width: 36, height: 36, borderRadius: "50%", display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "var(--red-light)", color: "var(--red)",
          }}>
            <Trash2 size={18} />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)" }}>Delete Campaign</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>This action cannot be undone</div>
          </div>
        </div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 20 }}>
          Are you sure you want to delete <strong>{campaignName}</strong>?
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

/* ── Main page ───────────────────────────────────────── */

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
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editCompanies, setEditCompanies] = useState("");
  const [editIndustry, setEditIndustry] = useState("");
  const [saving, setSaving] = useState(false);
  const [sendState, setSendState] = useState<Record<string, { loading: boolean; result?: { success: boolean; recipient: string; test_mode: boolean; error: string }; error?: string }>>({});
  const [campaignLeads, setCampaignLeads] = useState<LeadRecord[]>([]);
  const cleanupRef = useRef<(() => void) | null>(null);

  const loadCampaign = useCallback(async () => {
    try {
      const data = await api.getCampaign(campaignId);
      setCampaign(data);
      if (data.status === "completed") {
        const firstDone = data.companies.find(c =>
          c.status === "done" && ((c.contacts?.length ?? 0) > 0 || (c.emails?.length ?? 0) > 0 || c.contacts_found > 0 || c.description),
        );
        if (firstDone) setExpandedCompanies(new Set([firstDone.company_name]));
        // Fetch latest leads and filter by campaign company names
        try {
          const leadsResp = await api.getLatestLeads(100);
          const campaignNames = new Set(data.companies.map(c => c.company_name.toLowerCase()));
          const filtered = leadsResp.leads.filter(l => campaignNames.has(l.company_name.toLowerCase()));
          setCampaignLeads(filtered.length > 0 ? filtered : leadsResp.leads.slice(0, 20));
        } catch { /* non-critical */ }
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

  function startStream(current: Campaign) {
    cleanupRef.current?.();
    setStreaming(true);
    setProgressText("");

    const cleanup = api.streamCampaign(
      campaignId,
      (event: CampaignStreamEvent) => {
        setCampaign(prev => {
          if (!prev) return prev;
          const updated = { ...prev, companies: [...prev.companies] };
          switch (event.event) {
            case "campaign_start": updated.status = "running"; break;
            case "company_start":
              if (event.company) {
                const idx = updated.companies.findIndex(c => c.company_name === event.company);
                if (idx >= 0) updated.companies[idx] = { ...updated.companies[idx], status: "enriching" };
              }
              if (event.index !== undefined && event.total !== undefined)
                setProgressText(`Processing ${event.index + 1}/${event.total}...`);
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
                updated.completed_companies = updated.companies.filter(c => c.status === "done" || c.status === "failed").length;
              }
              break;
            case "company_error":
              if (event.company) {
                const idx = updated.companies.findIndex(c => c.company_name === event.company);
                if (idx >= 0) updated.companies[idx] = { ...updated.companies[idx], status: "failed", error: event.error ?? "Unknown error" };
                updated.completed_companies = updated.companies.filter(c => c.status === "done" || c.status === "failed").length;
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
      () => { setStreaming(false); setRunning(false); setProgressText(""); loadCampaign(); },
      (errMsg: string) => { setStreaming(false); setRunning(false); setError(errMsg); },
    );
    cleanupRef.current = cleanup;
  }

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
      setError(err instanceof Error ? err.message : "Failed to delete");
      setShowDeleteModal(false);
    }
  }

  function startEdit() {
    if (!campaign) return;
    setEditName(campaign.name);
    setEditCompanies(campaign.companies.map(c => c.company_name).join("\n"));
    setEditIndustry("");
    setEditing(true);
  }

  async function handleSave() {
    if (!campaign) return;
    setSaving(true);
    try {
      const updates: Parameters<typeof api.updateCampaign>[1] = {};
      if (editName.trim() !== campaign.name) updates.name = editName.trim();
      if (campaign.campaign_type === "company_first") {
        const lines = editCompanies.split("\n").map(l => l.trim()).filter(Boolean);
        updates.companies = lines.map(name => ({ company_name: name }));
      }
      const saved = await api.updateCampaign(campaignId, updates);
      setCampaign(saved);
      setEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save changes");
    }
    setSaving(false);
  }

  async function handleSendEmail(companyName: string, email: CampaignEmail) {
    const key = `${companyName}-${email.recipient_name}`;
    setSendState(prev => ({ ...prev, [key]: { loading: true } }));
    try {
      const result = await api.sendCampaignEmail(campaignId, {
        company_name: companyName,
        recipient_name: email.recipient_name,
        recipient_email: "", // Will be matched from contacts
        subject: email.subject,
        body: email.body,
      });
      setSendState(prev => ({ ...prev, [key]: { loading: false, result } }));
    } catch (err) {
      setSendState(prev => ({ ...prev, [key]: { loading: false, error: err instanceof Error ? err.message : "Send failed" } }));
    }
  }

  async function handleSendEmailDirect(companyName: string, contact: CampaignContact, email: CampaignEmail) {
    const key = `${companyName}-${contact.full_name}`;
    setSendState(prev => ({ ...prev, [key]: { loading: true } }));
    try {
      const result = await api.sendCampaignEmail(campaignId, {
        company_name: companyName,
        recipient_name: contact.full_name,
        recipient_email: contact.email || "",  // Backend falls back to test recipient
        subject: email.subject,
        body: email.body,
      });
      setSendState(prev => ({ ...prev, [key]: { loading: false, result } }));
    } catch (err) {
      setSendState(prev => ({ ...prev, [key]: { loading: false, error: err instanceof Error ? err.message : "Send failed" } }));
    }
  }

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "80px 0", color: "var(--text-muted)", gap: 8 }}>
      <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} />
      <span style={{ fontSize: 13 }}>Loading campaign...</span>
    </div>
  );

  if (!campaign) return (
    <div style={{ padding: "60px 24px", textAlign: "center" }}>
      <AlertCircle size={28} style={{ color: "var(--red)", margin: "0 auto 10px" }} />
      <div style={{ color: "var(--red)", fontSize: 13, marginBottom: 12 }}>{error || "Campaign not found"}</div>
      <Link href="/campaigns" style={{ fontSize: 13, color: "var(--accent)", textDecoration: "none" }}>Back to Campaigns</Link>
    </div>
  );

  const canRun = campaign.status === "draft" || campaign.status === "failed";
  const canEdit = campaign.status === "draft" || campaign.status === "failed";
  const isActive = campaign.status === "running" || streaming;
  const completedCount = campaign.companies.filter(c => c.status === "done").length;
  const failedCount = campaign.companies.filter(c => c.status === "failed").length;
  const hasResults = campaign.companies.some(c => (c.contacts?.length ?? 0) > 0 || (c.emails?.length ?? 0) > 0);
  const progressPct = campaign.total_companies > 0 ? Math.round((campaign.completed_companies / campaign.total_companies) * 100) : 0;

  // Flatten all contacts/emails for summary
  const allContacts = campaign.companies.flatMap(c => (c.contacts || []).map(ct => ({ ...ct, company: c.company_name, domain: c.domain })));
  const allEmails = campaign.companies.flatMap(c => (c.emails || []).map(em => ({ ...em, company: c.company_name })));

  return (
    <>
      {showDeleteModal && (
        <DeleteModal
          campaignName={campaign.name || `Campaign ${campaign.id.slice(0, 8)}`}
          onConfirm={handleDelete}
          onCancel={() => setShowDeleteModal(false)}
        />
      )}

      {/* ── Header ───────────────────────────────────── */}
      <div style={{ padding: "16px 24px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, flex: 1, minWidth: 0 }}>
            <Link href="/campaigns" style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-muted)", fontSize: 12, textDecoration: "none", flexShrink: 0 }}>
              <ArrowLeft size={14} /> Back
            </Link>
            {editing ? (
              <input value={editName} onChange={e => setEditName(e.target.value)}
                style={{ fontSize: 16, fontWeight: 600, color: "var(--text)", background: "var(--surface-raised)", border: "1px solid var(--accent)", borderRadius: 6, padding: "4px 10px", flex: 1, outline: "none" }}
                autoFocus
              />
            ) : (
              <h1 className="font-display" style={{ fontSize: 18, color: "var(--text)", letterSpacing: "-0.02em", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {campaign.name || `Campaign ${campaign.id.slice(0, 8)}`}
              </h1>
            )}
            <span className={`badge badge-${campaign.status === "completed" ? "green" : campaign.status === "running" ? "amber" : campaign.status === "failed" ? "red" : "blue"}`} style={{ fontSize: 10, flexShrink: 0 }}>
              {campaign.status}
            </span>
            <span className="badge badge-muted" style={{ fontSize: 10, flexShrink: 0 }}>
              {campaign.campaign_type.replace(/_/g, " ")}
            </span>
          </div>
          <div style={{ display: "flex", gap: 6, flexShrink: 0, alignItems: "center" }}>
            {isActive && (
              <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--amber)" }}>
                <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} />
                {progressText || "Running..."}
              </div>
            )}
            {editing ? (
              <>
                <button onClick={() => setEditing(false)} style={{ display: "flex", alignItems: "center", gap: 4, background: "var(--surface-raised)", color: "var(--text-muted)", borderRadius: 8, padding: "7px 14px", fontSize: 12, border: "1px solid var(--border)", cursor: "pointer" }}>
                  <X size={13} /> Cancel
                </button>
                <button onClick={handleSave} disabled={saving} style={{ display: "flex", alignItems: "center", gap: 4, background: "var(--accent)", color: "#fff", borderRadius: 8, padding: "7px 14px", fontSize: 12, fontWeight: 500, border: "none", cursor: saving ? "not-allowed" : "pointer", opacity: saving ? 0.7 : 1 }}>
                  {saving ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} /> : <Save size={13} />}
                  Save
                </button>
              </>
            ) : (
              <>
                {canEdit && (
                  <button onClick={startEdit} style={{ display: "flex", alignItems: "center", gap: 4, background: "var(--surface-raised)", color: "var(--text-secondary)", borderRadius: 8, padding: "7px 14px", fontSize: 12, border: "1px solid var(--border)", cursor: "pointer" }}>
                    <Pencil size={13} /> Edit
                  </button>
                )}
                {canRun && (
                  <button onClick={handleRun} disabled={running} style={{ display: "flex", alignItems: "center", gap: 6, background: "var(--green)", color: "#fff", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, border: "none", cursor: running ? "not-allowed" : "pointer", opacity: running ? 0.7 : 1 }}>
                    {running ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Play size={14} />}
                    {campaign.status === "failed" ? "Retry" : "Run Campaign"}
                  </button>
                )}
                {campaign.status === "completed" && (
                  <button onClick={loadCampaign} style={{ display: "flex", alignItems: "center", gap: 4, background: "var(--surface-raised)", color: "var(--text-secondary)", borderRadius: 8, padding: "7px 12px", fontSize: 12, border: "1px solid var(--border)", cursor: "pointer" }}>
                    <RefreshCw size={13} />
                  </button>
                )}
                {canEdit && (
                  <button onClick={() => setShowDeleteModal(true)} style={{ display: "flex", alignItems: "center", gap: 4, background: "none", color: "var(--red)", borderRadius: 8, padding: "7px 10px", fontSize: 12, border: "1px solid var(--red)33", cursor: "pointer" }}>
                    <Trash2 size={13} />
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        {error && (
          <div style={{ padding: "10px 14px", background: "var(--red-light)", color: "var(--red)", borderRadius: 8, fontSize: 12, marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            {error}
            <button onClick={() => setError(null)} style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 14, fontWeight: 600 }}>&times;</button>
          </div>
        )}

        {/* ── Edit mode: company list editor ──── */}
        {editing && campaign.campaign_type === "company_first" && (
          <div className="card" style={{ padding: 18, marginBottom: 18 }}>
            <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: 6 }}>
              Companies (one per line)
            </label>
            <textarea value={editCompanies} onChange={e => setEditCompanies(e.target.value)}
              rows={Math.max(4, editCompanies.split("\n").length + 1)}
              style={{ width: "100%", padding: "10px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 13, color: "var(--text)", outline: "none", resize: "vertical", fontFamily: "inherit", lineHeight: 1.7 }}
            />
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
              {editCompanies.split("\n").filter(l => l.trim()).length} companies
            </div>
          </div>
        )}

        {/* ── Stats bar ─────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10, marginBottom: 18 }}>
          <StatCard icon={<Building2 size={15} />} value={campaign.total_companies} label="Companies" color="var(--blue)" bg="var(--blue-light)" />
          <StatCard icon={completedCount === campaign.total_companies && campaign.total_companies > 0 ? <CheckCircle size={15} /> : <Clock size={15} />} value={completedCount} label="Done" color="var(--green)" bg="var(--green-light)" />
          {failedCount > 0 && <StatCard icon={<XCircle size={15} />} value={failedCount} label="Failed" color="var(--red)" bg="var(--red-light)" />}
          <StatCard icon={<Users size={15} />} value={campaign.total_contacts || allContacts.length} label="Contacts" color="var(--amber)" bg="var(--amber-light)" />
          <StatCard icon={<Mail size={15} />} value={campaign.total_outreach || allEmails.length} label="Emails" color="var(--accent)" bg="var(--accent-light)" />
        </div>

        {/* ── Progress bar ────────────────────── */}
        {campaign.total_companies > 0 && (
          <div style={{ marginBottom: 18 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "var(--text-secondary)" }}>Progress</span>
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{campaign.completed_companies}/{campaign.total_companies} ({progressPct}%)</span>
            </div>
            <div style={{ height: 5, background: "var(--surface-raised)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${progressPct}%`, borderRadius: 3,
                background: campaign.status === "failed" ? "var(--red)" : campaign.status === "completed" ? "var(--green)" : "var(--accent)",
                transition: "width 400ms ease",
              }} />
            </div>
          </div>
        )}

        {/* ── Companies accordion ─────────────── */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>Companies ({campaign.companies.length})</span>
            {campaign.companies.length > 1 && (
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={expandAll} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: "var(--accent)" }}>Expand All</button>
                <button onClick={() => setExpandedCompanies(new Set())} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: "var(--text-muted)" }}>Collapse All</button>
              </div>
            )}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {campaign.companies.length === 0 ? (
              <div style={{ padding: "40px 24px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>No companies in this campaign.</div>
            ) : campaign.companies.map((company, i) => (
              <CompanyAccordion
                key={`${company.company_name}-${i}`}
                company={company}
                campaignId={campaignId}
                isActive={isActive}
                expanded={expandedCompanies.has(company.company_name)}
                onToggle={() => toggleCompany(company.company_name)}
                sendState={sendState}
                onSendEmail={handleSendEmailDirect}
              />
            ))}
          </div>
        </div>

        {/* ── Completion summary ──────────────── */}
        {campaign.status === "completed" && (
          <div style={{ padding: "18px 22px", background: "var(--green-light)", borderRadius: 10, border: "1px solid var(--green)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <CheckCircle size={16} style={{ color: "var(--green)" }} />
              <span style={{ fontSize: 14, fontWeight: 600, color: "var(--green)" }}>Campaign Complete</span>
            </div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 12 }}>
              Processed {campaign.total_companies} companies. Found <strong>{campaign.total_contacts || allContacts.length} contacts</strong> and generated <strong>{campaign.total_outreach || allEmails.length} outreach emails</strong>.
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {hasResults}
              <Link href="/companies" style={{ display: "inline-flex", alignItems: "center", gap: 5, background: "var(--surface)", color: "var(--text-secondary)", borderRadius: 7, padding: "7px 14px", fontSize: 12, fontWeight: 500, textDecoration: "none", border: "1px solid var(--border)" }}>
                <Building2 size={12} /> Companies
              </Link>
              <Link href="/leads" style={{ display: "inline-flex", alignItems: "center", gap: 5, background: "var(--surface)", color: "var(--text-secondary)", borderRadius: 7, padding: "7px 14px", fontSize: 12, fontWeight: 500, textDecoration: "none", border: "1px solid var(--border)" }}>
                <Eye size={12} /> All Leads
              </Link>
            </div>
          </div>
        )}

        {/* ── Inline leads grid ────────────────── */}
        {campaign.status === "completed" && campaignLeads.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-xmuted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10, display: "flex", alignItems: "center", gap: 5 }}>
              <Eye size={11} style={{ color: "var(--accent)" }} />
              Pipeline Leads ({campaignLeads.length})
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 8 }}>
              {campaignLeads.map((lead, i) => (
                <Link key={lead.id ?? i} href={`/leads/${lead.id}`} style={{ textDecoration: "none" }}>
                  <div style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
                    borderRadius: 9, border: "1px solid var(--border)", background: "var(--surface)",
                    cursor: "pointer", transition: "all 120ms",
                  }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; (e.currentTarget as HTMLElement).style.borderColor = "var(--accent)44"; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "var(--surface)"; (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; }}
                  >
                    <div style={{ width: 32, height: 32, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--accent-light)", fontSize: 11, fontWeight: 800, color: "var(--accent)" }}>
                      {lead.company_name.slice(0, 2).toUpperCase()}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {lead.company_name}
                      </div>
                      {lead.contact_name && (
                        <div style={{ fontSize: 10, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {lead.contact_name}{lead.contact_role ? ` · ${lead.contact_role}` : ""}
                        </div>
                      )}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3, flexShrink: 0 }}>
                      {lead.lead_type && (
                        <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 4, background: "var(--accent-light)", color: "var(--accent)", fontWeight: 600 }}>
                          {lead.lead_type.replace(/_/g, " ")}
                        </span>
                      )}
                      {lead.confidence > 0 && (
                        <span style={{ fontSize: 10, fontWeight: 700, color: lead.confidence >= 0.7 ? "var(--green)" : "var(--amber)" }}>
                          {Math.round(lead.confidence * 100)}%
                        </span>
                      )}
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        )}

        {campaign.status === "failed" && campaign.error && (
          <div style={{ marginTop: 8, padding: "14px 18px", background: "var(--red-light)", borderRadius: 10, border: "1px solid var(--red)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <XCircle size={16} style={{ color: "var(--red)" }} />
              <span style={{ fontSize: 14, fontWeight: 600, color: "var(--red)" }}>Campaign Failed</span>
            </div>
            <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{campaign.error}</div>
          </div>
        )}
      </div>
    </>
  );
}

/* ── Company accordion row ────────────────────────────── */

function CompanyAccordion({ company, campaignId, isActive, expanded, onToggle, sendState, onSendEmail }: {
  company: CampaignCompanyStatus;
  campaignId: string;
  isActive: boolean;
  expanded: boolean;
  onToggle: () => void;
  sendState: Record<string, { loading: boolean; result?: { success: boolean; recipient: string; test_mode: boolean; error: string }; error?: string }>;
  onSendEmail: (company: string, contact: CampaignContact, email: CampaignEmail) => void;
}) {
  const dotColor = STATUS_DOT[company.status] ?? "var(--text-muted)";
  const label = STATUS_LABEL[company.status] ?? company.status;
  const isProcessing = isActive && !["done", "failed", "pending"].includes(company.status);

  const hasContactData = (company.contacts?.length ?? 0) > 0;
  const hasEmailData = (company.emails?.length ?? 0) > 0;
  const hasCountsOnly = !hasContactData && !hasEmailData && (company.contacts_found > 0 || company.outreach_generated > 0);
  const canExpand = hasContactData || hasEmailData || hasCountsOnly || company.description || company.status === "done";
  const contactCount = hasContactData ? company.contacts!.length : company.contacts_found;
  const emailCount = hasEmailData ? company.emails!.length : company.outreach_generated;

  // Build email lookup by recipient name
  const emailMap = new Map<string, CampaignEmail>();
  if (hasEmailData) {
    for (const em of company.emails!) {
      emailMap.set(em.recipient_name.toLowerCase().trim(), em);
    }
  }

  return (
    <div className="card" style={{ overflow: "hidden" }}>
      {/* Header row */}
      <div onClick={canExpand ? onToggle : undefined}
        style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", cursor: canExpand ? "pointer" : "default", transition: "background 100ms ease" }}
        onMouseEnter={(e) => { if (canExpand) (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = ""; }}
      >
        <div style={{ flexShrink: 0, color: "var(--text-muted)", width: 14 }}>
          {canExpand ? (expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />) : null}
        </div>
        <div style={{ flexShrink: 0 }}>
          {isProcessing ? <Loader2 size={15} style={{ color: dotColor, animation: "spin 1s linear infinite" }} />
            : company.status === "done" ? <CheckCircle size={15} style={{ color: dotColor }} />
            : company.status === "failed" ? <XCircle size={15} style={{ color: dotColor }} />
            : <Clock size={15} style={{ color: dotColor }} />}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 1 }}>
            <Link href={`/companies/${companyHash(company.company_name)}`}
              onClick={e => e.stopPropagation()}
              style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", textDecoration: "none" }}
              onMouseEnter={e => (e.currentTarget.style.color = "var(--accent)")}
              onMouseLeave={e => (e.currentTarget.style.color = "var(--text)")}
            >
              {company.company_name}
            </Link>
            <span style={{ fontSize: 11, color: dotColor, fontWeight: 500 }}>{label}</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--text-muted)", flexWrap: "wrap" }}>
            {company.domain && <span style={{ display: "flex", alignItems: "center", gap: 3 }}><Globe size={9} /> {company.domain}</span>}
            {company.industry && <span>{company.industry}</span>}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          {contactCount > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 11, padding: "2px 7px", borderRadius: 5, background: "var(--amber-light)" }}>
              <Users size={11} style={{ color: "var(--amber)" }} />
              <span className="num" style={{ fontWeight: 600, color: "var(--amber)" }}>{contactCount}</span>
            </div>
          )}
          {emailCount > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 11, padding: "2px 7px", borderRadius: 5, background: "var(--accent-light)" }}>
              <Mail size={11} style={{ color: "var(--accent)" }} />
              <span className="num" style={{ fontWeight: 600, color: "var(--accent)" }}>{emailCount}</span>
            </div>
          )}
        </div>
      </div>

      {company.error && (
        <div style={{ margin: "0 16px 10px", fontSize: 11, color: "var(--red)", padding: "5px 10px", background: "var(--red-light)", borderRadius: 6 }}>{company.error}</div>
      )}

      {/* Expanded content */}
      {expanded && canExpand && (
        <div style={{ borderTop: "1px solid var(--border)", background: "var(--surface)" }}>

          {/* Company description */}
          {company.description && (
            <div style={{ padding: "12px 16px 0", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65 }}>
              {company.description.length > 280 ? company.description.slice(0, 280) + "…" : company.description}
              <Link href={`/companies/${companyHash(company.company_name)}`} onClick={e => e.stopPropagation()}
                style={{ marginLeft: 6, fontSize: 11, color: "var(--accent)", textDecoration: "none" }}>
                Full profile →
              </Link>
            </div>
          )}

          {/* Contacts with paired emails — person cards */}
          {hasContactData && (
            <div style={{ padding: "12px 16px" }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-xmuted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
                People & Outreach ({company.contacts!.length})
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 10 }}>
                {company.contacts!.map((contact, ci) => {
                  const matchedEmail = emailMap.get(contact.full_name.toLowerCase().trim());
                  const sendKey = `${company.company_name}-${contact.full_name}`;
                  const emailState = sendState[sendKey];
                  const seniorityInfo = contact.seniority ? SENIORITY_COLORS[contact.seniority] : null;

                  return (
                    <div key={ci} style={{ borderRadius: 10, border: "1px solid var(--border)", background: "var(--surface-raised)", overflow: "hidden" }}>
                      {/* Person header */}
                      <div style={{ padding: "12px 14px", display: "flex", alignItems: "flex-start", gap: 10 }}>
                        <div style={{ width: 36, height: 36, borderRadius: "50%", background: "var(--accent-light)", color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 800, flexShrink: 0 }}>
                          {(contact.full_name || "?")[0].toUpperCase()}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap", marginBottom: 2 }}>
                            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>{contact.full_name || "Unknown"}</span>
                            {seniorityInfo && (
                              <span style={{ fontSize: 9, color: seniorityInfo.color, display: "inline-flex", alignItems: "center", gap: 2, padding: "2px 6px", borderRadius: 4, background: "var(--surface)", border: `1px solid ${seniorityInfo.color}33`, fontWeight: 600 }}>
                                {seniorityInfo.icon} {contact.seniority!.replace(/_/g, " ")}
                              </span>
                            )}
                          </div>
                          {contact.role && <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{contact.role}</div>}
                          {contact.email && (
                            <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 5 }}>
                              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono, monospace)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {contact.email}
                              </span>
                              {contact.email_confidence > 0 && (
                                <span style={{ fontSize: 9, fontWeight: 700, color: contact.email_confidence >= 80 ? "var(--green)" : "var(--amber)", padding: "1px 4px", borderRadius: 3, background: contact.email_confidence >= 80 ? "var(--green-light)" : "var(--amber-light)", flexShrink: 0 }}>
                                  {Math.round(contact.email_confidence)}%
                                </span>
                              )}
                              <CopyButton text={contact.email} />
                            </div>
                          )}
                          {contact.linkedin_url && (
                            <a href={contact.linkedin_url} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()}
                              style={{ display: "inline-flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--blue)", textDecoration: "none", marginTop: 3 }}>
                              <ExternalLink size={9} /> LinkedIn
                            </a>
                          )}
                        </div>
                      </div>

                      {/* Paired email */}
                      {matchedEmail && (
                        <div style={{ borderTop: "1px solid var(--border)", padding: "10px 14px", background: "var(--bg)" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                            <Mail size={10} style={{ color: "var(--accent)", flexShrink: 0 }} />
                            <span style={{ fontSize: 11.5, fontWeight: 600, color: "var(--text)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              {matchedEmail.subject || "(no subject)"}
                            </span>
                            <CopyButton text={matchedEmail.subject} />
                          </div>
                          <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6, maxHeight: 80, overflow: "hidden", marginBottom: 8 }}>
                            {matchedEmail.body?.slice(0, 180)}{(matchedEmail.body?.length ?? 0) > 180 ? "…" : ""}
                          </div>
                          <div style={{ display: "flex", gap: 6 }}>
                            <CopyButton text={`Subject: ${matchedEmail.subject}\n\n${matchedEmail.body}`} label="Copy full email" size="md" />
                            {contact.email && (
                              <SendBtn state={emailState} onSend={() => onSendEmail(company.company_name, contact, matchedEmail)} />
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Standalone emails */}
          {hasEmailData && (() => {
            const pairedNames = new Set((company.contacts || []).map(c => c.full_name.toLowerCase().trim()));
            const unpairedEmails = company.emails!.filter(e => !pairedNames.has(e.recipient_name.toLowerCase().trim()));
            if (unpairedEmails.length === 0) return null;
            return (
              <div style={{ padding: "0 16px 12px" }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-xmuted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                  Additional Emails ({unpairedEmails.length})
                </div>
                {unpairedEmails.map((email, ei) => (
                  <EmailCard key={ei} email={email} companyName={company.company_name} campaignId={campaignId} sendState={sendState} onSend={(cn, em) => {
                    // Wrap as contact-level send (with empty contact so backend uses test recipient)
                    onSendEmail(cn, { full_name: em.recipient_name, role: em.recipient_role, email: "", linkedin_url: "", seniority: "", email_confidence: 0 }, em);
                  }} />
                ))}
              </div>
            );
          })()}

          {/* Legacy: counts only */}
          {hasCountsOnly && company.status === "done" && (
            <div style={{ padding: "12px 16px" }}>
              <div style={{ padding: "12px 14px", borderRadius: 8, background: "var(--surface-raised)", border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                  {company.contacts_found > 0 && <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 12, fontWeight: 600 }}><Users size={13} style={{ color: "var(--amber)" }} /> {company.contacts_found} contacts</span>}
                  {company.outreach_generated > 0 && <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 12, fontWeight: 600 }}><Mail size={13} style={{ color: "var(--accent)" }} /> {company.outreach_generated} emails</span>}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Re-run to see full contacts and emails.</div>
              </div>
            </div>
          )}

          {!hasContactData && !hasEmailData && !hasCountsOnly && company.status === "done" && (
            <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center", padding: "20px 0", display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <AlertCircle size={18} /> No contacts found for this company.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Send button (reusable) ──────────────────────────── */

function SendBtn({ state, onSend }: {
  state?: { loading: boolean; result?: { success: boolean; recipient: string; test_mode: boolean; error: string }; error?: string };
  onSend: () => void;
}) {
  if (state?.result) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "5px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: state.result.success ? "var(--green-light)" : "var(--red-light)", color: state.result.success ? "var(--green)" : "var(--red)" }}>
        {state.result.success ? <Check size={11} /> : <AlertTriangle size={11} />}
        {state.result.success ? "Sent" : (state.result.error || "Failed")}
        {state.result.test_mode && <span style={{ fontSize: 8, opacity: 0.8, marginLeft: 2 }}>TEST</span>}
      </div>
    );
  }
  if (state?.error) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span style={{ fontSize: 11, color: "var(--red)" }}><AlertTriangle size={11} /> {state.error.slice(0, 30)}</span>
        <button onClick={(e) => { e.stopPropagation(); onSend(); }} style={{
          fontSize: 10, padding: "3px 8px", borderRadius: 5,
          border: "1px solid var(--border)", background: "var(--surface)",
          color: "var(--text-secondary)", cursor: "pointer",
        }}>Retry</button>
      </div>
    );
  }
  return (
    <button onClick={(e) => { e.stopPropagation(); onSend(); }} disabled={state?.loading}
      style={{ display: "flex", alignItems: "center", gap: 4, padding: "5px 12px", fontSize: 11, fontWeight: 600, borderRadius: 6, cursor: state?.loading ? "not-allowed" : "pointer", color: "#fff", background: "var(--green)", border: "none", opacity: state?.loading ? 0.7 : 1 }}>
      {state?.loading ? <Loader2 size={11} style={{ animation: "spin 1s linear infinite" }} /> : <Send size={11} />}
      {state?.loading ? "Sending..." : "Send"}
    </button>
  );
}

/* ── Standalone email card ───────────────────────────── */

function EmailCard({ email, companyName, campaignId, sendState, onSend }: {
  email: CampaignEmail; companyName: string; campaignId: string;
  sendState: Record<string, { loading: boolean; result?: { success: boolean; recipient: string; test_mode: boolean; error: string }; error?: string }>;
  onSend: (companyName: string, email: CampaignEmail) => void;
}) {
  const key = `${companyName}-${email.recipient_name}`;
  const state = sendState[key];
  return (
    <div style={{ padding: "12px 14px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface-raised)", marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}>
        <Mail size={11} style={{ color: "var(--accent)" }} />
        <span style={{ fontSize: 12 }}>
          <span style={{ color: "var(--text-muted)" }}>To: </span>
          <strong>{email.recipient_name}</strong>
          {email.recipient_role && <span style={{ color: "var(--text-muted)" }}> · {email.recipient_role}</span>}
        </span>
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>{email.subject || "(no subject)"}</div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6, whiteSpace: "pre-wrap", maxHeight: 120, overflow: "hidden", padding: "8px 10px", background: "var(--bg)", borderRadius: 6, border: "1px solid var(--border)" }}>
        {email.body?.slice(0, 300)}{email.body?.length > 300 ? "..." : ""}
      </div>
      <div style={{ display: "flex", gap: 6, marginTop: 8, alignItems: "center" }}>
        <CopyButton text={`Subject: ${email.subject}\n\n${email.body}`} label="Copy Email" size="md" />
        {state?.result?.success ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--green)", padding: "4px 10px", background: "var(--green-light)", borderRadius: 6 }}>
            <CheckCircle size={12} /> Sent{state.result.test_mode ? " (test)" : ""}
          </span>
        ) : state?.error ? (
          <div style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: 11, color: "var(--red)" }}><AlertTriangle size={11} /> {state.error.slice(0, 40)}</span>
            <button onClick={() => onSend(companyName, email)} style={{
              fontSize: 11, padding: "4px 10px", borderRadius: 6,
              border: "1px solid var(--border)", background: "var(--surface)",
              color: "var(--text-secondary)", cursor: "pointer",
            }}>Retry</button>
          </div>
        ) : (
          <button onClick={() => onSend(companyName, email)} disabled={state?.loading} style={{
            display: "inline-flex", alignItems: "center", gap: 4,
            background: "var(--accent)", color: "white",
            border: "none", borderRadius: 6, padding: "5px 12px",
            fontSize: 11, cursor: state?.loading ? "wait" : "pointer",
            opacity: state?.loading ? 0.6 : 1,
          }}>
            {state?.loading ? <Loader2 size={11} style={{ animation: "spin 1s linear infinite" }} /> : <Send size={11} />}
            {state?.loading ? "Sending..." : "Send Test"}
          </button>
        )}
      </div>
    </div>
  );
}

/* ── Stat card ────────────────────────────────────────── */

function StatCard({ icon, value, label, color, bg }: { icon: React.ReactNode; value: number; label: string; color: string; bg: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", background: bg, borderRadius: 9, border: `1px solid ${color}22` }}>
      <div style={{ color, display: "flex" }}>{icon}</div>
      <div>
        <div className="num" style={{ fontSize: 18, fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>{label}</div>
      </div>
    </div>
  );
}
