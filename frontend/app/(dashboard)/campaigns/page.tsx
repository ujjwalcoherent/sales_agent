"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Plus, Crosshair, Building2, Globe, FileText, Users, Mail,
  Loader2, Trash2, Play, X,
  CheckCircle, XCircle, Clock,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import type { Campaign, CampaignType, CreateCampaignRequest } from "@/lib/types";

const TYPE_LABELS: Record<CampaignType, string> = {
  company_first: "Company-First",
  industry_first: "Industry",
  report_driven: "Report",
};

const TYPE_ICONS: Record<CampaignType, React.ReactNode> = {
  company_first: <Building2 size={13} />,
  industry_first: <Globe size={13} />,
  report_driven: <FileText size={13} />,
};

const TYPE_DESCRIPTIONS: Record<CampaignType, string> = {
  company_first: "Provide company names directly. Best when you know your targets.",
  industry_first: "Enter an industry and we'll discover companies for you.",
  report_driven: "Paste a report or article and we'll extract companies from it.",
};

const STATUS_BADGE: Record<string, string> = {
  draft: "badge-blue",
  running: "badge-amber",
  completed: "badge-green",
  failed: "badge-red",
};

const STATUS_ICON: Record<string, React.ReactNode> = {
  draft: <Clock size={12} />,
  running: <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} />,
  completed: <CheckCircle size={12} />,
  failed: <XCircle size={12} />,
};

export default function CampaignsPage() {
  const router = useRouter();
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Create form state
  const [campaignType, setCampaignType] = useState<CampaignType>("company_first");
  const [campaignName, setCampaignName] = useState("");
  const [companiesText, setCompaniesText] = useState("");
  const [industryText, setIndustryText] = useState("");
  const [reportText, setReportText] = useState("");
  const [showConfig, setShowConfig] = useState(false);
  const [maxCompanies, setMaxCompanies] = useState(10);
  const [maxContacts, setMaxContacts] = useState(5);
  const [generateOutreach, setGenerateOutreach] = useState(true);

  useEffect(() => {
    loadCampaigns();
  }, []);

  async function loadCampaigns() {
    try {
      const res = await api.listCampaigns(50);
      setCampaigns(res.campaigns);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load campaigns");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate() {
    setCreating(true);
    setError(null);
    try {
      const req: CreateCampaignRequest = {
        campaign_type: campaignType,
        ...(campaignName.trim() ? { name: campaignName.trim() } : {}),
        config: {
          max_companies: maxCompanies,
          max_contacts_per_company: maxContacts,
          generate_outreach: generateOutreach,
        },
      };

      if (campaignType === "company_first") {
        const lines = companiesText.split("\n").map(l => l.trim()).filter(Boolean);
        if (lines.length === 0) { setError("Enter at least one company name"); setCreating(false); return; }
        req.companies = lines.map(name => ({ company_name: name }));
      } else if (campaignType === "industry_first") {
        if (!industryText.trim()) { setError("Enter an industry name"); setCreating(false); return; }
        req.industry = industryText.trim();
      } else {
        if (!reportText.trim()) { setError("Paste report text"); setCreating(false); return; }
        req.report_text = reportText.trim();
      }

      const campaign = await api.createCampaign(req);
      // Navigate to campaign detail page immediately
      router.push(`/campaigns/${campaign.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create campaign");
      setCreating(false);
    }
  }

  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);

  function requestDelete(e: React.MouseEvent, campaign: Campaign) {
    e.stopPropagation();
    setDeleteTarget({ id: campaign.id, name: campaign.name || `Campaign ${campaign.id.slice(0, 8)}` });
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(deleteTarget.id);
    setDeleteTarget(null);
    try {
      await api.deleteCampaign(deleteTarget.id);
      setCampaigns(prev => prev.filter(c => c.id !== deleteTarget.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete");
    } finally {
      setDeleting(null);
    }
  }

  async function handleQuickRun(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    try {
      await api.runCampaign(id);
      router.push(`/campaigns/${id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start");
    }
  }

  function resetForm() {
    setCampaignName("");
    setCompaniesText("");
    setIndustryText("");
    setReportText("");
    setCampaignType("company_first");
    setShowConfig(false);
    setMaxCompanies(10);
    setMaxContacts(5);
    setGenerateOutreach(true);
    setError(null);
  }

  return (
    <>
      {/* Delete confirmation modal */}
      {deleteTarget && (
        <div style={{ position: "fixed", inset: 0, zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)" }}
          onClick={() => setDeleteTarget(null)}>
          <div onClick={e => e.stopPropagation()} style={{ background: "var(--surface)", borderRadius: 14, padding: "24px 28px", width: "min(400px, 90vw)", border: "1px solid var(--border)", boxShadow: "0 20px 60px rgba(0,0,0,0.3)" }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>Delete Campaign</div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 18 }}>
              Delete <strong>{deleteTarget.name}</strong>? This cannot be undone.
            </div>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button onClick={() => setDeleteTarget(null)} style={{ padding: "8px 16px", fontSize: 13, borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text-secondary)", cursor: "pointer" }}>Cancel</button>
              <button onClick={confirmDelete} style={{ padding: "8px 16px", fontSize: 13, fontWeight: 600, borderRadius: 8, border: "none", background: "var(--red)", color: "#fff", cursor: "pointer" }}>Delete</button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{ padding: "16px 24px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
              Campaigns
            </h1>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
              {loading ? "" : `${campaigns.length} campaign${campaigns.length !== 1 ? "s" : ""}`}
            </span>
          </div>
          <button
            onClick={() => setShowCreate(true)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "var(--text)", color: "var(--bg)",
              border: "none",
              borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, cursor: "pointer",
            }}
          >
            <Plus size={14} />
            New Campaign
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        {/* Error banner */}
        {error && (
          <div style={{ padding: "10px 14px", background: "var(--red-light)", color: "var(--red)", borderRadius: 8, fontSize: 12, marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            {error}
            <button onClick={() => setError(null)} style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 14, fontWeight: 600 }}>&times;</button>
          </div>
        )}

        {/* Create campaign modal */}
        {showCreate && (
          <div style={{ position: "fixed", inset: 0, zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.45)", backdropFilter: "blur(6px)" }}
            onClick={() => { setShowCreate(false); resetForm(); }}>
            <div onClick={e => e.stopPropagation()} style={{
              background: "var(--surface)", borderRadius: 16, width: "min(580px, 96vw)", maxHeight: "90vh", overflow: "auto",
              border: "1px solid var(--border)", boxShadow: "0 24px 80px rgba(0,0,0,0.35)",
            }}>
              {/* Modal header */}
              <div style={{ padding: "20px 24px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.01em" }}>New Campaign</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>Automate outreach across multiple companies</div>
                </div>
                <button onClick={() => { setShowCreate(false); resetForm(); }} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 4, borderRadius: 6, display: "flex" }}>
                  <X size={16} />
                </button>
              </div>

              <div style={{ padding: "20px 24px" }}>
                {/* Campaign name */}
                <div style={{ marginBottom: 20 }}>
                  <label style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 6 }}>
                    Campaign Name
                  </label>
                  <input
                    type="text"
                    value={campaignName}
                    onChange={e => setCampaignName(e.target.value)}
                    placeholder="e.g. Q2 Fintech Outreach"
                    autoFocus
                    style={{ width: "100%", padding: "9px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface-raised)", fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box" }}
                  />
                </div>

                {/* Type selection — card style */}
                <div style={{ marginBottom: 20 }}>
                  <label style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 8 }}>
                    Campaign Type
                  </label>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {(["company_first", "industry_first", "report_driven"] as CampaignType[]).map((t) => (
                      <div
                        key={t}
                        onClick={() => setCampaignType(t)}
                        style={{
                          display: "flex", alignItems: "center", gap: 12, padding: "12px 14px",
                          borderRadius: 10, border: `2px solid ${campaignType === t ? "var(--accent)" : "var(--border)"}`,
                          background: campaignType === t ? "var(--accent-light)" : "var(--surface-raised)",
                          cursor: "pointer", transition: "all 120ms",
                        }}
                      >
                        <div style={{ width: 36, height: 36, borderRadius: 9, background: campaignType === t ? "var(--accent)" : "var(--surface)", border: `1px solid ${campaignType === t ? "transparent" : "var(--border)"}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: campaignType === t ? "#fff" : "var(--text-muted)" }}>
                          {TYPE_ICONS[t]}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 13, fontWeight: 600, color: campaignType === t ? "var(--accent)" : "var(--text)", marginBottom: 2 }}>{TYPE_LABELS[t]}</div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.4 }}>{TYPE_DESCRIPTIONS[t]}</div>
                        </div>
                        <div style={{ width: 16, height: 16, borderRadius: "50%", border: `2px solid ${campaignType === t ? "var(--accent)" : "var(--border)"}`, background: campaignType === t ? "var(--accent)" : "transparent", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                          {campaignType === t && <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#fff" }} />}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Type-specific input */}
                <div style={{ marginBottom: 20 }}>
                  <label style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 6 }}>
                    {campaignType === "company_first" ? "Target Companies" : campaignType === "industry_first" ? "Industry / Sector" : "Report Text"}
                  </label>
                  {campaignType === "company_first" && (
                    <>
                      <textarea
                        value={companiesText}
                        onChange={e => setCompaniesText(e.target.value)}
                        placeholder={"NVIDIA\nInfosys\nTesla\nFreshworks"}
                        rows={5}
                        style={{ width: "100%", padding: "9px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface-raised)", fontSize: 13, color: "var(--text)", outline: "none", resize: "vertical", fontFamily: "inherit", lineHeight: 1.7, boxSizing: "border-box" }}
                      />
                      <span style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4, display: "block" }}>
                        {companiesText.split("\n").filter(l => l.trim()).length} companies · one per line
                      </span>
                    </>
                  )}
                  {campaignType === "industry_first" && (
                    <input
                      type="text"
                      value={industryText}
                      onChange={e => setIndustryText(e.target.value)}
                      placeholder="e.g. fintech India, cybersecurity, edtech"
                      style={{ width: "100%", padding: "9px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface-raised)", fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box" }}
                    />
                  )}
                  {campaignType === "report_driven" && (
                    <textarea
                      value={reportText}
                      onChange={e => setReportText(e.target.value)}
                      placeholder="Paste an industry report, news article, or trend analysis..."
                      rows={5}
                      style={{ width: "100%", padding: "9px 12px", borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface-raised)", fontSize: 13, color: "var(--text)", outline: "none", resize: "vertical", fontFamily: "inherit", boxSizing: "border-box" }}
                    />
                  )}
                </div>

                {/* Config */}
                <div style={{ marginBottom: 24, padding: "12px 14px", background: "var(--surface-raised)", borderRadius: 9, border: "1px solid var(--border)" }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>Settings</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    <div>
                      <label style={{ fontSize: 11, color: "var(--text-muted)", display: "block", marginBottom: 4 }}>Max Companies</label>
                      <input
                        type="number" min={1} max={50} value={maxCompanies}
                        onChange={e => setMaxCompanies(Number(e.target.value))}
                        style={{ width: "100%", padding: "6px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box" }}
                      />
                    </div>
                    <div>
                      <label style={{ fontSize: 11, color: "var(--text-muted)", display: "block", marginBottom: 4 }}>Max Contacts / Company</label>
                      <input
                        type="number" min={1} max={20} value={maxContacts}
                        onChange={e => setMaxContacts(Number(e.target.value))}
                        style={{ width: "100%", padding: "6px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box" }}
                      />
                    </div>
                  </div>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 12, color: "var(--text-secondary)", marginTop: 10 }}>
                    <input
                      type="checkbox" checked={generateOutreach}
                      onChange={e => setGenerateOutreach(e.target.checked)}
                      style={{ accentColor: "var(--accent)", width: 14, height: 14 }}
                    />
                    Generate personalized outreach emails for each contact
                  </label>
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: 10 }}>
                  <button
                    onClick={() => { setShowCreate(false); resetForm(); }}
                    style={{ flex: 1, padding: "10px", fontSize: 13, fontWeight: 500, borderRadius: 9, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text-secondary)", cursor: "pointer" }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreate}
                    disabled={creating}
                    style={{
                      flex: 2, display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                      background: creating ? "var(--surface-raised)" : "var(--accent)", color: creating ? "var(--text-muted)" : "#fff",
                      borderRadius: 9, padding: "10px 20px", fontSize: 13, fontWeight: 600,
                      border: "none", cursor: creating ? "not-allowed" : "pointer",
                    }}
                  >
                    {creating ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Crosshair size={14} />}
                    {creating ? "Creating campaign..." : "Create Campaign"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Campaign list */}
        {loading ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 14 }}>
            {[0, 1, 2].map(i => (
              <div key={i} className="card" style={{ padding: "16px 18px" }}>
                <div className="skeleton" style={{ height: 15, width: "60%", marginBottom: 10 }} />
                <div className="skeleton" style={{ height: 12, width: "40%", marginBottom: 14 }} />
                <div className="skeleton" style={{ height: 12, width: "80%" }} />
              </div>
            ))}
          </div>
        ) : campaigns.length === 0 ? (
          <div style={{ padding: "60px 24px", textAlign: "center", color: "var(--text-muted)" }}>
            <Crosshair size={32} style={{ margin: "0 auto 12px", opacity: 0.3 }} />
            <div style={{ fontSize: 14, fontWeight: 500, color: "var(--text-secondary)", marginBottom: 6 }}>
              No campaigns yet
            </div>
            <div style={{ fontSize: 12, lineHeight: 1.6, maxWidth: 400, margin: "0 auto" }}>
              Campaigns automate lead generation. Enter company names, an industry, or paste a report
              {" "}and we'll find contacts and generate personalized outreach emails.
            </div>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 14 }}>
            {campaigns.map(c => {
              const previewContacts = c.companies
                ?.flatMap(co => co.contacts || [])
                .slice(0, 3)
                .map(ct => ct.full_name)
                .filter(Boolean);

              return (
                <div
                  key={c.id}
                  className="card card-hover"
                  onClick={() => router.push(`/campaigns/${c.id}`)}
                  style={{ padding: "16px 18px", cursor: "pointer" }}
                >
                  {/* Name + type + status */}
                  <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8, marginBottom: 10 }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 4 }}>
                        {c.name || `Campaign ${c.id.slice(0, 8)}`}
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span className={`badge ${STATUS_BADGE[c.status] ?? "badge-muted"}`} style={{ fontSize: 10, display: "flex", alignItems: "center", gap: 3 }}>
                          {STATUS_ICON[c.status]} {c.status}
                        </span>
                        <span className="badge badge-muted" style={{ fontSize: 10, display: "flex", alignItems: "center", gap: 3 }}>
                          {TYPE_ICONS[c.campaign_type]} {TYPE_LABELS[c.campaign_type]}
                        </span>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
                      {/* Quick run button for draft campaigns */}
                      {(c.status === "draft" || c.status === "failed") && (
                        <button
                          onClick={(e) => handleQuickRun(e, c.id)}
                          style={{
                            background: "var(--green)", color: "#fff", border: "none",
                            borderRadius: 6, padding: "4px 8px", cursor: "pointer",
                            display: "flex", alignItems: "center", gap: 3, fontSize: 11, fontWeight: 500,
                          }}
                          title="Run campaign"
                        >
                          <Play size={11} /> Run
                        </button>
                      )}
                      <button
                        onClick={(e) => requestDelete(e, c)}
                        disabled={deleting === c.id}
                        style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 4, borderRadius: 4, flexShrink: 0 }}
                        title="Delete campaign"
                      >
                        {deleting === c.id ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} /> : <Trash2 size={13} />}
                      </button>
                    </div>
                  </div>

                  {/* Stats row */}
                  <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 10 }}>
                    <StatPill icon={<Building2 size={11} />} value={c.total_companies} label="companies" />
                    <StatPill icon={<Users size={11} />} value={c.total_contacts} label="contacts" />
                    <StatPill icon={<Mail size={11} />} value={c.total_outreach} label="emails" />
                  </div>

                  {/* Progress bar */}
                  {c.total_companies > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ height: 4, background: "var(--surface-raised)", borderRadius: 2, overflow: "hidden" }}>
                        <div style={{
                          height: "100%",
                          width: `${Math.round((c.completed_companies / c.total_companies) * 100)}%`,
                          background: c.status === "failed" ? "var(--red)" : c.status === "completed" ? "var(--green)" : "var(--accent)",
                          borderRadius: 2,
                          transition: "width 300ms ease",
                        }} />
                      </div>
                      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3 }}>
                        {c.completed_companies}/{c.total_companies} companies processed
                      </div>
                    </div>
                  )}

                  {/* Contact preview for completed campaigns */}
                  {c.status === "completed" && previewContacts && previewContacts.length > 0 && (
                    <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 6 }}>
                      {previewContacts.join(", ")}
                      {c.total_contacts > 3 && ` +${c.total_contacts - 3} more`}
                    </div>
                  )}

                  {/* Date */}
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                    Created {formatDate(c.created_at, "short")}
                    {c.completed_at && ` \u00b7 Done ${formatDate(c.completed_at, "short")}`}
                  </div>

                  {/* Error */}
                  {c.error && (
                    <div style={{ marginTop: 6, fontSize: 11, color: "var(--red)", padding: "4px 8px", background: "var(--red-light)", borderRadius: 5 }}>
                      {c.error}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}

function StatPill({ icon, value, label }: { icon: React.ReactNode; value: number; label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, color: "var(--text-secondary)" }}>
      <span style={{ color: "var(--text-muted)", display: "flex" }}>{icon}</span>
      <span className="num" style={{ fontWeight: 600 }}>{value}</span>
      <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{label}</span>
    </div>
  );
}
