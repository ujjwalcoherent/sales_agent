"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Plus, Building2, Globe, FileText, Users, Mail,
  Loader2, Trash2, Play, X, Crosshair,
  CheckCircle, XCircle, Clock, ChevronRight,
} from "lucide-react";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import type { Campaign, CampaignType, CreateCampaignRequest } from "@/lib/types";

/* ── Constants ────────────────────────────────────────── */

const TYPE_META: Record<CampaignType, {
  label: string;
  subtitle: string;
  description: string;
  icon: React.ReactNode;
  hoverBorder: string;
  hoverBg: string;
  iconBg: string;
  iconColor: string;
}> = {
  company_first: {
    label: "Company-First",
    subtitle: "I know who to target",
    description: "Provide company names directly and we'll enrich each one, find decision-makers, and generate personalized outreach.",
    icon: <Building2 size={20} />,
    hoverBorder: "var(--blue)",
    hoverBg: "var(--blue-light)",
    iconBg: "#2A5A8A",
    iconColor: "#fff",
  },
  industry_first: {
    label: "Industry-First",
    subtitle: "Discover by sector",
    description: "Enter an industry or sector and Harbinger will discover the most relevant companies for you automatically.",
    icon: <Globe size={20} />,
    hoverBorder: "var(--green)",
    hoverBg: "var(--green-light)",
    iconBg: "var(--green)",
    iconColor: "#fff",
  },
  report_driven: {
    label: "Report-Driven",
    subtitle: "Extract from content",
    description: "Paste a research report, news article, or analyst note and we'll extract target companies from it.",
    icon: <FileText size={20} />,
    hoverBorder: "#7C3AED",
    hoverBg: "#F3F0FF",
    iconBg: "#7C3AED",
    iconColor: "#fff",
  },
};

const STATUS_BADGE: Record<string, string> = {
  draft: "badge-blue",
  running: "badge-amber",
  completed: "badge-green",
  failed: "badge-red",
};

const STATUS_ICON: Record<string, React.ReactNode> = {
  draft: <Clock size={11} />,
  running: <Loader2 size={11} style={{ animation: "spin 1s linear infinite" }} />,
  completed: <CheckCircle size={11} />,
  failed: <XCircle size={11} />,
};

/* ── Sub-components ───────────────────────────────────── */

function UseCaseCard({
  type, selected, onClick,
}: {
  type: CampaignType;
  selected?: boolean;
  onClick: () => void;
}) {
  const m = TYPE_META[type];
  const [hovered, setHovered] = useState(false);
  const active = selected || hovered;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        flex: 1,
        minWidth: 0,
        padding: "18px 20px",
        borderRadius: 12,
        border: `1.5px solid ${active ? m.hoverBorder : "var(--border)"}`,
        background: active ? m.hoverBg : "var(--surface)",
        cursor: "pointer",
        transition: "all 180ms ease",
        boxShadow: active ? "0 4px 16px rgba(0,0,0,0.07)" : "var(--shadow-xs)",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: active ? m.iconBg : "var(--surface-raised)",
          color: active ? m.iconColor : "var(--text-muted)",
          display: "flex", alignItems: "center", justifyContent: "center",
          transition: "all 180ms ease", flexShrink: 0,
        }}>
          {m.icon}
        </div>
        <div>
          <div style={{
            fontSize: 14, fontWeight: 700,
            color: active ? "var(--text)" : "var(--text)",
            letterSpacing: "-0.01em",
            marginBottom: 2,
          }}>
            {m.label}
          </div>
          <div style={{ fontSize: 11, color: active ? "var(--text-secondary)" : "var(--text-muted)" }}>
            {m.subtitle}
          </div>
        </div>
        <ChevronRight size={14} style={{ marginLeft: "auto", color: "var(--text-muted)", flexShrink: 0 }} />
      </div>
      <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.55 }}>
        {m.description}
      </div>
    </div>
  );
}

function StatPill({ icon, value, label }: { icon: React.ReactNode; value: number; label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, color: "var(--text-secondary)" }}>
      <span style={{ color: "var(--text-muted)", display: "flex" }}>{icon}</span>
      <span className="num" style={{ fontWeight: 600 }}>{value ?? 0}</span>
      <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{label}</span>
    </div>
  );
}

function CampaignCard({
  campaign,
  onRun,
  onDelete,
  deleting,
  onClick,
}: {
  campaign: Campaign;
  onRun: (e: React.MouseEvent) => void;
  onDelete: (e: React.MouseEvent) => void;
  deleting: boolean;
  onClick: () => void;
}) {
  const m = TYPE_META[campaign.campaign_type];
  const pct = campaign.total_companies > 0
    ? Math.round((campaign.completed_companies / campaign.total_companies) * 100)
    : 0;
  const progressColor =
    campaign.status === "failed" ? "var(--red)" :
    campaign.status === "completed" ? "var(--green)" :
    "var(--accent)";

  return (
    <div
      className="card card-hover"
      onClick={onClick}
      style={{ padding: "16px 18px", cursor: "pointer" }}
    >
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 12 }}>
        <div style={{
          width: 34, height: 34, borderRadius: 8, flexShrink: 0,
          background: "var(--surface-raised)", color: "var(--text-muted)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          {m.icon}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 14, fontWeight: 600, color: "var(--text)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 4,
          }}>
            {campaign.name || `Campaign ${campaign.id.slice(0, 8)}`}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
            <span
              className={`badge ${STATUS_BADGE[campaign.status] ?? "badge-muted"}`}
              style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 3 }}
            >
              {STATUS_ICON[campaign.status]} {campaign.status}
            </span>
            <span className="badge badge-muted" style={{ fontSize: 10 }}>
              {m.label}
            </span>
          </div>
        </div>
        {/* Actions */}
        <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
          {(campaign.status === "draft" || campaign.status === "failed") && (
            <button
              onClick={onRun}
              title="Run campaign"
              style={{
                background: "var(--green)", color: "#fff", border: "none",
                borderRadius: 6, padding: "4px 9px", cursor: "pointer",
                display: "flex", alignItems: "center", gap: 3, fontSize: 11, fontWeight: 500,
              }}
            >
              <Play size={10} /> Run
            </button>
          )}
          <button
            onClick={onDelete}
            disabled={deleting}
            title="Delete"
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: "var(--text-muted)", padding: "4px 6px", borderRadius: 4,
            }}
          >
            {deleting
              ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} />
              : <Trash2 size={13} />
            }
          </button>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 10 }}>
        <StatPill icon={<Building2 size={11} />} value={campaign.total_companies} label="companies" />
        <StatPill icon={<Users size={11} />} value={campaign.total_contacts} label="contacts" />
        <StatPill icon={<Mail size={11} />} value={campaign.total_outreach} label="emails" />
      </div>

      {/* Progress bar */}
      {campaign.total_companies > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ height: 4, background: "var(--surface-raised)", borderRadius: 2, overflow: "hidden" }}>
            <div style={{
              height: "100%", width: `${pct}%`,
              background: progressColor, borderRadius: 2,
              transition: "width 400ms ease",
            }} />
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3 }}>
            {campaign.completed_companies}/{campaign.total_companies} processed
          </div>
        </div>
      )}

      {/* Error */}
      {campaign.error && (
        <div style={{
          marginBottom: 6, fontSize: 11, color: "var(--red)",
          padding: "4px 8px", background: "var(--red-light)", borderRadius: 5,
        }}>
          {campaign.error}
        </div>
      )}

      {/* Date */}
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        Created {formatDate(campaign.created_at, "short")}
        {campaign.completed_at && ` · Done ${formatDate(campaign.completed_at, "short")}`}
      </div>
    </div>
  );
}

/* ── Create modal ──────────────────────────────────────── */

function CreateModal({
  initialType,
  onClose,
  onCreate,
}: {
  initialType: CampaignType;
  onClose: () => void;
  onCreate: (campaign: Campaign) => void;
}) {
  const router = useRouter();
  const [campaignType, setCampaignType] = useState<CampaignType>(initialType);
  const [campaignName, setCampaignName] = useState("");
  const [companiesText, setCompaniesText] = useState("");
  const [industryText, setIndustryText] = useState("");
  const [reportText, setReportText] = useState("");
  const [maxCompanies, setMaxCompanies] = useState(10);
  const [maxContacts, setMaxContacts] = useState(5);
  const [generateOutreach, setGenerateOutreach] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const companyCount = companiesText.split("\n").filter(l => l.trim()).length;

  async function handleCreate() {
    setError(null);
    setCreating(true);
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
        if (!reportText.trim()) { setError("Paste report content"); setCreating(false); return; }
        req.report_text = reportText.trim();
      }

      const campaign = await api.createCampaign(req);
      router.push(`/campaigns/${campaign.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create campaign");
      setCreating(false);
    }
  }

  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "rgba(0,0,0,0.45)", backdropFilter: "blur(6px)",
      }}
      onClick={onClose}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: "var(--surface)", borderRadius: 16,
          width: "min(580px, 96vw)", maxHeight: "92vh", overflow: "auto",
          border: "1px solid var(--border)", boxShadow: "0 24px 80px rgba(0,0,0,0.3)",
        }}
      >
        {/* Modal header */}
        <div style={{
          padding: "20px 24px 16px",
          borderBottom: "1px solid var(--border)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.01em" }}>
              New Campaign
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
              Automate outreach across multiple companies
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: "var(--text-muted)", padding: 6, borderRadius: 6, display: "flex",
            }}
          >
            <X size={15} />
          </button>
        </div>

        <div style={{ padding: "20px 24px 24px" }}>

          {/* Error */}
          {error && (
            <div style={{
              padding: "9px 12px", background: "var(--red-light)", color: "var(--red)",
              borderRadius: 8, fontSize: 12, marginBottom: 16,
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              {error}
              <button onClick={() => setError(null)} style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 14 }}>×</button>
            </div>
          )}

          {/* Campaign name */}
          <div style={{ marginBottom: 20 }}>
            <label style={{
              fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
              textTransform: "uppercase", letterSpacing: "0.06em",
              display: "block", marginBottom: 6,
            }}>
              Campaign Name
            </label>
            <input
              type="text"
              value={campaignName}
              onChange={e => setCampaignName(e.target.value)}
              placeholder="e.g. Q2 Fintech Outreach"
              autoFocus
              style={{
                width: "100%", padding: "9px 12px", borderRadius: 8,
                border: "1px solid var(--border)", background: "var(--surface-raised)",
                fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box",
              }}
            />
          </div>

          {/* Type selector */}
          <div style={{ marginBottom: 20 }}>
            <label style={{
              fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
              textTransform: "uppercase", letterSpacing: "0.06em",
              display: "block", marginBottom: 8,
            }}>
              Campaign Type
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
              {(["company_first", "industry_first", "report_driven"] as CampaignType[]).map(t => {
                const m = TYPE_META[t];
                const active = campaignType === t;
                return (
                  <div
                    key={t}
                    onClick={() => setCampaignType(t)}
                    style={{
                      display: "flex", alignItems: "center", gap: 12,
                      padding: "12px 14px", borderRadius: 10,
                      border: `2px solid ${active ? m.hoverBorder : "var(--border)"}`,
                      background: active ? m.hoverBg : "var(--surface-raised)",
                      cursor: "pointer", transition: "all 130ms",
                    }}
                  >
                    <div style={{
                      width: 34, height: 34, borderRadius: 8, flexShrink: 0,
                      background: active ? m.iconBg : "var(--surface)",
                      color: active ? m.iconColor : "var(--text-muted)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      transition: "all 130ms",
                    }}>
                      {m.icon}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{
                        fontSize: 13, fontWeight: 600, marginBottom: 1,
                        color: active ? "var(--text)" : "var(--text)",
                      }}>
                        {m.label}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        {m.subtitle}
                      </div>
                    </div>
                    {/* Radio dot */}
                    <div style={{
                      width: 16, height: 16, borderRadius: "50%", flexShrink: 0,
                      border: `2px solid ${active ? m.hoverBorder : "var(--border)"}`,
                      background: active ? m.hoverBorder : "transparent",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                      {active && <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#fff" }} />}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Type-specific input */}
          <div style={{ marginBottom: 20 }}>
            <label style={{
              fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
              textTransform: "uppercase", letterSpacing: "0.06em",
              display: "block", marginBottom: 6,
            }}>
              {campaignType === "company_first" ? "Target Companies" :
               campaignType === "industry_first" ? "Industry / Sector" :
               "Report Content"}
            </label>

            {campaignType === "company_first" && (
              <>
                <textarea
                  value={companiesText}
                  onChange={e => setCompaniesText(e.target.value)}
                  placeholder={"NVIDIA\nInfosys\nTesla\nFreshworks"}
                  rows={5}
                  style={{
                    width: "100%", padding: "9px 12px", borderRadius: 8,
                    border: "1px solid var(--border)", background: "var(--surface-raised)",
                    fontSize: 13, color: "var(--text)", outline: "none",
                    resize: "vertical", fontFamily: "inherit", lineHeight: 1.7,
                    boxSizing: "border-box",
                  }}
                />
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                  {companyCount > 0 ? `${companyCount} compan${companyCount === 1 ? "y" : "ies"} · ` : ""}one per line
                </div>
              </>
            )}

            {campaignType === "industry_first" && (
              <input
                type="text"
                value={industryText}
                onChange={e => setIndustryText(e.target.value)}
                placeholder="e.g. fintech India, cybersecurity, edtech"
                style={{
                  width: "100%", padding: "9px 12px", borderRadius: 8,
                  border: "1px solid var(--border)", background: "var(--surface-raised)",
                  fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box",
                }}
              />
            )}

            {campaignType === "report_driven" && (
              <>
                <textarea
                  value={reportText}
                  onChange={e => setReportText(e.target.value)}
                  placeholder="Paste an industry report, news article, or trend analysis here — Harbinger will extract target companies automatically."
                  rows={6}
                  style={{
                    width: "100%", padding: "9px 12px", borderRadius: 8,
                    border: "1px solid var(--border)", background: "var(--surface-raised)",
                    fontSize: 13, color: "var(--text)", outline: "none",
                    resize: "vertical", fontFamily: "inherit", lineHeight: 1.6,
                    boxSizing: "border-box",
                  }}
                />
                {reportText.trim() && (
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                    {reportText.trim().split(/\s+/).length} words · {Math.ceil(reportText.trim().split(/\s+/).length / 200)} min read
                  </div>
                )}
              </>
            )}
          </div>

          {/* Settings */}
          <div style={{
            marginBottom: 24, padding: "14px 16px",
            background: "var(--surface-raised)", borderRadius: 10,
            border: "1px solid var(--border)",
          }}>
            <div style={{
              fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
              textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12,
            }}>
              Settings
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 10 }}>
              <div>
                <label style={{ fontSize: 11, color: "var(--text-secondary)", display: "block", marginBottom: 5 }}>
                  Max Companies
                </label>
                <input
                  type="number" min={1} max={50} value={maxCompanies}
                  onChange={e => setMaxCompanies(Number(e.target.value))}
                  style={{
                    width: "100%", padding: "6px 10px", borderRadius: 6,
                    border: "1px solid var(--border)", background: "var(--surface)",
                    fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, color: "var(--text-secondary)", display: "block", marginBottom: 5 }}>
                  Contacts per Company
                </label>
                <input
                  type="number" min={1} max={20} value={maxContacts}
                  onChange={e => setMaxContacts(Number(e.target.value))}
                  style={{
                    width: "100%", padding: "6px 10px", borderRadius: 6,
                    border: "1px solid var(--border)", background: "var(--surface)",
                    fontSize: 13, color: "var(--text)", outline: "none", boxSizing: "border-box",
                  }}
                />
              </div>
            </div>
            <label style={{
              display: "flex", alignItems: "center", gap: 8,
              cursor: "pointer", fontSize: 12, color: "var(--text-secondary)",
            }}>
              <input
                type="checkbox"
                checked={generateOutreach}
                onChange={e => setGenerateOutreach(e.target.checked)}
                style={{ accentColor: "var(--accent)", width: 14, height: 14 }}
              />
              Generate personalized outreach emails for each contact
            </label>
          </div>

          {/* Actions */}
          <div style={{ display: "flex", gap: 10 }}>
            <button
              onClick={onClose}
              style={{
                flex: 1, padding: "10px", fontSize: 13, fontWeight: 500,
                borderRadius: 9, border: "1px solid var(--border)",
                background: "var(--surface)", color: "var(--text-secondary)", cursor: "pointer",
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleCreate}
              disabled={creating}
              style={{
                flex: 2, display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
                background: creating ? "var(--surface-raised)" : "var(--accent)",
                color: creating ? "var(--text-muted)" : "#fff",
                borderRadius: 9, padding: "10px 20px", fontSize: 13, fontWeight: 600,
                border: "none", cursor: creating ? "not-allowed" : "pointer",
                transition: "all 150ms",
              }}
            >
              {creating
                ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
                : <Crosshair size={14} />
              }
              {creating ? "Creating campaign..." : "Create Campaign"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Delete confirmation ────────────────────────────────── */

function DeleteModal({ name, onConfirm, onCancel }: {
  name: string; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 1001,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)",
      }}
      onClick={onCancel}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: "var(--surface)", borderRadius: 14, padding: "24px 28px",
          width: "min(400px, 90vw)", border: "1px solid var(--border)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
        }}
      >
        <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>
          Delete Campaign
        </div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 20 }}>
          Delete <strong>{name}</strong>? This cannot be undone.
        </div>
        <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
          <button
            onClick={onCancel}
            style={{
              padding: "8px 16px", fontSize: 13, borderRadius: 8,
              border: "1px solid var(--border)", background: "var(--surface)",
              color: "var(--text-secondary)", cursor: "pointer",
            }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            style={{
              padding: "8px 16px", fontSize: 13, fontWeight: 600, borderRadius: 8,
              border: "none", background: "var(--red)", color: "#fff", cursor: "pointer",
            }}
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Page ──────────────────────────────────────────────── */

export default function CampaignsPage() {
  const router = useRouter();
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal state
  const [showCreate, setShowCreate] = useState(false);
  const [createType, setCreateType] = useState<CampaignType>("company_first");

  // Delete state
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => { loadCampaigns(); }, []);

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

  function openCreate(type: CampaignType) {
    setCreateType(type);
    setShowCreate(true);
  }

  function requestDelete(e: React.MouseEvent, campaign: Campaign) {
    e.stopPropagation();
    setDeleteTarget({ id: campaign.id, name: campaign.name || `Campaign ${campaign.id.slice(0, 8)}` });
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    const id = deleteTarget.id;
    setDeleting(id);
    setDeleteTarget(null);
    try {
      await api.deleteCampaign(id);
      setCampaigns(prev => prev.filter(c => c.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete campaign");
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
      setError(err instanceof Error ? err.message : "Failed to start campaign");
    }
  }

  return (
    <>
      {/* Modals */}
      {showCreate && (
        <CreateModal
          initialType={createType}
          onClose={() => setShowCreate(false)}
          onCreate={() => setShowCreate(false)}
        />
      )}
      {deleteTarget && (
        <DeleteModal
          name={deleteTarget.name}
          onConfirm={confirmDelete}
          onCancel={() => setDeleteTarget(null)}
        />
      )}

      {/* Header */}
      <div style={{
        padding: "15px 24px 13px",
        borderBottom: "1px solid var(--border)",
        background: "var(--surface)",
        flexShrink: 0,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Campaigns
          </h1>
          {!loading && (
            <span style={{
              fontSize: 11, color: "var(--text-muted)", padding: "2px 8px",
              background: "var(--surface-raised)", borderRadius: 999,
              border: "1px solid var(--border)",
            }}>
              {campaigns.length}
            </span>
          )}
        </div>
        <button
          onClick={() => openCreate("company_first")}
          style={{
            display: "flex", alignItems: "center", gap: 6,
            background: "var(--text)", color: "var(--bg)",
            border: "none", borderRadius: 8, padding: "8px 16px",
            fontSize: 13, fontWeight: 500, cursor: "pointer",
          }}
        >
          <Plus size={14} /> New Campaign
        </button>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>

        {/* Error banner */}
        {error && (
          <div style={{
            padding: "10px 14px", background: "var(--red-light)", color: "var(--red)",
            borderRadius: 8, fontSize: 12, marginBottom: 18,
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            {error}
            <button
              onClick={() => setError(null)}
              style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 16 }}
            >×</button>
          </div>
        )}

        {/* Use-case cards — always visible */}
        <div style={{ marginBottom: 24 }}>
          <div style={{
            fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
            textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 12,
          }}>
            Start a New Campaign
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {(["company_first", "industry_first", "report_driven"] as CampaignType[]).map(t => (
              <UseCaseCard
                key={t}
                type={t}
                onClick={() => openCreate(t)}
              />
            ))}
          </div>
        </div>

        {/* Divider with label */}
        {(campaigns.length > 0 || loading) && (
          <div style={{
            display: "flex", alignItems: "center", gap: 12, marginBottom: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em", whiteSpace: "nowrap" }}>
              Recent Campaigns
            </div>
            <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
          </div>
        )}

        {/* Campaign list */}
        {loading ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 14 }}>
            {[0, 1, 2].map(i => (
              <div key={i} className="card" style={{ padding: "16px 18px" }}>
                <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                  <div className="skeleton" style={{ width: 34, height: 34, borderRadius: 8, flexShrink: 0 }} />
                  <div style={{ flex: 1 }}>
                    <div className="skeleton" style={{ height: 14, width: "65%", marginBottom: 7 }} />
                    <div className="skeleton" style={{ height: 11, width: "40%" }} />
                  </div>
                </div>
                <div className="skeleton" style={{ height: 11, width: "80%", marginBottom: 8 }} />
                <div className="skeleton" style={{ height: 4, width: "100%", borderRadius: 2 }} />
              </div>
            ))}
          </div>
        ) : campaigns.length === 0 ? (
          <div style={{
            padding: "48px 24px", textAlign: "center", color: "var(--text-muted)",
            background: "var(--surface-raised)", borderRadius: 12, border: "1px dashed var(--border)",
          }}>
            <Crosshair size={28} style={{ margin: "0 auto 12px", opacity: 0.3 }} />
            <div style={{ fontSize: 14, fontWeight: 500, color: "var(--text-secondary)", marginBottom: 6 }}>
              No campaigns yet
            </div>
            <div style={{ fontSize: 12, lineHeight: 1.6, maxWidth: 380, margin: "0 auto 16px" }}>
              Use the cards above to start your first campaign. Harbinger will find contacts
              and generate personalized outreach automatically.
            </div>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 14 }}>
            {campaigns.map(c => (
              <CampaignCard
                key={c.id}
                campaign={c}
                deleting={deleting === c.id}
                onRun={e => handleQuickRun(e, c.id)}
                onDelete={e => requestDelete(e, c)}
                onClick={() => router.push(`/campaigns/${c.id}`)}
              />
            ))}
          </div>
        )}
      </div>
    </>
  );
}
