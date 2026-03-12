"use client";

import { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  Plus, Building2, Globe, FileText, Users, Mail,
  Loader2, Trash2, Play, X, Crosshair,
  CheckCircle, XCircle, Clock, ChevronRight, Search,
  ChevronLeft, Tag, MapPin, Settings2, FlaskConical,
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

/* ── Campaign Wizard (4-step) ──────────────────────────── */

// Dynamic role suggestions by sector keyword
const SECTOR_ROLE_SUGGESTIONS: Record<string, string[]> = {
  fintech: ["CTO", "CISO", "Head of Digital", "VP Technology", "Head of Compliance"],
  bfsi: ["CTO", "CISO", "Head of Digital", "VP Technology", "Chief Compliance Officer"],
  healthcare: ["CTO", "CMO", "VP Operations", "Head of Digital Health", "CIO"],
  ecommerce: ["CTO", "VP Engineering", "Head of Technology", "VP Product"],
  retail: ["CTO", "VP Engineering", "Head of Technology", "VP Operations"],
  cybersecurity: ["CISO", "CTO", "VP Security", "Head of Information Security"],
  logistics: ["CTO", "VP Operations", "Head of Technology", "VP Supply Chain"],
  edtech: ["CTO", "VP Product", "Head of Technology", "Chief Academic Officer"],
  saas: ["CTO", "VP Engineering", "Head of Product", "VP Sales"],
  manufacturing: ["CTO", "VP Operations", "Head of Automation", "Plant Director"],
  default: ["CTO", "VP Engineering", "VP Product", "Head of Technology", "VP Operations"],
};

function getSectorRoles(sector: string): string[] {
  const s = sector.toLowerCase();
  for (const [key, roles] of Object.entries(SECTOR_ROLE_SUGGESTIONS)) {
    if (s.includes(key)) return roles;
  }
  return SECTOR_ROLE_SUGGESTIONS.default;
}

const TRIGGER_SIGNALS = [
  { id: "regulatory_pressure", label: "Regulatory pressure" },
  { id: "recent_funding", label: "Recent funding" },
  { id: "hiring_surge", label: "Hiring surge" },
  { id: "digital_transformation", label: "Digital transformation" },
  { id: "ipo_ma", label: "IPO / M&A" },
  { id: "competitive_pressure", label: "Competitive pressure" },
];

const BROAD_SECTORS = [
  "Fintech & Banking (BFSI)",
  "Healthcare & Life Sciences",
  "E-commerce & Retail",
  "Cybersecurity",
  "Logistics & Supply Chain",
  "EdTech",
  "SaaS / B2B Software",
  "Manufacturing & Automation",
  "Telecom & Infrastructure",
  "Real Estate & PropTech",
  "Energy & Utilities",
  "Media & Entertainment",
];

// Label input — reusable field block
function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, color: "var(--text-muted)",
      textTransform: "uppercase", letterSpacing: "0.06em",
      marginBottom: 6,
    }}>
      {children}
    </div>
  );
}

function fieldStyle(focus?: boolean): React.CSSProperties {
  return {
    width: "100%", padding: "9px 12px", borderRadius: 8, boxSizing: "border-box",
    border: `1px solid ${focus ? "var(--accent)" : "var(--border)"}`,
    background: "var(--surface-raised)", fontSize: 13, color: "var(--text)",
    outline: "none", fontFamily: "inherit",
  };
}

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
  const [step, setStep] = useState(1);

  // Step 1
  const [campaignType, setCampaignType] = useState<CampaignType>(initialType);
  const [campaignName, setCampaignName] = useState("");

  // Step 2 — Company-First
  const [companiesText, setCompaniesText] = useState("");
  const [productContext, setProductContext] = useState("");
  const [triggerReason, setTriggerReason] = useState("");

  // Step 2 — Industry-First
  const [broadSector, setBroadSector] = useState("");
  const [narrowKeyword, setNarrowKeyword] = useState("");
  const [companySize, setCompanySize] = useState<"smb" | "mid_market" | "enterprise" | "all">("all");
  const [triggerSignals, setTriggerSignals] = useState<string[]>([]);
  const [maxCompanies, setMaxCompanies] = useState(15);

  // Step 2 — Report-Driven
  const [reportMode, setReportMode] = useState<"report" | "pitch">("report");
  const [reportText, setReportText] = useState("");
  const [pitchDescription, setPitchDescription] = useState("");

  // Step 3 — Personas
  const [targetRoles, setTargetRoles] = useState<string[]>([]);
  const [customRole, setCustomRole] = useState("");
  const [seniority, setSeniority] = useState<"decision_maker" | "influencer" | "both">("both");
  const [country, setCountry] = useState("India");
  const [maxContacts, setMaxContacts] = useState(5);
  const [generateOutreach, setGenerateOutreach] = useState(true);
  const [backgroundDeep, setBackgroundDeep] = useState(false);

  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const companyCount = companiesText.split("\n").filter(l => l.trim()).length;
  const wordCount = reportText.trim().split(/\s+/).filter(Boolean).length;

  // Auto-suggest roles when sector changes
  const suggestedRoles = getSectorRoles(broadSector || campaignType.replace("_", " "));

  function toggleSignal(id: string) {
    setTriggerSignals(prev => prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]);
  }

  function toggleRole(role: string) {
    setTargetRoles(prev => prev.includes(role) ? prev.filter(r => r !== role) : [...prev, role]);
  }

  function addCustomRole() {
    const r = customRole.trim();
    if (r && !targetRoles.includes(r)) setTargetRoles(prev => [...prev, r]);
    setCustomRole("");
  }

  function validateStep(): string | null {
    if (step === 1) return null;
    if (step === 2) {
      if (campaignType === "company_first" && companyCount === 0) return "Enter at least one company name";
      if (campaignType === "industry_first" && !broadSector) return "Select a broad sector";
      if (campaignType === "report_driven") {
        if (reportMode === "report" && !reportText.trim()) return "Paste report content";
        if (reportMode === "pitch" && !pitchDescription.trim()) return "Describe your product pitch";
      }
    }
    return null;
  }

  function nextStep() {
    const err = validateStep();
    if (err) { setError(err); return; }
    setError(null);
    if (step === 2 && targetRoles.length === 0) {
      // Auto-fill roles before going to personas step
      setTargetRoles(suggestedRoles.slice(0, 3));
    }
    setStep(s => s + 1);
  }

  async function handleCreate(runNow: boolean) {
    setError(null);
    setCreating(true);
    try {
      const industry = broadSector
        ? (narrowKeyword ? `${broadSector} ${narrowKeyword}` : broadSector)
        : "";

      const req: CreateCampaignRequest = {
        campaign_type: campaignType,
        ...(campaignName.trim() ? { name: campaignName.trim() } : {}),
        config: {
          max_companies: maxCompanies,
          max_contacts_per_company: maxContacts,
          generate_outreach: generateOutreach,
          target_roles: targetRoles,
          country: country.trim(),
          background_deep: backgroundDeep,
          seniority_filter: seniority,
          company_size_filter: companySize,
          trigger_signals: triggerSignals,
          product_context: productContext.trim() || pitchDescription.trim(),
          narrow_keyword: narrowKeyword.trim(),
        } as any,
      };

      if (campaignType === "company_first") {
        const lines = companiesText.split("\n").map(l => l.trim()).filter(Boolean);
        req.companies = lines.map(name => ({
          company_name: name,
          context: triggerReason.trim() || undefined,
        }));
      } else if (campaignType === "industry_first") {
        req.industry = industry;
      } else {
        req.report_text = reportMode === "report" ? reportText.trim() : pitchDescription.trim();
      }

      const campaign = await api.createCampaign(req);
      if (runNow) {
        await api.runCampaign(campaign.id);
      }
      router.push(`/campaigns/${campaign.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create campaign");
      setCreating(false);
    }
  }

  const STEP_LABELS = ["Type", "Details", "Targeting", "Review"];

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
          width: "min(560px, 96vw)", maxHeight: "92vh", overflow: "auto",
          border: "1px solid var(--border)", boxShadow: "0 24px 80px rgba(0,0,0,0.3)",
          display: "flex", flexDirection: "column",
        }}
      >
        {/* Header */}
        <div style={{
          padding: "18px 22px 14px",
          borderBottom: "1px solid var(--border)",
          flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.01em" }}>
                New Campaign
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1 }}>
                Step {step} of 4 — {STEP_LABELS[step - 1]}
              </div>
            </div>
            <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 6, borderRadius: 6, display: "flex" }}>
              <X size={15} />
            </button>
          </div>
          {/* Step progress */}
          <div style={{ display: "flex", gap: 4 }}>
            {STEP_LABELS.map((label, i) => (
              <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", gap: 3 }}>
                <div style={{
                  height: 3, borderRadius: 2,
                  background: i + 1 <= step ? "var(--accent)" : "var(--border)",
                  transition: "background 200ms",
                }} />
                <div style={{ fontSize: 9, color: i + 1 <= step ? "var(--accent)" : "var(--text-xmuted)", textAlign: "center", fontWeight: i + 1 === step ? 600 : 400 }}>
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Body */}
        <div style={{ padding: "18px 22px 20px", flex: 1, overflow: "auto" }}>

          {/* Error */}
          {error && (
            <div style={{ padding: "8px 12px", background: "var(--red-light)", color: "var(--red)", borderRadius: 8, fontSize: 12, marginBottom: 14, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              {error}
              <button onClick={() => setError(null)} style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 14, padding: 0 }}>×</button>
            </div>
          )}

          {/* ── Step 1: Type + Name ── */}
          {step === 1 && (
            <>
              <div style={{ marginBottom: 18 }}>
                <FieldLabel>Campaign Name <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>(optional)</span></FieldLabel>
                <input
                  type="text"
                  value={campaignName}
                  onChange={e => setCampaignName(e.target.value)}
                  placeholder="e.g. Q2 Fintech Outreach"
                  autoFocus
                  style={fieldStyle()}
                />
              </div>

              <FieldLabel>Campaign Type</FieldLabel>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
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
                        border: `1.5px solid ${active ? "var(--accent)" : "var(--border)"}`,
                        background: active ? "var(--accent-light)" : "var(--surface-raised)",
                        cursor: "pointer", transition: "all 130ms",
                      }}
                    >
                      <div style={{
                        width: 34, height: 34, borderRadius: 8, flexShrink: 0,
                        background: active ? "var(--accent)" : "var(--surface)",
                        color: active ? "#fff" : "var(--text-muted)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        transition: "all 130ms",
                      }}>
                        {m.icon}
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: 1 }}>{m.label}</div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{m.subtitle}</div>
                      </div>
                      <div style={{
                        width: 15, height: 15, borderRadius: "50%", flexShrink: 0,
                        border: `2px solid ${active ? "var(--accent)" : "var(--border)"}`,
                        background: active ? "var(--accent)" : "transparent",
                        display: "flex", alignItems: "center", justifyContent: "center",
                      }}>
                        {active && <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#fff" }} />}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {/* ── Step 2A: Company-First details ── */}
          {step === 2 && campaignType === "company_first" && (
            <>
              <div style={{ marginBottom: 16 }}>
                <FieldLabel>Target Companies <span style={{ fontWeight: 400, textTransform: "none" }}>— one per line</span></FieldLabel>
                <textarea
                  value={companiesText}
                  onChange={e => setCompaniesText(e.target.value)}
                  placeholder={"NVIDIA\nInfosys\nFreshworks\nRazorpay"}
                  rows={5}
                  autoFocus
                  style={{ ...fieldStyle(), resize: "vertical", lineHeight: 1.7 }}
                />
                {companyCount > 0 && (
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                    {companyCount} compan{companyCount === 1 ? "y" : "ies"} queued
                  </div>
                )}
              </div>

              <div style={{ marginBottom: 16 }}>
                <FieldLabel>Product / Service to Pitch <span style={{ fontWeight: 400, textTransform: "none" }}>(optional)</span></FieldLabel>
                <input
                  type="text"
                  value={productContext}
                  onChange={e => setProductContext(e.target.value)}
                  placeholder="e.g. AI Compliance Suite, Logistics SaaS, Cybersecurity Platform"
                  style={fieldStyle()}
                />
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                  Used to personalise outreach emails and target the right contacts
                </div>
              </div>

              <div>
                <FieldLabel>Why Now? <span style={{ fontWeight: 400, textTransform: "none" }}>(optional trigger context)</span></FieldLabel>
                <input
                  type="text"
                  value={triggerReason}
                  onChange={e => setTriggerReason(e.target.value)}
                  placeholder="e.g. Recent RBI mandate, Series B funding, hiring surge"
                  style={fieldStyle()}
                />
              </div>
            </>
          )}

          {/* ── Step 2B: Industry-First details ── */}
          {step === 2 && campaignType === "industry_first" && (
            <>
              <div style={{ marginBottom: 14 }}>
                <FieldLabel>Broad Sector</FieldLabel>
                <select
                  value={broadSector}
                  onChange={e => setBroadSector(e.target.value)}
                  autoFocus
                  style={{ ...fieldStyle(), appearance: "none", WebkitAppearance: "none" }}
                >
                  <option value="">— Select a sector —</option>
                  {BROAD_SECTORS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
                {broadSector && (
                  <div style={{ fontSize: 11, color: "var(--accent)", marginTop: 4 }}>
                    ✦ Suggested roles: {getSectorRoles(broadSector).slice(0, 3).join(", ")}
                  </div>
                )}
              </div>

              <div style={{ marginBottom: 14 }}>
                <FieldLabel>Narrow Focus <span style={{ fontWeight: 400, textTransform: "none" }}>(optional)</span></FieldLabel>
                <input
                  type="text"
                  value={narrowKeyword}
                  onChange={e => setNarrowKeyword(e.target.value)}
                  placeholder="e.g. payment infrastructure, HIPAA compliance, SMB lending"
                  style={fieldStyle()}
                />
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>Leave blank to discover broadly across the full sector</div>
              </div>

              <div style={{ marginBottom: 14 }}>
                <FieldLabel>Company Size</FieldLabel>
                <div style={{ display: "flex", gap: 5 }}>
                  {(["smb", "mid_market", "enterprise", "all"] as const).map(s => (
                    <button
                      key={s}
                      onClick={() => setCompanySize(s)}
                      style={{
                        flex: 1, padding: "6px 4px", fontSize: 11, fontWeight: companySize === s ? 600 : 400,
                        borderRadius: 7, border: `1.5px solid ${companySize === s ? "var(--accent)" : "var(--border)"}`,
                        background: companySize === s ? "var(--accent-light)" : "var(--surface-raised)",
                        color: companySize === s ? "var(--accent)" : "var(--text-secondary)",
                        cursor: "pointer", transition: "all 120ms",
                      }}
                    >
                      {s === "smb" ? "SMB" : s === "mid_market" ? "Mid-market" : s === "enterprise" ? "Enterprise" : "All"}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: 14 }}>
                <FieldLabel>Trigger Signals <span style={{ fontWeight: 400, textTransform: "none" }}>(optional — narrows company discovery)</span></FieldLabel>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                  {TRIGGER_SIGNALS.map(sig => {
                    const on = triggerSignals.includes(sig.id);
                    return (
                      <button
                        key={sig.id}
                        onClick={() => toggleSignal(sig.id)}
                        style={{
                          padding: "4px 10px", fontSize: 11, borderRadius: 20,
                          border: `1.5px solid ${on ? "var(--accent)" : "var(--border)"}`,
                          background: on ? "var(--accent-light)" : "var(--surface-raised)",
                          color: on ? "var(--accent)" : "var(--text-secondary)",
                          cursor: "pointer", fontWeight: on ? 600 : 400, transition: "all 120ms",
                        }}
                      >
                        {on ? "✓ " : ""}{sig.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <FieldLabel>Max Companies to Discover</FieldLabel>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <input
                    type="range" min={5} max={50} step={5} value={maxCompanies}
                    onChange={e => setMaxCompanies(Number(e.target.value))}
                    style={{ flex: 1, accentColor: "var(--accent)" }}
                  />
                  <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", minWidth: 28 }}>{maxCompanies}</span>
                </div>
              </div>
            </>
          )}

          {/* ── Step 2C: Report-Driven details ── */}
          {step === 2 && campaignType === "report_driven" && (
            <>
              <div style={{ marginBottom: 14 }}>
                <FieldLabel>Source Type</FieldLabel>
                <div style={{ display: "flex", border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden" }}>
                  {(["report", "pitch"] as const).map(mode => (
                    <button
                      key={mode}
                      onClick={() => setReportMode(mode)}
                      style={{
                        flex: 1, padding: "8px", fontSize: 12, fontWeight: reportMode === mode ? 600 : 400,
                        background: reportMode === mode ? "var(--accent)" : "var(--surface-raised)",
                        color: reportMode === mode ? "#fff" : "var(--text-secondary)",
                        border: "none", cursor: "pointer", transition: "all 120ms",
                      }}
                    >
                      {mode === "report" ? "📄 Paste Report / Article" : "📦 Describe Product Pitch"}
                    </button>
                  ))}
                </div>
              </div>

              {reportMode === "report" ? (
                <div>
                  <FieldLabel>Report Content</FieldLabel>
                  <textarea
                    value={reportText}
                    onChange={e => setReportText(e.target.value)}
                    placeholder="Paste an industry report, news article, or analyst note — Harbinger will extract target companies automatically."
                    rows={7}
                    autoFocus
                    style={{ ...fieldStyle(), resize: "vertical", lineHeight: 1.6 }}
                  />
                  {wordCount > 10 && (
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                      {wordCount} words · Harbinger will extract companies on the backend
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <FieldLabel>Describe Your Product / Pitch</FieldLabel>
                  <textarea
                    value={pitchDescription}
                    onChange={e => setPitchDescription(e.target.value)}
                    placeholder={"Describe what you're selling and the problem it solves.\nExample: We help mid-size NBFCs automate RBI compliance reporting, reducing audit prep time by 60%."}
                    rows={5}
                    autoFocus
                    style={{ ...fieldStyle(), resize: "vertical", lineHeight: 1.6 }}
                  />
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                    Harbinger will identify companies most likely to need this
                  </div>
                </div>
              )}
            </>
          )}

          {/* ── Step 3: Targeting / Personas ── */}
          {step === 3 && (
            <>
              <div style={{ marginBottom: 16 }}>
                <FieldLabel>Target Roles</FieldLabel>
                {targetRoles.length === 0 && (
                  <div style={{ fontSize: 11, color: "var(--accent)", marginBottom: 6 }}>
                    ✦ Suggested for {campaignType === "industry_first" ? broadSector || "this sector" : "this campaign"}:
                  </div>
                )}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
                  {suggestedRoles.filter(r => !targetRoles.includes(r)).map(r => (
                    <button
                      key={r}
                      onClick={() => toggleRole(r)}
                      style={{
                        padding: "3px 9px", fontSize: 11, borderRadius: 20,
                        border: "1.5px dashed var(--border)",
                        background: "var(--surface-raised)", color: "var(--text-muted)",
                        cursor: "pointer",
                      }}
                    >
                      + {r}
                    </button>
                  ))}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
                  {targetRoles.map(r => (
                    <button
                      key={r}
                      onClick={() => toggleRole(r)}
                      style={{
                        padding: "3px 10px", fontSize: 11, borderRadius: 20,
                        border: "1.5px solid var(--accent)", background: "var(--accent-light)",
                        color: "var(--accent)", cursor: "pointer", fontWeight: 500,
                      }}
                    >
                      {r} ×
                    </button>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input
                    type="text"
                    value={customRole}
                    onChange={e => setCustomRole(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && addCustomRole()}
                    placeholder="Add custom role…"
                    style={{ ...fieldStyle(), flex: 1 }}
                  />
                  <button
                    onClick={addCustomRole}
                    style={{
                      padding: "8px 12px", borderRadius: 8, fontSize: 12, fontWeight: 500,
                      border: "1px solid var(--border)", background: "var(--surface-raised)",
                      color: "var(--text-secondary)", cursor: "pointer",
                    }}
                  >
                    Add
                  </button>
                </div>
                {targetRoles.length === 0 && (
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 5 }}>
                    Skip to let Harbinger auto-select based on trend type
                  </div>
                )}
              </div>

              <div style={{ marginBottom: 14 }}>
                <FieldLabel>Seniority</FieldLabel>
                <div style={{ display: "flex", gap: 5 }}>
                  {([
                    { v: "decision_maker", label: "Decision-Makers", hint: "C-suite, VP, Director" },
                    { v: "influencer", label: "Influencers", hint: "Manager, Lead, Senior" },
                    { v: "both", label: "Both", hint: "Recommended" },
                  ] as const).map(opt => (
                    <button
                      key={opt.v}
                      onClick={() => setSeniority(opt.v)}
                      style={{
                        flex: 1, padding: "7px 6px", fontSize: 11, fontWeight: seniority === opt.v ? 600 : 400,
                        borderRadius: 8, border: `1.5px solid ${seniority === opt.v ? "var(--accent)" : "var(--border)"}`,
                        background: seniority === opt.v ? "var(--accent-light)" : "var(--surface-raised)",
                        color: seniority === opt.v ? "var(--accent)" : "var(--text-secondary)",
                        cursor: "pointer", transition: "all 120ms", textAlign: "center",
                      }}
                    >
                      <div>{opt.label}</div>
                      <div style={{ fontSize: 9, fontWeight: 400, color: "var(--text-muted)", marginTop: 1 }}>{opt.hint}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
                <div>
                  <FieldLabel>Country / Region</FieldLabel>
                  <input
                    type="text"
                    value={country}
                    onChange={e => setCountry(e.target.value)}
                    list="country-suggestions"
                    style={fieldStyle()}
                  />
                  <datalist id="country-suggestions">
                    {["India", "United States", "United Kingdom", "Singapore", "UAE", "Australia", "Global"].map(c => (
                      <option key={c} value={c} />
                    ))}
                  </datalist>
                </div>
                <div>
                  <FieldLabel>Contacts per Company</FieldLabel>
                  <input
                    type="number" min={1} max={10} value={maxContacts}
                    onChange={e => setMaxContacts(Number(e.target.value))}
                    style={fieldStyle()}
                  />
                </div>
              </div>

              <div style={{ padding: "12px 14px", background: "var(--surface-raised)", borderRadius: 10, border: "1px solid var(--border)" }}>
                <FieldLabel>Options</FieldLabel>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 12, color: "var(--text-secondary)" }}>
                    <input type="checkbox" checked={generateOutreach} onChange={e => setGenerateOutreach(e.target.checked)} style={{ accentColor: "var(--accent)", width: 14, height: 14 }} />
                    <span>Generate personalised outreach emails <span style={{ color: "var(--text-muted)" }}>(recommended)</span></span>
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 12, color: "var(--text-secondary)" }}>
                    <input type="checkbox" checked={backgroundDeep} onChange={e => setBackgroundDeep(e.target.checked)} style={{ accentColor: "var(--accent)", width: 14, height: 14 }} />
                    <span>Deep background research <span style={{ color: "var(--text-muted)" }}>(slower — 30-80s per company)</span></span>
                  </label>
                </div>
              </div>
            </>
          )}

          {/* ── Step 4: Review ── */}
          {step === 4 && (
            <>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: 14 }}>
                Review your campaign before launching
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  { label: "Type", value: TYPE_META[campaignType].label },
                  ...(campaignName ? [{ label: "Name", value: campaignName }] : []),
                  campaignType === "company_first"
                    ? { label: "Companies", value: `${companyCount} compan${companyCount === 1 ? "y" : "ies"}` }
                    : campaignType === "industry_first"
                    ? { label: "Sector", value: broadSector + (narrowKeyword ? ` — ${narrowKeyword}` : "") }
                    : { label: "Source", value: reportMode === "report" ? `Report (${wordCount} words)` : "Product pitch" },
                  targetRoles.length > 0 ? { label: "Target Roles", value: targetRoles.join(", ") } : null,
                  { label: "Seniority", value: seniority === "both" ? "Decision-Makers + Influencers" : seniority === "decision_maker" ? "Decision-Makers only" : "Influencers only" },
                  { label: "Country", value: country },
                  { label: "Contacts/Company", value: String(maxContacts) },
                  productContext || pitchDescription ? { label: "Product", value: productContext || pitchDescription } : null,
                  campaignType === "industry_first" && companySize !== "all" ? { label: "Company Size", value: companySize } : null,
                  triggerSignals.length > 0 ? { label: "Trigger Signals", value: triggerSignals.map(s => TRIGGER_SIGNALS.find(t => t.id === s)?.label || s).join(", ") } : null,
                  { label: "Outreach Emails", value: generateOutreach ? "Yes" : "No" },
                  backgroundDeep ? { label: "Deep Research", value: "Enabled (slower)" } : null,
                ].filter(Boolean).map((row, i) => (
                  <div key={i} style={{ display: "flex", gap: 10, padding: "8px 12px", background: i % 2 === 0 ? "var(--surface-raised)" : "transparent", borderRadius: 6 }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", minWidth: 110, textTransform: "uppercase", letterSpacing: "0.05em", paddingTop: 1 }}>{row!.label}</div>
                    <div style={{ fontSize: 12, color: "var(--text)", lineHeight: 1.5 }}>{row!.value}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 12, padding: "10px 12px", background: "var(--accent-light)", borderRadius: 8, border: "1px solid var(--accent-mid)", fontSize: 12, color: "var(--text-secondary)" }}>
                <strong style={{ color: "var(--accent)" }}>Tip:</strong> Save as draft to review before running, or create and run immediately.
              </div>
            </>
          )}
        </div>

        {/* Footer nav */}
        <div style={{
          padding: "14px 22px 18px",
          borderTop: "1px solid var(--border)",
          display: "flex", gap: 8, flexShrink: 0,
        }}>
          {step > 1 && (
            <button
              onClick={() => setStep(s => s - 1)}
              style={{
                padding: "9px 14px", fontSize: 12, fontWeight: 500, borderRadius: 8,
                border: "1px solid var(--border)", background: "var(--surface)",
                color: "var(--text-secondary)", cursor: "pointer",
                display: "flex", alignItems: "center", gap: 5,
              }}
            >
              <ChevronLeft size={13} /> Back
            </button>
          )}
          {step === 1 && (
            <button
              onClick={onClose}
              style={{ padding: "9px 14px", fontSize: 12, fontWeight: 500, borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text-secondary)", cursor: "pointer" }}
            >
              Cancel
            </button>
          )}
          <div style={{ flex: 1 }} />
          {step < 4 && (
            <button
              onClick={nextStep}
              style={{
                padding: "9px 20px", fontSize: 13, fontWeight: 600, borderRadius: 8,
                background: "var(--accent)", color: "#fff", border: "none", cursor: "pointer",
                display: "flex", alignItems: "center", gap: 6,
              }}
            >
              Next <ChevronRight size={13} />
            </button>
          )}
          {step === 4 && (
            <>
              <button
                onClick={() => handleCreate(false)}
                disabled={creating}
                style={{
                  padding: "9px 16px", fontSize: 12, fontWeight: 500, borderRadius: 8,
                  border: "1px solid var(--border)", background: "var(--surface)",
                  color: "var(--text-secondary)", cursor: creating ? "not-allowed" : "pointer",
                }}
              >
                Save Draft
              </button>
              <button
                onClick={() => handleCreate(true)}
                disabled={creating}
                style={{
                  padding: "9px 18px", fontSize: 13, fontWeight: 600, borderRadius: 8,
                  background: creating ? "var(--surface-raised)" : "var(--accent)",
                  color: creating ? "var(--text-muted)" : "#fff",
                  border: "none", cursor: creating ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", gap: 6, transition: "all 150ms",
                }}
              >
                {creating
                  ? <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} />
                  : <Crosshair size={13} />
                }
                {creating ? "Creating…" : "Create & Run Now"}
              </button>
            </>
          )}
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

  // Search / filter
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState<CampaignType | "all">("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  // Modal state
  const [showCreate, setShowCreate] = useState(false);
  const [createType, setCreateType] = useState<CampaignType>("company_first");

  // Delete state
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const filteredCampaigns = useMemo(() => {
    let list = campaigns;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(c =>
        (c.name || "").toLowerCase().includes(q) ||
        c.id.toLowerCase().includes(q)
      );
    }
    if (typeFilter !== "all") list = list.filter(c => c.campaign_type === typeFilter);
    if (statusFilter !== "all") list = list.filter(c => c.status === statusFilter);
    return list;
  }, [campaigns, searchQuery, typeFilter, statusFilter]);

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
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
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
                {filteredCampaigns.length}{filteredCampaigns.length !== campaigns.length ? ` of ${campaigns.length}` : ""}
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

        {/* Search + filter row */}
        {campaigns.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <div style={{ position: "relative", flex: "1 1 200px", maxWidth: 340 }}>
              <Search size={12} style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }} />
              <input
                type="text"
                placeholder="Search campaigns..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                style={{
                  width: "100%", padding: "6px 10px 6px 28px", borderRadius: 7,
                  border: "1px solid var(--border)", background: "var(--surface-raised)",
                  fontSize: 12, color: "var(--text)", outline: "none", boxSizing: "border-box",
                }}
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  style={{ position: "absolute", right: 7, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", display: "flex" }}
                >
                  <X size={11} />
                </button>
              )}
            </div>
            <select
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value as CampaignType | "all")}
              style={{
                padding: "5px 10px", borderRadius: 7, border: "1px solid var(--border)",
                background: typeFilter !== "all" ? "var(--accent-light)" : "var(--surface-raised)",
                color: typeFilter !== "all" ? "var(--accent)" : "var(--text-muted)",
                fontSize: 11, fontWeight: typeFilter !== "all" ? 600 : 400, cursor: "pointer", outline: "none",
              }}
            >
              <option value="all">All types</option>
              <option value="company_first">Company-First</option>
              <option value="industry_first">Industry-First</option>
              <option value="report_driven">Report-Driven</option>
            </select>
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              style={{
                padding: "5px 10px", borderRadius: 7, border: "1px solid var(--border)",
                background: statusFilter !== "all" ? "var(--accent-light)" : "var(--surface-raised)",
                color: statusFilter !== "all" ? "var(--accent)" : "var(--text-muted)",
                fontSize: 11, fontWeight: statusFilter !== "all" ? 600 : 400, cursor: "pointer", outline: "none",
              }}
            >
              <option value="all">All statuses</option>
              <option value="draft">Draft</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
            {(searchQuery || typeFilter !== "all" || statusFilter !== "all") && (
              <button
                onClick={() => { setSearchQuery(""); setTypeFilter("all"); setStatusFilter("all"); }}
                style={{
                  display: "flex", alignItems: "center", gap: 3, padding: "5px 10px",
                  borderRadius: 7, border: "1px solid var(--red)", background: "var(--red-light)",
                  color: "var(--red)", fontSize: 11, fontWeight: 500, cursor: "pointer",
                }}
              >
                <X size={10} /> Clear
              </button>
            )}
          </div>
        )}
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
              {filteredCampaigns.length !== campaigns.length
                ? `${filteredCampaigns.length} of ${campaigns.length} Campaigns`
                : "Recent Campaigns"}
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
        ) : filteredCampaigns.length === 0 ? (
          <div style={{
            padding: "32px 24px", textAlign: "center", color: "var(--text-muted)",
            background: "var(--surface-raised)", borderRadius: 12, border: "1px dashed var(--border)",
          }}>
            <Search size={24} style={{ margin: "0 auto 10px", opacity: 0.3 }} />
            <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary)", marginBottom: 4 }}>
              No campaigns match your filters
            </div>
            <button
              onClick={() => { setSearchQuery(""); setTypeFilter("all"); setStatusFilter("all"); }}
              style={{ fontSize: 12, color: "var(--accent)", background: "none", border: "none", cursor: "pointer", marginTop: 4 }}
            >
              Clear filters
            </button>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 14 }}>
            {filteredCampaigns.map(c => (
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
