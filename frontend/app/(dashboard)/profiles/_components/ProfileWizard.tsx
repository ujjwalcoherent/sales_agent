"use client";

import { useState, useCallback } from "react";
import { X, Plus, Trash2, Check, ChevronRight, ChevronLeft, Zap, Building2, BarChart3, Cpu } from "lucide-react";
import { createProfile, updateProfile } from "@/lib/api";
import type {
  UserProfile,
  CreateProfileRequest,
  IndustryTarget,
  ProductEntry,
} from "@/lib/types";

// ── Helpers ────────────────────────────────────────────────────────────────

function emptyIndustry(): IndustryTarget {
  return {
    industry_id: crypto.randomUUID(),
    display_name: "",
    order: "1st",
    first_order_description: "",
    second_order_description: "",
    use_builtin: true,
  };
}

function emptyProduct(): ProductEntry {
  return {
    name: "",
    value_prop: "",
    case_studies: [],
    target_roles: [],
    relevant_event_types: [],
  };
}

// ── Step config ───────────────────────────────────────────────────────────

const STEPS = [
  { id: 1, title: "About You", desc: "Identity & region" },
  { id: 2, title: "Pipeline Mode", desc: "How you target" },
  { id: 3, title: "What You Sell", desc: "Products & value props" },
  { id: 4, title: "Email & Scoring", desc: "Outreach config" },
  { id: 5, title: "Review", desc: "Confirm & save" },
] as const;

type StepId = (typeof STEPS)[number]["id"];

// ── Path mode config ───────────────────────────────────────────────────────

const PATH_MODES: {
  value: UserProfile["path_preference"];
  label: string;
  subtitle: string;
  icon: React.ReactNode;
  color: string;
  bg: string;
}[] = [
  {
    value: "auto",
    label: "Auto",
    subtitle: "Let Harbinger decide the best path based on available signals",
    icon: <Zap size={18} />,
    color: "var(--accent)",
    bg: "var(--accent-light)",
  },
  {
    value: "industry_first",
    label: "Industry-First",
    subtitle: "Track trends across target industries, surface companies from signals",
    icon: <BarChart3 size={18} />,
    color: "var(--green)",
    bg: "var(--green-light)",
  },
  {
    value: "company_first",
    label: "Company-First",
    subtitle: "Monitor your named account list directly for buying signals",
    icon: <Building2 size={18} />,
    color: "var(--blue)",
    bg: "#EBF2F9",
  },
  {
    value: "report_driven",
    label: "Report-Driven",
    subtitle: "Build intelligence from a specific research brief or report",
    icon: <Cpu size={18} />,
    color: "#6B3FA0",
    bg: "#EDE8F5",
  },
];

// ── Shared field styles ────────────────────────────────────────────────────

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  color: "var(--text-muted)",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  display: "block",
  marginBottom: 6,
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "9px 12px",
  borderRadius: 8,
  border: "1px solid var(--border)",
  background: "var(--bg)",
  fontSize: 13,
  color: "var(--text)",
  outline: "none",
  boxSizing: "border-box",
  fontFamily: "inherit",
  transition: "border-color 150ms",
};

const textareaStyle: React.CSSProperties = {
  ...inputStyle,
  resize: "vertical",
  lineHeight: 1.6,
  minHeight: 80,
};

const fieldGroupStyle: React.CSSProperties = { marginBottom: 18 };

// ── Props ──────────────────────────────────────────────────────────────────

interface Props {
  profile: UserProfile | null;
  onClose: () => void;
  onSaved: (profile: UserProfile) => void;
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function ProfileWizard({ profile, onClose, onSaved }: Props) {
  const isNew = profile === null;

  // ── Step state ──
  const [step, setStep] = useState<StepId>(1);
  const [direction, setDirection] = useState<"forward" | "back">("forward");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Step 1: About You ──
  const [userName, setUserName] = useState(profile?.user_name ?? "");
  const [ownCompany, setOwnCompany] = useState(profile?.own_company ?? "");
  const [region, setRegion] = useState(profile?.region ?? "IN");

  // ── Step 2: Pipeline Mode ──
  const [pathPreference, setPathPreference] = useState<UserProfile["path_preference"]>(
    profile?.path_preference ?? "auto"
  );
  const [industries, setIndustries] = useState<IndustryTarget[]>(
    profile?.target_industries ?? []
  );
  const [accountText, setAccountText] = useState(
    (profile?.account_list ?? []).join("\n")
  );
  const [reportTitle, setReportTitle] = useState(profile?.report_title ?? "");
  const [reportSummary, setReportSummary] = useState(profile?.report_summary ?? "");

  // ── Step 3: Products ──
  const [products, setProducts] = useState<ProductEntry[]>(
    profile?.own_products ?? []
  );
  const [productRolesText, setProductRolesText] = useState<string[]>(
    (profile?.own_products ?? []).map((p) => p.target_roles.join(", "))
  );
  const [productCasesText, setProductCasesText] = useState<string[]>(
    (profile?.own_products ?? []).map((p) => p.case_studies.join(", "))
  );
  const [productEventsText, setProductEventsText] = useState<string[]>(
    (profile?.own_products ?? []).map((p) => p.relevant_event_types.join(", "))
  );

  // ── Step 4: Email & Scoring ──
  const [fromName, setFromName] = useState(profile?.email_config?.from_name ?? "");
  const [fromEmail, setFromEmail] = useState(profile?.email_config?.from_email ?? "");
  const [minLeadScore, setMinLeadScore] = useState(profile?.min_lead_score ?? 0.5);

  // ── Navigation helpers ──────────────────────────────────────────────────

  function goNext() {
    if (step === 1 && !userName.trim()) {
      setError("Your name is required");
      return;
    }
    setError(null);
    setDirection("forward");
    setStep((s) => Math.min(5, s + 1) as StepId);
  }

  function goBack() {
    setError(null);
    setDirection("back");
    setStep((s) => Math.max(1, s - 1) as StepId);
  }

  function jumpTo(s: StepId) {
    if (s < step) {
      setDirection("back");
    } else {
      setDirection("forward");
    }
    setStep(s);
  }

  // ── Product helpers ──────────────────────────────────────────────────────

  const addProduct = useCallback(() => {
    setProducts((p) => [...p, emptyProduct()]);
    setProductRolesText((t) => [...t, ""]);
    setProductCasesText((t) => [...t, ""]);
    setProductEventsText((t) => [...t, ""]);
  }, []);

  const removeProduct = useCallback((idx: number) => {
    setProducts((p) => p.filter((_, i) => i !== idx));
    setProductRolesText((t) => t.filter((_, i) => i !== idx));
    setProductCasesText((t) => t.filter((_, i) => i !== idx));
    setProductEventsText((t) => t.filter((_, i) => i !== idx));
  }, []);

  const updateProduct = useCallback((idx: number, patch: Partial<ProductEntry>) => {
    setProducts((p) => p.map((item, i) => (i === idx ? { ...item, ...patch } : item)));
  }, []);

  // ── Industry helpers ─────────────────────────────────────────────────────

  const addIndustry = useCallback(() => {
    setIndustries((list) => [...list, emptyIndustry()]);
  }, []);

  const removeIndustry = useCallback((idx: number) => {
    setIndustries((list) => list.filter((_, i) => i !== idx));
  }, []);

  const updateIndustry = useCallback((idx: number, patch: Partial<IndustryTarget>) => {
    setIndustries((list) => list.map((item, i) => (i === idx ? { ...item, ...patch } : item)));
  }, []);

  // ── Save ──────────────────────────────────────────────────────────────────

  async function handleSave() {
    if (!userName.trim()) {
      setError("Your name is required");
      setStep(1);
      return;
    }

    const mergedProducts: ProductEntry[] = products.map((p, i) => ({
      ...p,
      case_studies: (productCasesText[i] ?? "").split(",").map((s) => s.trim()).filter(Boolean),
      target_roles: (productRolesText[i] ?? "").split(",").map((s) => s.trim()).filter(Boolean),
      relevant_event_types: (productEventsText[i] ?? "").split(",").map((s) => s.trim()).filter(Boolean),
    }));

    const payload: CreateProfileRequest = {
      user_name: userName.trim(),
      own_company: ownCompany.trim(),
      region,
      path_preference: pathPreference,
      min_lead_score: minLeadScore,
      target_industries: industries,
      own_products: mergedProducts,
      account_list: accountText.split("\n").map((s) => s.trim()).filter(Boolean),
      report_title: reportTitle.trim(),
      report_summary: reportSummary.trim(),
      contact_hierarchy: profile?.contact_hierarchy ?? [],
      email_config: { from_name: fromName.trim(), from_email: fromEmail.trim() },
    };

    setSaving(true);
    setError(null);
    try {
      const saved = isNew
        ? await createProfile(payload)
        : await updateProfile(profile.profile_id, payload);
      onSaved(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  // ── Step renderers ─────────────────────────────────────────────────────

  function renderStep1() {
    return (
      <div>
        <div style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--text)", marginBottom: 6, letterSpacing: "-0.02em" }}>
            Tell us about yourself
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
            This anchors every pipeline run to your identity and market.
          </p>
        </div>

        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Your name *</label>
          <input
            style={inputStyle}
            type="text"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            placeholder="e.g. Arjun Sharma"
            autoFocus
          />
        </div>

        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Company</label>
          <input
            style={inputStyle}
            type="text"
            value={ownCompany}
            onChange={(e) => setOwnCompany(e.target.value)}
            placeholder="e.g. Acme Consulting"
          />
        </div>

        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Region</label>
          <select
            style={{ ...inputStyle, cursor: "pointer" }}
            value={region}
            onChange={(e) => setRegion(e.target.value)}
          >
            <option value="IN">India</option>
            <option value="US">United States</option>
            <option value="EU">Europe</option>
            <option value="SEA">Southeast Asia</option>
            <option value="global">Global</option>
          </select>
        </div>
      </div>
    );
  }

  function renderStep2() {
    return (
      <div>
        <div style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--text)", marginBottom: 6, letterSpacing: "-0.02em" }}>
            How should Harbinger find leads?
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
            Choose a pipeline mode. You can change this any time.
          </p>
        </div>

        {/* Mode cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 28 }}>
          {PATH_MODES.map((mode) => {
            const active = pathPreference === mode.value;
            return (
              <button
                key={mode.value}
                onClick={() => setPathPreference(mode.value)}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 14,
                  padding: "14px 16px",
                  borderRadius: 10,
                  border: `1.5px solid ${active ? mode.color : "var(--border)"}`,
                  background: active ? mode.bg : "var(--surface)",
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "border-color 150ms, background 150ms",
                }}
              >
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 8,
                    background: active ? mode.color : "var(--surface-raised)",
                    color: active ? "#fff" : "var(--text-muted)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    transition: "background 150ms, color 150ms",
                  }}
                >
                  {mode.icon}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: active ? mode.color : "var(--text)", marginBottom: 3 }}>
                    {mode.label}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
                    {mode.subtitle}
                  </div>
                </div>
                {active && (
                  <div
                    style={{
                      width: 18,
                      height: 18,
                      borderRadius: "50%",
                      background: mode.color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      marginTop: 2,
                    }}
                  >
                    <Check size={11} color="#fff" strokeWidth={3} />
                  </div>
                )}
              </button>
            );
          })}
        </div>

        {/* Conditional sub-fields */}
        {pathPreference === "industry_first" && (
          <div>
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: "var(--green-light)",
                border: "1px solid rgba(45,106,79,0.15)",
                fontSize: 12,
                color: "var(--green)",
                marginBottom: 16,
                lineHeight: 1.6,
              }}
            >
              <strong>1st Order</strong> = direct industry companies. <strong>2nd Order</strong> = companies that service or supply that industry.
            </div>

            {industries.length === 0 && (
              <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, padding: "16px 0" }}>
                No industries added yet
              </div>
            )}

            {industries.map((ind, idx) => (
              <div
                key={ind.industry_id}
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: 10,
                  padding: "14px",
                  marginBottom: 10,
                  background: "var(--surface-raised)",
                }}
              >
                <div style={{ display: "flex", gap: 10, alignItems: "flex-start", marginBottom: 10 }}>
                  <div style={{ flex: 1 }}>
                    <label style={{ ...labelStyle, marginBottom: 4 }}>Industry name</label>
                    <input
                      style={inputStyle}
                      type="text"
                      value={ind.display_name}
                      onChange={(e) => updateIndustry(idx, { display_name: e.target.value })}
                      placeholder="e.g. Fintech"
                    />
                  </div>
                  <div style={{ width: 120 }}>
                    <label style={{ ...labelStyle, marginBottom: 4 }}>Order</label>
                    <select
                      style={{ ...inputStyle, cursor: "pointer", width: "100%" }}
                      value={ind.order}
                      onChange={(e) => updateIndustry(idx, { order: e.target.value as IndustryTarget["order"] })}
                    >
                      <option value="1st">1st Order</option>
                      <option value="2nd">2nd Order</option>
                      <option value="both">Both</option>
                    </select>
                  </div>
                  <button
                    onClick={() => removeIndustry(idx)}
                    style={{
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      color: "var(--red)",
                      padding: "4px",
                      marginTop: 20,
                      borderRadius: 4,
                      display: "flex",
                    }}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>

                {(ind.order === "1st" || ind.order === "both") && (
                  <div style={{ marginBottom: ind.order === "both" ? 10 : 0 }}>
                    <label style={{ ...labelStyle, marginBottom: 4 }}>1st-order description</label>
                    <textarea
                      style={{ ...textareaStyle, minHeight: 52 }}
                      rows={2}
                      value={ind.first_order_description}
                      onChange={(e) => updateIndustry(idx, { first_order_description: e.target.value })}
                      placeholder="What do direct industry companies look like?"
                    />
                  </div>
                )}

                {(ind.order === "2nd" || ind.order === "both") && (
                  <div>
                    <label style={{ ...labelStyle, marginBottom: 4 }}>2nd-order description</label>
                    <textarea
                      style={{ ...textareaStyle, minHeight: 52 }}
                      rows={2}
                      value={ind.second_order_description}
                      onChange={(e) => updateIndustry(idx, { second_order_description: e.target.value })}
                      placeholder="What companies service or supply this industry?"
                    />
                  </div>
                )}

                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7,
                    cursor: "pointer",
                    fontSize: 12,
                    color: "var(--text-secondary)",
                    marginTop: 10,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={ind.use_builtin}
                    onChange={(e) => updateIndustry(idx, { use_builtin: e.target.checked })}
                    style={{ accentColor: "var(--accent)", width: 13, height: 13 }}
                  />
                  Use built-in industry signals
                </label>
              </div>
            ))}

            <AddRowButton onClick={addIndustry} label="Add industry" />
          </div>
        )}

        {pathPreference === "company_first" && (
          <div>
            <div style={{ ...fieldGroupStyle }}>
              <label style={labelStyle}>Account list — one company per line</label>
              <textarea
                style={{ ...textareaStyle, minHeight: 140 }}
                rows={7}
                value={accountText}
                onChange={(e) => setAccountText(e.target.value)}
                placeholder={"Infosys\nTCS\nWipro\nHCL Technologies"}
              />
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 5 }}>
                {accountText.split("\n").filter(Boolean).length} companies
              </div>
            </div>
          </div>
        )}

        {pathPreference === "report_driven" && (
          <div>
            <div style={fieldGroupStyle}>
              <label style={labelStyle}>Report title</label>
              <input
                style={inputStyle}
                type="text"
                value={reportTitle}
                onChange={(e) => setReportTitle(e.target.value)}
                placeholder="e.g. Q2 2026 Fintech Landscape"
              />
            </div>
            <div style={fieldGroupStyle}>
              <label style={labelStyle}>Research brief</label>
              <textarea
                style={{ ...textareaStyle, minHeight: 120 }}
                rows={5}
                value={reportSummary}
                onChange={(e) => setReportSummary(e.target.value)}
                placeholder="Describe the report topic, scope, and what companies or trends you want to surface..."
              />
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderStep3() {
    return (
      <div>
        <div style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--text)", marginBottom: 6, letterSpacing: "-0.02em" }}>
            What do you sell?
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
            Harbinger uses your products to match signals and personalise outreach.
          </p>
        </div>

        {products.length === 0 && (
          <div
            style={{
              textAlign: "center",
              color: "var(--text-muted)",
              fontSize: 12,
              padding: "28px 0",
              border: "1px dashed var(--border)",
              borderRadius: 10,
              marginBottom: 14,
            }}
          >
            No products added yet. Add at least one.
          </div>
        )}

        {products.map((p, idx) => (
          <div
            key={idx}
            style={{
              border: "1px solid var(--border)",
              borderRadius: 10,
              padding: "16px",
              marginBottom: 12,
              background: "var(--surface-raised)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Product {idx + 1}
              </span>
              <button
                onClick={() => removeProduct(idx)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--red)", padding: 4, borderRadius: 4, display: "flex" }}
              >
                <Trash2 size={14} />
              </button>
            </div>

            <div style={{ marginBottom: 10 }}>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Product name</label>
              <input
                style={inputStyle}
                type="text"
                value={p.name}
                onChange={(e) => updateProduct(idx, { name: e.target.value })}
                placeholder="e.g. Risk Assessment Suite"
              />
            </div>

            <div style={{ marginBottom: 10 }}>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Value proposition</label>
              <textarea
                style={{ ...textareaStyle, minHeight: 64 }}
                rows={3}
                value={p.value_prop}
                onChange={(e) => updateProduct(idx, { value_prop: e.target.value })}
                placeholder="What problem does this solve and for whom?"
              />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
              <div>
                <label style={{ ...labelStyle, marginBottom: 4 }}>Target roles (comma-sep)</label>
                <input
                  style={inputStyle}
                  type="text"
                  value={productRolesText[idx] ?? ""}
                  onChange={(e) => {
                    const arr = [...productRolesText];
                    arr[idx] = e.target.value;
                    setProductRolesText(arr);
                  }}
                  placeholder="CTO, CISO, VP Eng"
                />
              </div>
              <div>
                <label style={{ ...labelStyle, marginBottom: 4 }}>Event triggers (comma-sep)</label>
                <input
                  style={inputStyle}
                  type="text"
                  value={productEventsText[idx] ?? ""}
                  onChange={(e) => {
                    const arr = [...productEventsText];
                    arr[idx] = e.target.value;
                    setProductEventsText(arr);
                  }}
                  placeholder="funding, acquisition"
                />
              </div>
            </div>

            <div>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Case studies (comma-sep)</label>
              <input
                style={inputStyle}
                type="text"
                value={productCasesText[idx] ?? ""}
                onChange={(e) => {
                  const arr = [...productCasesText];
                  arr[idx] = e.target.value;
                  setProductCasesText(arr);
                }}
                placeholder="Acme Bank 2025, Sun Pharma rollout"
              />
            </div>
          </div>
        ))}

        <AddRowButton onClick={addProduct} label="Add product" />
      </div>
    );
  }

  function renderStep4() {
    return (
      <div>
        <div style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--text)", marginBottom: 6, letterSpacing: "-0.02em" }}>
            Email & lead scoring
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
            Outreach emails are sent via Brevo from your sender identity.
          </p>
        </div>

        <div
          style={{
            padding: "14px 16px",
            background: "var(--surface-raised)",
            borderRadius: 10,
            border: "1px solid var(--border)",
            marginBottom: 24,
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
            Sender identity
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={{ ...labelStyle, marginBottom: 4 }}>From name</label>
              <input
                style={inputStyle}
                type="text"
                value={fromName}
                onChange={(e) => setFromName(e.target.value)}
                placeholder="e.g. Arjun from Acme"
              />
            </div>
            <div>
              <label style={{ ...labelStyle, marginBottom: 4 }}>From email</label>
              <input
                style={inputStyle}
                type="email"
                value={fromEmail}
                onChange={(e) => setFromEmail(e.target.value)}
                placeholder="arjun@acme.com"
              />
            </div>
          </div>
        </div>

        <div
          style={{
            padding: "14px 16px",
            background: "var(--surface-raised)",
            borderRadius: 10,
            border: "1px solid var(--border)",
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 14 }}>
            Lead quality threshold
          </div>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>Min lead score</span>
            <span
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: "var(--accent)",
                fontFamily: "var(--font-mono, monospace)",
                background: "var(--accent-light)",
                padding: "2px 10px",
                borderRadius: 6,
              }}
            >
              {minLeadScore.toFixed(1)}
            </span>
          </div>

          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.1}
            value={minLeadScore}
            onChange={(e) => setMinLeadScore(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent)", cursor: "pointer", height: 4 }}
          />

          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)", marginTop: 6 }}>
            <span>0.1 — Cast wide</span>
            <span>0.5 — Balanced</span>
            <span>0.9 — High confidence</span>
          </div>
        </div>
      </div>
    );
  }

  function renderStep5() {
    const modeConfig = PATH_MODES.find((m) => m.value === pathPreference);
    const accountCount = accountText.split("\n").filter(Boolean).length;

    return (
      <div>
        <div style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--text)", marginBottom: 6, letterSpacing: "-0.02em" }}>
            Review your profile
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
            Everything looks right? Save to activate this profile.
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Identity */}
          <ReviewCard
            title="About You"
            onEdit={() => jumpTo(1)}
            rows={[
              { label: "Name", value: userName || "—" },
              { label: "Company", value: ownCompany || "—" },
              { label: "Region", value: region },
            ]}
          />

          {/* Pipeline mode */}
          <ReviewCard
            title="Pipeline Mode"
            onEdit={() => jumpTo(2)}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div
                style={{
                  padding: "2px 10px",
                  borderRadius: 999,
                  fontSize: 11,
                  fontWeight: 600,
                  background: modeConfig?.bg,
                  color: modeConfig?.color,
                }}
              >
                {modeConfig?.label}
              </div>
            </div>
            {pathPreference === "industry_first" && (
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                {industries.length === 0
                  ? "No industries configured"
                  : industries.map((i) => i.display_name || "Unnamed").join(" · ")}
              </div>
            )}
            {pathPreference === "company_first" && (
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                {accountCount} {accountCount === 1 ? "company" : "companies"} in account list
              </div>
            )}
            {pathPreference === "report_driven" && (
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                {reportTitle || "No title set"}
              </div>
            )}
          </ReviewCard>

          {/* Products */}
          <ReviewCard
            title="Products"
            onEdit={() => jumpTo(3)}
          >
            {products.length === 0 ? (
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>No products added</span>
            ) : (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {products.map((p, i) => (
                  <span
                    key={i}
                    style={{
                      padding: "2px 9px",
                      borderRadius: 999,
                      fontSize: 11,
                      fontWeight: 500,
                      background: "var(--accent-light)",
                      color: "var(--accent)",
                      border: "1px solid rgba(176,112,48,0.2)",
                    }}
                  >
                    {p.name || `Product ${i + 1}`}
                  </span>
                ))}
              </div>
            )}
          </ReviewCard>

          {/* Email */}
          <ReviewCard
            title="Email & Scoring"
            onEdit={() => jumpTo(4)}
            rows={[
              { label: "Sender", value: fromName ? `${fromName} <${fromEmail}>` : fromEmail || "—" },
              { label: "Min score", value: minLeadScore.toFixed(1) },
            ]}
          />
        </div>
      </div>
    );
  }

  // ── Progress indicator ────────────────────────────────────────────────────

  const progressPct = ((step - 1) / (STEPS.length - 1)) * 100;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 1000,
          background: "rgba(10,9,6,0.55)",
          backdropFilter: "blur(6px)",
        }}
      />

      {/* Dialog */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 1001,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "16px",
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            width: "min(640px, 100%)",
            maxHeight: "min(88vh, 780px)",
            background: "var(--surface)",
            borderRadius: 16,
            border: "1px solid var(--border)",
            boxShadow: "0 24px 80px rgba(0,0,0,0.28), 0 4px 16px rgba(0,0,0,0.12)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            pointerEvents: "auto",
            animation: "wizard-enter 220ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
          }}
        >
          {/* ── Header ── */}
          <div
            style={{
              padding: "18px 22px 0",
              flexShrink: 0,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
              <div>
                <div
                  className="font-display"
                  style={{ fontSize: 13, fontWeight: 700, color: "var(--accent)", letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: 2 }}
                >
                  Harbinger
                </div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  {isNew ? "New profile" : `Editing: ${profile.user_name}`}
                </div>
              </div>
              <button
                onClick={onClose}
                style={{
                  background: "none",
                  border: "1px solid var(--border)",
                  cursor: "pointer",
                  color: "var(--text-muted)",
                  padding: 7,
                  borderRadius: 8,
                  display: "flex",
                  transition: "background 150ms",
                }}
              >
                <X size={14} />
              </button>
            </div>

            {/* Step indicator */}
            <div style={{ display: "flex", alignItems: "center", gap: 0, marginBottom: 20 }}>
              {STEPS.map((s, idx) => {
                const done = step > s.id;
                const active = step === s.id;
                return (
                  <div key={s.id} style={{ display: "flex", alignItems: "center", flex: idx < STEPS.length - 1 ? 1 : "none" }}>
                    {/* Circle */}
                    <button
                      onClick={() => done ? jumpTo(s.id) : undefined}
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: "50%",
                        border: `2px solid ${done ? "var(--accent)" : active ? "var(--accent)" : "var(--border)"}`,
                        background: done ? "var(--accent)" : active ? "var(--accent-light)" : "var(--surface)",
                        color: done ? "#fff" : active ? "var(--accent)" : "var(--text-muted)",
                        fontSize: 11,
                        fontWeight: 700,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        cursor: done ? "pointer" : "default",
                        flexShrink: 0,
                        transition: "all 200ms",
                      }}
                    >
                      {done ? <Check size={12} strokeWidth={3} /> : s.id}
                    </button>

                    {/* Connector line */}
                    {idx < STEPS.length - 1 && (
                      <div
                        style={{
                          flex: 1,
                          height: 2,
                          background: done ? "var(--accent)" : "var(--border)",
                          transition: "background 300ms",
                          margin: "0 4px",
                        }}
                      />
                    )}
                  </div>
                );
              })}
            </div>

            {/* Progress bar */}
            <div
              style={{
                height: 2,
                background: "var(--border)",
                borderRadius: 1,
                overflow: "hidden",
                marginBottom: 0,
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${progressPct}%`,
                  background: "var(--accent)",
                  borderRadius: 1,
                  transition: "width 250ms ease",
                }}
              />
            </div>

            {/* Step label row */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: 8,
                marginBottom: 4,
              }}
            >
              {STEPS.map((s) => (
                <div
                  key={s.id}
                  style={{
                    fontSize: 10,
                    color: step === s.id ? "var(--accent)" : "var(--text-muted)",
                    fontWeight: step === s.id ? 600 : 400,
                    transition: "color 200ms",
                    textAlign: "center",
                    flex: 1,
                  }}
                >
                  {s.title}
                </div>
              ))}
            </div>

            <div style={{ borderBottom: "1px solid var(--border)", marginTop: 12 }} />
          </div>

          {/* ── Body ── */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "24px 24px 8px",
            }}
          >
            {/* Error banner */}
            {error && (
              <div
                style={{
                  padding: "9px 12px",
                  background: "#FEF1F0",
                  color: "var(--red)",
                  borderRadius: 8,
                  fontSize: 12,
                  marginBottom: 16,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  border: "1px solid rgba(168,50,38,0.15)",
                }}
              >
                {error}
                <button
                  onClick={() => setError(null)}
                  style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontWeight: 700, fontSize: 14, padding: 0 }}
                >
                  &times;
                </button>
              </div>
            )}

            {step === 1 && renderStep1()}
            {step === 2 && renderStep2()}
            {step === 3 && renderStep3()}
            {step === 4 && renderStep4()}
            {step === 5 && renderStep5()}
          </div>

          {/* ── Footer ── */}
          <div
            style={{
              padding: "16px 24px",
              borderTop: "1px solid var(--border)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 10,
              flexShrink: 0,
              background: "var(--surface)",
            }}
          >
            {/* Back */}
            <button
              onClick={step === 1 ? onClose : goBack}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 5,
                padding: "9px 16px",
                fontSize: 13,
                fontWeight: 500,
                borderRadius: 8,
                border: "1px solid var(--border)",
                background: "var(--surface-raised)",
                color: "var(--text-secondary)",
                cursor: "pointer",
                transition: "background 150ms",
              }}
            >
              <ChevronLeft size={14} />
              {step === 1 ? "Cancel" : "Back"}
            </button>

            {/* Step counter */}
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              Step {step} of {STEPS.length}
            </span>

            {/* Next / Save */}
            {step < 5 ? (
              <button
                onClick={goNext}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                  padding: "9px 20px",
                  fontSize: 13,
                  fontWeight: 600,
                  borderRadius: 8,
                  border: "none",
                  background: "var(--accent)",
                  color: "#fff",
                  cursor: "pointer",
                  transition: "opacity 150ms",
                }}
              >
                Continue
                <ChevronRight size={14} />
              </button>
            ) : (
              <button
                onClick={handleSave}
                disabled={saving}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "9px 22px",
                  fontSize: 13,
                  fontWeight: 600,
                  borderRadius: 8,
                  border: "none",
                  background: saving ? "var(--surface-raised)" : "var(--accent)",
                  color: saving ? "var(--text-muted)" : "#fff",
                  cursor: saving ? "not-allowed" : "pointer",
                  transition: "background 150ms",
                }}
              >
                {saving ? (
                  "Saving…"
                ) : (
                  <>
                    <Check size={14} strokeWidth={3} />
                    {isNew ? "Create Profile" : "Save Changes"}
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes wizard-enter {
          from { opacity: 0; transform: scale(0.96) translateY(8px); }
          to   { opacity: 1; transform: scale(1) translateY(0); }
        }
      `}</style>
    </>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────

function AddRowButton({ onClick, label }: { onClick: () => void; label: string }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: "100%",
        padding: "10px",
        fontSize: 12,
        fontWeight: 500,
        borderRadius: 9,
        border: `2px dashed ${hovered ? "var(--accent)" : "var(--border)"}`,
        background: "transparent",
        color: hovered ? "var(--accent)" : "var(--text-muted)",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        transition: "border-color 150ms, color 150ms",
      }}
    >
      <Plus size={13} />
      {label}
    </button>
  );
}

interface ReviewCardProps {
  title: string;
  onEdit: () => void;
  rows?: { label: string; value: string }[];
  children?: React.ReactNode;
}

function ReviewCard({ title, onEdit, rows, children }: ReviewCardProps) {
  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: 10,
        overflow: "hidden",
        background: "var(--surface)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface-raised)",
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {title}
        </span>
        <button
          onClick={onEdit}
          style={{
            fontSize: 11,
            color: "var(--accent)",
            background: "none",
            border: "none",
            cursor: "pointer",
            fontWeight: 600,
            padding: 0,
          }}
        >
          Edit
        </button>
      </div>
      <div style={{ padding: "12px 14px" }}>
        {rows && rows.map((row) => (
          <div
            key={row.label}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-start",
              marginBottom: 6,
              gap: 12,
            }}
          >
            <span style={{ fontSize: 11, color: "var(--text-muted)", flexShrink: 0 }}>{row.label}</span>
            <span style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, textAlign: "right" }}>{row.value}</span>
          </div>
        ))}
        {children}
      </div>
    </div>
  );
}
