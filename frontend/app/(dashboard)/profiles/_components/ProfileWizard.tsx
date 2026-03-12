"use client";

import { useState, useCallback } from "react";
import { X, Plus, Trash2, Check, ChevronRight, ChevronLeft, ChevronDown, Zap, Building2, BarChart3, Cpu } from "lucide-react";
import { createProfile, updateProfile } from "@/lib/api";
import { REGION_OPTIONS, regionLabel } from "@/lib/countries";
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
  { id: 3, title: "Products", desc: "What you sell" },
  { id: 4, title: "Outreach", desc: "Email sender config" },
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
  const [region, setRegion] = useState(profile?.region ?? "global");

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

  // ── Step 3 accordion ──
  const [expandedProductIdx, setExpandedProductIdx] = useState<number | null>(null);

  // ── Step 4: Outreach ──
  const [fromName, setFromName] = useState(profile?.email_config?.from_name ?? "");
  const [fromEmail, setFromEmail] = useState(profile?.email_config?.from_email ?? "");

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
      min_lead_score: 0.5,
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
          <label style={labelStyle}>Region / Country</label>
          <select
            style={{ ...inputStyle, cursor: "pointer" }}
            value={region}
            onChange={(e) => setRegion(e.target.value)}
          >
            {REGION_OPTIONS.map((r) => (
              <option key={r.code} value={r.code}>{r.name}</option>
            ))}
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
            Your products
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
            The pipeline matches market signals to your products and uses them to personalise outreach emails.
          </p>
        </div>

        {products.length === 0 && (
          <div
            style={{
              textAlign: "center",
              padding: "36px 24px",
              border: "1px dashed var(--border)",
              borderRadius: 10,
              marginBottom: 14,
              background: "var(--surface-raised)",
            }}
          >
            <div style={{ fontSize: 22, marginBottom: 8 }}>+</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 4 }}>
              No products yet
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
              Add a product so Harbinger can connect signals to what you sell.
            </div>
          </div>
        )}

        {products.map((p, idx) => (
          <ProductCard
            key={idx}
            idx={idx}
            product={p}
            rolesText={productRolesText[idx] ?? ""}
            eventsText={productEventsText[idx] ?? ""}
            casesText={productCasesText[idx] ?? ""}
            isExpanded={expandedProductIdx === idx}
            onToggle={() => setExpandedProductIdx(expandedProductIdx === idx ? null : idx)}
            onUpdate={(patch) => updateProduct(idx, patch)}
            onUpdateRoles={(v) => { const a = [...productRolesText]; a[idx] = v; setProductRolesText(a); }}
            onUpdateEvents={(v) => { const a = [...productEventsText]; a[idx] = v; setProductEventsText(a); }}
            onUpdateCases={(v) => { const a = [...productCasesText]; a[idx] = v; setProductCasesText(a); }}
            onRemove={() => removeProduct(idx)}
          />
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
            Outreach
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
            Configure your sender identity for outreach emails via Brevo.
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
              { label: "Company", value: ownCompany || "Not set" },
              { label: "Region", value: regionLabel(region) },
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
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>No products added — optional but improves signal matching</span>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {products.map((p, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                    <span style={{
                      padding: "2px 9px", borderRadius: 999, fontSize: 11, fontWeight: 600,
                      background: "var(--accent-light)", color: "var(--accent)", border: "1px solid rgba(176,112,48,0.2)", flexShrink: 0,
                    }}>
                      {p.name || `Product ${i + 1}`}
                    </span>
                    {p.value_prop && (
                      <span style={{ fontSize: 11, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {p.value_prop}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </ReviewCard>

          {/* Outreach */}
          <ReviewCard
            title="Outreach"
            onEdit={() => jumpTo(4)}
            rows={[
              { label: "Sender", value: fromName ? `${fromName} <${fromEmail}>` : fromEmail || "Not configured" },
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

            {/* Step indicator — unified grid so circles and labels stay aligned */}
            <div style={{ position: "relative", marginBottom: 16 }}>
              {/* Connector track behind circles */}
              <div style={{
                position: "absolute",
                top: 14,
                left: "calc(10% + 2px)",
                right: "calc(10% + 2px)",
                height: 2,
                background: "var(--border)",
                zIndex: 0,
              }} />
              {/* Filled portion of connector */}
              <div style={{
                position: "absolute",
                top: 14,
                left: "calc(10% + 2px)",
                width: `calc(${progressPct * 0.8}% - 4px)`,
                height: 2,
                background: "var(--accent)",
                transition: "width 300ms ease",
                zIndex: 0,
              }} />

              {/* Columns: circle + label */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)" }}>
                {STEPS.map((s) => {
                  const done = step > s.id;
                  const active = step === s.id;
                  return (
                    <div
                      key={s.id}
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 8,
                        position: "relative",
                        zIndex: 1,
                      }}
                    >
                      <button
                        onClick={() => done ? jumpTo(s.id) : undefined}
                        style={{
                          width: 28,
                          height: 28,
                          borderRadius: "50%",
                          border: `2px solid ${done || active ? "var(--accent)" : "var(--border)"}`,
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
                      <span style={{
                        fontSize: 10,
                        fontWeight: active ? 600 : 400,
                        color: active ? "var(--accent)" : "var(--text-muted)",
                        textAlign: "center",
                        lineHeight: 1.3,
                        transition: "color 200ms",
                        whiteSpace: "nowrap",
                      }}>
                        {s.title}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ borderBottom: "1px solid var(--border)", marginBottom: 4 }} />
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

// ── Product card with collapsible targeting ──────────────────────────────────

function ProductCard({
  idx, product, rolesText, eventsText, casesText,
  isExpanded, onToggle,
  onUpdate, onUpdateRoles, onUpdateEvents, onUpdateCases, onRemove,
}: {
  idx: number;
  product: ProductEntry;
  rolesText: string;
  eventsText: string;
  casesText: string;
  isExpanded: boolean;
  onToggle: () => void;
  onUpdate: (patch: Partial<ProductEntry>) => void;
  onUpdateRoles: (v: string) => void;
  onUpdateEvents: (v: string) => void;
  onUpdateCases: (v: string) => void;
  onRemove: () => void;
}) {
  const hasTargeting = !!(rolesText || eventsText || casesText);
  const hasValue = !!product.value_prop;
  const fieldCount = [rolesText, eventsText, casesText].filter(Boolean).length;

  // ── Collapsed: single compact row ──
  if (!isExpanded) {
    return (
      <div
        onClick={onToggle}
        style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 14px",
          border: "1px solid var(--border)", borderRadius: 8,
          background: "var(--surface-raised)", cursor: "pointer",
          marginBottom: 6, transition: "border-color 150ms",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; }}
      >
        <div style={{
          width: 22, height: 22, borderRadius: 5,
          background: "var(--accent-light)", color: "var(--accent)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 10, fontWeight: 700, flexShrink: 0,
        }}>
          {idx + 1}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 13, fontWeight: 600, color: "var(--text)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {product.name || "Untitled product"}
          </div>
          {hasValue && (
            <div style={{
              fontSize: 11, color: "var(--text-muted)", marginTop: 1,
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>
              {product.value_prop}
            </div>
          )}
        </div>
        {/* Status pills */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
          {hasTargeting && (
            <span style={{
              fontSize: 9, fontWeight: 600, color: "var(--green)",
              background: "var(--green-light)", padding: "2px 7px",
              borderRadius: 999, letterSpacing: "0.02em",
            }}>
              {fieldCount}/3
            </span>
          )}
          <ChevronDown size={14} style={{ color: "var(--text-muted)" }} />
        </div>
      </div>
    );
  }

  // ── Expanded: full edit form ──
  return (
    <div style={{
      border: "1.5px solid var(--accent)",
      borderRadius: 10,
      padding: "14px 16px",
      marginBottom: 6,
      background: "var(--surface-raised)",
    }}>
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <div style={{
          width: 22, height: 22, borderRadius: 5,
          background: "var(--accent)", color: "#fff",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 10, fontWeight: 700, flexShrink: 0,
        }}>
          {idx + 1}
        </div>
        <input
          style={{
            flex: 1, padding: "5px 8px", borderRadius: 6,
            border: "1px solid var(--border)", background: "var(--bg)",
            fontWeight: 600, fontSize: 13, color: "var(--text)",
            fontFamily: "inherit", outline: "none",
          }}
          type="text"
          value={product.name}
          onChange={(e) => onUpdate({ name: e.target.value })}
          placeholder="Product name"
        />
        <button
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          title="Remove"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--text-muted)", padding: 4, display: "flex",
            opacity: 0.5, transition: "opacity 150ms, color 150ms",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.opacity = "1"; e.currentTarget.style.color = "var(--red)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = "0.5"; e.currentTarget.style.color = "var(--text-muted)"; }}
        >
          <Trash2 size={13} />
        </button>
        <button
          onClick={onToggle}
          title="Collapse"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--text-muted)", padding: 4, display: "flex",
          }}
        >
          <ChevronDown size={14} style={{ transform: "rotate(180deg)" }} />
        </button>
      </div>

      {/* Fields — compact 2-column grid for targeting */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div>
          <label style={{ fontSize: 10, fontWeight: 600, color: "var(--text-muted)", display: "block", marginBottom: 3 }}>
            Value proposition
          </label>
          <textarea
            style={{
              width: "100%", padding: "6px 10px", borderRadius: 7,
              border: "1px solid var(--border)", background: "var(--bg)",
              fontSize: 12, color: "var(--text)", fontFamily: "inherit",
              lineHeight: 1.5, resize: "vertical", minHeight: 36, outline: "none",
            }}
            rows={2}
            value={product.value_prop}
            onChange={(e) => onUpdate({ value_prop: e.target.value })}
            placeholder="e.g. Cuts compliance audit prep from 3 weeks to 2 days"
          />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <div>
            <label style={{ fontSize: 10, fontWeight: 600, color: "var(--text-muted)", display: "block", marginBottom: 3 }}>
              Who buys this?
            </label>
            <input
              style={{
                width: "100%", padding: "6px 10px", borderRadius: 7,
                border: "1px solid var(--border)", background: "var(--bg)",
                fontSize: 11, color: "var(--text)", fontFamily: "inherit", outline: "none",
              }}
              type="text"
              value={rolesText}
              onChange={(e) => onUpdateRoles(e.target.value)}
              placeholder="CTO, VP Engineering"
            />
          </div>
          <div>
            <label style={{ fontSize: 10, fontWeight: 600, color: "var(--text-muted)", display: "block", marginBottom: 3 }}>
              Buying signals
            </label>
            <input
              style={{
                width: "100%", padding: "6px 10px", borderRadius: 7,
                border: "1px solid var(--border)", background: "var(--bg)",
                fontSize: 11, color: "var(--text)", fontFamily: "inherit", outline: "none",
              }}
              type="text"
              value={eventsText}
              onChange={(e) => onUpdateEvents(e.target.value)}
              placeholder="funding, expansion"
            />
          </div>
        </div>
        <div>
          <label style={{ fontSize: 10, fontWeight: 600, color: "var(--text-muted)", display: "block", marginBottom: 3 }}>
            Past wins
          </label>
          <input
            style={{
              width: "100%", padding: "6px 10px", borderRadius: 7,
              border: "1px solid var(--border)", background: "var(--bg)",
              fontSize: 11, color: "var(--text)", fontFamily: "inherit", outline: "none",
            }}
            type="text"
            value={casesText}
            onChange={(e) => onUpdateCases(e.target.value)}
            placeholder="Acme Bank Q4 2025, Sun Pharma rollout"
          />
        </div>
      </div>
    </div>
  );
}

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
