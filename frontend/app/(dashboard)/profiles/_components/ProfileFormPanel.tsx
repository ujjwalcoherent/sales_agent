"use client";

import { useState } from "react";
import { X, Plus, Trash2 } from "lucide-react";
import { createProfile, updateProfile } from "@/lib/api";
import { REGION_OPTIONS } from "@/lib/countries";
import type {
  UserProfile,
  CreateProfileRequest,
  IndustryTarget,
  ProductEntry,
  ContactHierarchyEntry,
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

function emptyContact(): ContactHierarchyEntry {
  return {
    event_type: "",
    company_size: "any",
    role_priority: [],
  };
}

// ── Shared form field styles ───────────────────────────────────────────────

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
  padding: "8px 11px",
  borderRadius: 8,
  border: "1px solid var(--border)",
  background: "var(--surface-raised)",
  fontSize: 13,
  color: "var(--text)",
  outline: "none",
  boxSizing: "border-box",
  fontFamily: "inherit",
};

const textareaStyle: React.CSSProperties = {
  ...inputStyle,
  resize: "vertical",
  lineHeight: 1.6,
};

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: "pointer",
};

const fieldGroupStyle: React.CSSProperties = {
  marginBottom: 18,
};

// ── Tab definitions ────────────────────────────────────────────────────────

const TABS = ["Identity", "Industries", "Products", "Account List", "Contact Rules", "Email Config"] as const;
type Tab = (typeof TABS)[number];

// ── Props ──────────────────────────────────────────────────────────────────

interface Props {
  profile: UserProfile | null;
  onClose: () => void;
  onSaved: (profile: UserProfile) => void;
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function ProfileFormPanel({ profile, onClose, onSaved }: Props) {
  const isNew = profile === null;
  const [activeTab, setActiveTab] = useState<Tab>("Identity");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Identity fields ──
  const [userName, setUserName] = useState(profile?.user_name ?? "");
  const [ownCompany, setOwnCompany] = useState(profile?.own_company ?? "");
  const [region, setRegion] = useState(profile?.region ?? "global");
  const [pathPreference, setPathPreference] = useState<UserProfile["path_preference"]>(
    profile?.path_preference ?? "auto"
  );
  const [minLeadScore, setMinLeadScore] = useState(profile?.min_lead_score ?? 0.5);

  // ── Industries ──
  const [industries, setIndustries] = useState<IndustryTarget[]>(
    profile?.target_industries.length ? profile.target_industries : []
  );

  // ── Products ──
  const [products, setProducts] = useState<ProductEntry[]>(
    profile?.own_products.length ? profile.own_products : []
  );
  // Per-product: store target_roles, case_studies, and relevant_event_types as comma strings for editing
  const [productRolesText, setProductRolesText] = useState<string[]>(
    (profile?.own_products ?? []).map((p) => p.target_roles.join(", "))
  );
  const [productCaseStudiesText, setProductCaseStudiesText] = useState<string[]>(
    (profile?.own_products ?? []).map((p) => p.case_studies.join(", "))
  );
  const [productEventTypesText, setProductEventTypesText] = useState<string[]>(
    (profile?.own_products ?? []).map((p) => p.relevant_event_types.join(", "))
  );

  // ── Account list ──
  const [accountText, setAccountText] = useState(
    (profile?.account_list ?? []).join("\n")
  );

  // ── Contact rules ──
  const [contactRules, setContactRules] = useState<ContactHierarchyEntry[]>(
    profile?.contact_hierarchy ?? []
  );
  const [rulePriorityText, setRulePriorityText] = useState<string[]>(
    (profile?.contact_hierarchy ?? []).map((r) => r.role_priority.join(", "))
  );

  // ── Email config ──
  const [fromName, setFromName] = useState(profile?.email_config?.from_name ?? "");
  const [fromEmail, setFromEmail] = useState(profile?.email_config?.from_email ?? "");

  // ── Product helpers ──────────────────────────────────────────────────────

  function updateProduct(idx: number, patch: Partial<ProductEntry>) {
    setProducts((prev) => prev.map((p, i) => (i === idx ? { ...p, ...patch } : p)));
  }

  function addProduct() {
    setProducts((prev) => [...prev, emptyProduct()]);
    setProductRolesText((prev) => [...prev, ""]);
    setProductCaseStudiesText((prev) => [...prev, ""]);
    setProductEventTypesText((prev) => [...prev, ""]);
  }

  function removeProduct(idx: number) {
    setProducts((prev) => prev.filter((_, i) => i !== idx));
    setProductRolesText((prev) => prev.filter((_, i) => i !== idx));
    setProductCaseStudiesText((prev) => prev.filter((_, i) => i !== idx));
    setProductEventTypesText((prev) => prev.filter((_, i) => i !== idx));
  }

  // ── Industry helpers ─────────────────────────────────────────────────────

  function updateIndustry(idx: number, patch: Partial<IndustryTarget>) {
    setIndustries((prev) => prev.map((ind, i) => (i === idx ? { ...ind, ...patch } : ind)));
  }

  function addIndustry() {
    setIndustries((prev) => [...prev, emptyIndustry()]);
  }

  function removeIndustry(idx: number) {
    setIndustries((prev) => prev.filter((_, i) => i !== idx));
  }

  // ── Contact rule helpers ──────────────────────────────────────────────────

  function updateRule(idx: number, patch: Partial<ContactHierarchyEntry>) {
    setContactRules((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  }

  function addRule() {
    setContactRules((prev) => [...prev, emptyContact()]);
    setRulePriorityText((prev) => [...prev, ""]);
  }

  function removeRule(idx: number) {
    setContactRules((prev) => prev.filter((_, i) => i !== idx));
    setRulePriorityText((prev) => prev.filter((_, i) => i !== idx));
  }

  // ── Save ──────────────────────────────────────────────────────────────────

  async function handleSave() {
    if (!userName.trim()) {
      setError("Name is required");
      setActiveTab("Identity");
      return;
    }

    // Merge comma-split fields back into product arrays
    const mergedProducts: ProductEntry[] = products.map((p, i) => ({
      ...p,
      case_studies: (productCaseStudiesText[i] ?? "").split(",").map((s) => s.trim()).filter(Boolean),
      target_roles: (productRolesText[i] ?? "").split(",").map((s) => s.trim()).filter(Boolean),
      relevant_event_types: (productEventTypesText[i] ?? "").split(",").map((s) => s.trim()).filter(Boolean),
    }));

    // Merge comma-split fields back into contact rules
    const mergedRules: ContactHierarchyEntry[] = contactRules.map((r, i) => ({
      ...r,
      role_priority: (rulePriorityText[i] ?? "").split(",").map((s) => s.trim()).filter(Boolean),
    }));

    const payload: CreateProfileRequest = {
      user_name: userName.trim(),
      own_company: ownCompany.trim(),
      region,
      path_preference: pathPreference,
      min_lead_score: minLeadScore,
      target_industries: industries,
      own_products: mergedProducts,
      account_list: accountText.split("\n").filter(Boolean),
      report_title: profile?.report_title ?? "",
      report_summary: profile?.report_summary ?? "",
      contact_hierarchy: mergedRules,
      email_config: { from_name: fromName.trim(), from_email: fromEmail.trim() },
    };

    setSaving(true);
    setError(null);
    try {
      let saved: UserProfile;
      if (isNew) {
        saved = await createProfile(payload);
      } else {
        saved = await updateProfile(profile.profile_id, payload);
      }
      onSaved(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  // ── Tab content renderers ─────────────────────────────────────────────────

  function renderIdentity() {
    return (
      <div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>Name *</label>
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
          <label style={labelStyle}>Own Company</label>
          <input
            style={inputStyle}
            type="text"
            value={ownCompany}
            onChange={(e) => setOwnCompany(e.target.value)}
            placeholder="e.g. Acme Consulting"
          />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 18 }}>
          <div>
            <label style={labelStyle}>Region / Country</label>
            <select style={selectStyle} value={region} onChange={(e) => setRegion(e.target.value)}>
              {REGION_OPTIONS.map((r) => (
                <option key={r.code} value={r.code}>{r.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={labelStyle}>Path Preference</label>
            <select
              style={selectStyle}
              value={pathPreference}
              onChange={(e) => setPathPreference(e.target.value as UserProfile["path_preference"])}
            >
              <option value="auto">Auto</option>
              <option value="industry_first">Industry-First</option>
              <option value="company_first">Company-First</option>
              <option value="report_driven">Report-Driven</option>
            </select>
          </div>
        </div>
      </div>
    );
  }

  function renderIndustries() {
    return (
      <div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 16, lineHeight: 1.6, padding: "10px 12px", background: "var(--surface-raised)", borderRadius: 8, border: "1px solid var(--border)" }}>
          <strong style={{ color: "var(--text-secondary)" }}>1st Order</strong> — companies directly in the industry.{" "}
          <strong style={{ color: "var(--text-secondary)" }}>2nd Order</strong> — companies that service or supply to that industry.
        </div>
        {industries.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, padding: "24px 0" }}>
            No industries added yet
          </div>
        )}
        {industries.map((ind, idx) => (
          <div
            key={ind.industry_id}
            style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "14px", marginBottom: 12, background: "var(--surface-raised)" }}
          >
            <div style={{ display: "flex", gap: 10, alignItems: "flex-start", marginBottom: 10 }}>
              <div style={{ flex: 1 }}>
                <label style={{ ...labelStyle, marginBottom: 4 }}>Industry Name</label>
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
                  style={selectStyle}
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
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--red)", padding: "4px", marginTop: 20, borderRadius: 4, display: "flex" }}
                title="Remove"
              >
                <Trash2 size={14} />
              </button>
            </div>
            {(ind.order === "1st" || ind.order === "both") && (
              <div style={{ marginBottom: ind.order === "both" ? 10 : 0 }}>
                <label style={{ ...labelStyle, marginBottom: 4 }}>1st Order Description</label>
                <textarea
                  style={{ ...textareaStyle, minHeight: 56 }}
                  rows={2}
                  value={ind.first_order_description}
                  onChange={(e) => updateIndustry(idx, { first_order_description: e.target.value })}
                  placeholder="Describe what direct industry companies look like..."
                />
              </div>
            )}
            {(ind.order === "2nd" || ind.order === "both") && (
              <div>
                <label style={{ ...labelStyle, marginBottom: 4 }}>2nd Order Description</label>
                <textarea
                  style={{ ...textareaStyle, minHeight: 56 }}
                  rows={2}
                  value={ind.second_order_description}
                  onChange={(e) => updateIndustry(idx, { second_order_description: e.target.value })}
                  placeholder="Describe companies that service or supply this industry..."
                />
              </div>
            )}
            <label style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer", fontSize: 12, color: "var(--text-secondary)", marginTop: 10 }}>
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
        <button
          onClick={addIndustry}
          style={{
            width: "100%", padding: "10px", fontSize: 12, fontWeight: 500,
            borderRadius: 9, border: "2px dashed var(--border)", background: "transparent",
            color: "var(--text-muted)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            transition: "border-color 150ms, color 150ms",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--accent)"; (e.currentTarget as HTMLButtonElement).style.color = "var(--accent)"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLButtonElement).style.color = "var(--text-muted)"; }}
        >
          <Plus size={13} /> Add Industry
        </button>
      </div>
    );
  }

  function renderProducts() {
    return (
      <div>
        {products.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, padding: "24px 0" }}>
            No products added yet
          </div>
        )}
        {products.map((p, idx) => (
          <div
            key={idx}
            style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "14px", marginBottom: 12, background: "var(--surface-raised)" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>
                Product / Report {idx + 1}
              </span>
              <button
                onClick={() => removeProduct(idx)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--red)", padding: 4, borderRadius: 4, display: "flex" }}
              >
                <Trash2 size={14} />
              </button>
            </div>
            <div style={{ marginBottom: 10 }}>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Product Name</label>
              <input
                style={inputStyle}
                type="text"
                value={p.name}
                onChange={(e) => updateProduct(idx, { name: e.target.value })}
                placeholder="e.g. Risk Assessment Suite"
              />
            </div>
            <div style={{ marginBottom: 10 }}>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Value Proposition</label>
              <textarea
                style={{ ...textareaStyle, minHeight: 64 }}
                rows={3}
                value={p.value_prop}
                onChange={(e) => updateProduct(idx, { value_prop: e.target.value })}
                placeholder="What problem does this solve and for whom?"
              />
            </div>
            <div style={{ marginBottom: 10 }}>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Case Studies (comma-separated)</label>
              <input
                style={inputStyle}
                type="text"
                value={productCaseStudiesText[idx] ?? ""}
                onChange={(e) => {
                  const newArr = [...productCaseStudiesText];
                  newArr[idx] = e.target.value;
                  setProductCaseStudiesText(newArr);
                }}
                placeholder="e.g. Manipal Hospitals 2024, Sun Pharma rollout"
              />
            </div>
            <div style={{ marginBottom: 10 }}>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Target Roles (comma-separated)</label>
              <input
                style={inputStyle}
                type="text"
                value={productRolesText[idx] ?? ""}
                onChange={(e) => {
                  const newArr = [...productRolesText];
                  newArr[idx] = e.target.value;
                  setProductRolesText(newArr);
                }}
                placeholder="e.g. CTO, VP Engineering, CISO"
              />
            </div>
            <div>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Relevant Event Types (comma-separated)</label>
              <input
                style={inputStyle}
                type="text"
                value={productEventTypesText[idx] ?? ""}
                onChange={(e) => {
                  const newArr = [...productEventTypesText];
                  newArr[idx] = e.target.value;
                  setProductEventTypesText(newArr);
                }}
                placeholder="e.g. funding, acquisition, regulatory_action"
              />
            </div>
          </div>
        ))}
        <button
          onClick={addProduct}
          style={{
            width: "100%", padding: "10px", fontSize: 12, fontWeight: 500,
            borderRadius: 9, border: "2px dashed var(--border)", background: "transparent",
            color: "var(--text-muted)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            transition: "border-color 150ms, color 150ms",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--accent)"; (e.currentTarget as HTMLButtonElement).style.color = "var(--accent)"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLButtonElement).style.color = "var(--text-muted)"; }}
        >
          <Plus size={13} /> Add Product / Report
        </button>
      </div>
    );
  }

  function renderAccountList() {
    const count = accountText.split("\n").filter(Boolean).length;
    return (
      <div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12, lineHeight: 1.6 }}>
          Companies on this list will be tracked directly (Company-First path). One company per line.
        </div>
        <textarea
          style={{ ...textareaStyle, minHeight: 260 }}
          rows={12}
          value={accountText}
          onChange={(e) => setAccountText(e.target.value)}
          placeholder={"Infosys\nTCS\nWipro\nHCL Technologies"}
        />
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 6 }}>
          {count} {count === 1 ? "company" : "companies"} in account list
        </div>
      </div>
    );
  }

  function renderContactRules() {
    return (
      <div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 16, lineHeight: 1.6 }}>
          Map event types to preferred contact roles. When an event matches, the pipeline will target roles in priority order.
        </div>
        {contactRules.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, padding: "24px 0" }}>
            No rules added yet
          </div>
        )}
        {contactRules.map((rule, idx) => (
          <div
            key={idx}
            style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "14px", marginBottom: 12, background: "var(--surface-raised)" }}
          >
            <div style={{ display: "flex", gap: 10, alignItems: "flex-start", marginBottom: 10 }}>
              <div style={{ flex: 1 }}>
                <label style={{ ...labelStyle, marginBottom: 4 }}>Event Type</label>
                <input
                  style={inputStyle}
                  type="text"
                  value={rule.event_type}
                  onChange={(e) => updateRule(idx, { event_type: e.target.value })}
                  placeholder="e.g. funding, acquisition, IPO"
                />
              </div>
              <div style={{ width: 130 }}>
                <label style={{ ...labelStyle, marginBottom: 4 }}>Company Size</label>
                <select
                  style={selectStyle}
                  value={rule.company_size}
                  onChange={(e) => updateRule(idx, { company_size: e.target.value })}
                >
                  <option value="any">Any</option>
                  <option value="startup">Startup</option>
                  <option value="smb">SMB</option>
                  <option value="mid">Mid-market</option>
                  <option value="enterprise">Enterprise</option>
                </select>
              </div>
              <button
                onClick={() => removeRule(idx)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--red)", padding: "4px", marginTop: 20, borderRadius: 4, display: "flex" }}
              >
                <Trash2 size={14} />
              </button>
            </div>
            <div>
              <label style={{ ...labelStyle, marginBottom: 4 }}>Role Priority (comma-separated)</label>
              <input
                style={inputStyle}
                type="text"
                value={rulePriorityText[idx] ?? ""}
                onChange={(e) => {
                  const newArr = [...rulePriorityText];
                  newArr[idx] = e.target.value;
                  setRulePriorityText(newArr);
                }}
                placeholder="e.g. CFO, VP Finance, Controller"
              />
            </div>
          </div>
        ))}
        <button
          onClick={addRule}
          style={{
            width: "100%", padding: "10px", fontSize: 12, fontWeight: 500,
            borderRadius: 9, border: "2px dashed var(--border)", background: "transparent",
            color: "var(--text-muted)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            transition: "border-color 150ms, color 150ms",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--accent)"; (e.currentTarget as HTMLButtonElement).style.color = "var(--accent)"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLButtonElement).style.color = "var(--text-muted)"; }}
        >
          <Plus size={13} /> Add Rule
        </button>
      </div>
    );
  }

  function renderEmailConfig() {
    return (
      <div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 16, lineHeight: 1.6 }}>
          Outreach emails will be sent from this sender identity via Brevo.
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>From Name</label>
          <input
            style={inputStyle}
            type="text"
            value={fromName}
            onChange={(e) => setFromName(e.target.value)}
            placeholder="e.g. Arjun from Acme"
          />
        </div>
        <div style={fieldGroupStyle}>
          <label style={labelStyle}>From Email</label>
          <input
            style={inputStyle}
            type="email"
            value={fromEmail}
            onChange={(e) => setFromEmail(e.target.value)}
            placeholder="e.g. arjun@acmeconsulting.com"
          />
        </div>
      </div>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 900,
          background: "rgba(0,0,0,0.45)",
          backdropFilter: "blur(4px)",
        }}
      />

      {/* Panel */}
      <div
        style={{
          position: "fixed",
          top: 0,
          right: 0,
          bottom: 0,
          zIndex: 901,
          width: "min(560px, 100vw)",
          background: "var(--surface)",
          borderLeft: "1px solid var(--border)",
          boxShadow: "-8px 0 32px rgba(0,0,0,0.18)",
          display: "flex",
          flexDirection: "column",
          animation: "slide-in-right 280ms ease forwards",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "18px 22px 14px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexShrink: 0,
          }}
        >
          <div>
            <div className="font-display" style={{ fontSize: 17, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.01em" }}>
              {isNew ? "New Profile" : "Edit Profile"}
            </div>
            {!isNew && (
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                {profile.user_name}
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 6, borderRadius: 6, display: "flex" }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: 0,
            borderBottom: "1px solid var(--border)",
            flexShrink: 0,
            overflowX: "auto",
            padding: "0 22px",
          }}
        >
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "10px 14px",
                fontSize: 12,
                fontWeight: activeTab === tab ? 600 : 500,
                color: activeTab === tab ? "var(--accent)" : "var(--text-muted)",
                background: "none",
                border: "none",
                borderBottom: `2px solid ${activeTab === tab ? "var(--accent)" : "transparent"}`,
                cursor: "pointer",
                whiteSpace: "nowrap",
                transition: "color 150ms, border-color 150ms",
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab body */}
        <div style={{ flex: 1, overflowY: "auto", padding: "20px 22px" }}>
          {/* Error */}
          {error && (
            <div
              style={{
                padding: "9px 12px",
                background: "var(--red-light)",
                color: "var(--red)",
                borderRadius: 8,
                fontSize: 12,
                marginBottom: 16,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              {error}
              <button
                onClick={() => setError(null)}
                style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontWeight: 700, fontSize: 14 }}
              >
                &times;
              </button>
            </div>
          )}

          {activeTab === "Identity" && renderIdentity()}
          {activeTab === "Industries" && renderIndustries()}
          {activeTab === "Products" && renderProducts()}
          {activeTab === "Account List" && renderAccountList()}
          {activeTab === "Contact Rules" && renderContactRules()}
          {activeTab === "Email Config" && renderEmailConfig()}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: "14px 22px",
            borderTop: "1px solid var(--border)",
            display: "flex",
            gap: 10,
            flexShrink: 0,
            background: "var(--surface)",
          }}
        >
          <button
            onClick={onClose}
            style={{
              flex: 1,
              padding: "9px 16px",
              fontSize: 13,
              fontWeight: 500,
              borderRadius: 8,
              border: "1px solid var(--border)",
              background: "var(--surface-raised)",
              color: "var(--text-secondary)",
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            style={{
              flex: 2,
              padding: "9px 20px",
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
            {saving ? "Saving…" : isNew ? "Save Profile" : "Update Profile"}
          </button>
        </div>
      </div>
    </>
  );
}
