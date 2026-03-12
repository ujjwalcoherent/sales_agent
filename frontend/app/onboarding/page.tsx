"use client";

import { useState, useEffect, useCallback, KeyboardEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import {
  Zap, Check, ChevronLeft, ChevronRight, Plus, ArrowRight,
  BarChart3, Building2, Cpu, X,
} from "lucide-react";
import { listProfiles, createProfile, updateProfile, getProfile, api } from "@/lib/api";
import { REGION_OPTIONS, regionLabel } from "@/lib/countries";
import type {
  UserProfile, CreateProfileRequest, IndustryTarget, ProductEntry,
} from "@/lib/types";
import { APP_NAME } from "@/lib/config";

const STORAGE_KEY = "harbinger_active_profile_id";

// ── Step config ───────────────────────────────────────────────────────────────

const STEPS = [
  { id: 1, title: "You",            desc: "Identity & region" },
  { id: 2, title: "Products",        desc: "What you sell" },
  { id: 3, title: "Your buyers",    desc: "Industries & targeting" },
  { id: 4, title: "How to find",    desc: "Pipeline mode" },
  { id: 5, title: "Configure",      desc: "Mode-specific setup" },
  { id: 6, title: "Outreach",       desc: "Sender identity" },
] as const;

type StepId = 1 | 2 | 3 | 4 | 5 | 6;

// ── PATH_MODES ────────────────────────────────────────────────────────────────

const PATH_MODES = [
  {
    value: "auto" as const,
    label: "Auto",
    subtitle: "Let Harbinger decide the best path on each run",
    icon: <Zap size={18} />,
    color: "var(--accent)",
    bg: "var(--accent-light)",
    recommended: true,
  },
  {
    value: "industry_first" as const,
    label: "Industry-First",
    subtitle: "Track trends across your target industries",
    icon: <BarChart3 size={18} />,
    color: "var(--green)",
    bg: "var(--green-light)",
    recommended: false,
  },
  {
    value: "company_first" as const,
    label: "Company-First",
    subtitle: "Monitor a named account list directly",
    icon: <Building2 size={18} />,
    color: "var(--blue)",
    bg: "#EBF2F9",
    recommended: false,
  },
  {
    value: "report_driven" as const,
    label: "Report-Driven",
    subtitle: "Build intelligence from a research brief",
    icon: <Cpu size={18} />,
    color: "#6B3FA0",
    bg: "#EDE8F5",
    recommended: false,
  },
];

const SIZE_OPTIONS = [
  { value: "startup",    label: "Startup" },
  { value: "smb",        label: "SMB" },
  { value: "mid_market", label: "Mid-market" },
  { value: "enterprise", label: "Enterprise" },
  { value: "any",        label: "Any" },
];

// ── Shared styles ─────────────────────────────────────────────────────────────

const inp: React.CSSProperties = {
  width: "100%", padding: "10px 12px", borderRadius: 8,
  border: "1px solid var(--border)", background: "var(--bg)",
  fontSize: 13, color: "var(--text)", outline: "none",
  boxSizing: "border-box", fontFamily: "inherit",
};

const lbl: React.CSSProperties = {
  fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
  textTransform: "uppercase", letterSpacing: "0.06em",
  display: "block", marginBottom: 6,
};

const fg: React.CSSProperties = { marginBottom: 18 };

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeIndustry(name: string): IndustryTarget {
  return {
    industry_id: crypto.randomUUID(),
    display_name: name,
    order: "both",
    first_order_description: "",
    second_order_description: "",
    use_builtin: true,
  };
}

function emptyProduct(): ProductEntry {
  return { name: "", value_prop: "", case_studies: [], target_roles: [], relevant_event_types: [] };
}

// ── Tag input for industries ──────────────────────────────────────────────────

function IndustryTagInput({
  tags, onAdd, onRemove,
}: {
  tags: IndustryTarget[];
  onAdd: (name: string) => void;
  onRemove: (id: string) => void;
}) {
  const [input, setInput] = useState("");

  function commit() {
    const v = input.trim();
    if (v && tags.length < 5) {
      onAdd(v);
      setInput("");
    }
  }

  function handleKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    if (e.key === "Backspace" && !input && tags.length > 0) {
      onRemove(tags[tags.length - 1].industry_id);
    }
  }

  return (
    <div>
      <div style={{
        display: "flex", flexWrap: "wrap", gap: 6, padding: "8px 10px",
        border: "1px solid var(--border)", borderRadius: 8, background: "var(--bg)",
        minHeight: 44, alignItems: "center",
      }}>
        {tags.map(t => (
          <span key={t.industry_id} style={{
            display: "inline-flex", alignItems: "center", gap: 5,
            padding: "3px 10px 3px 10px", borderRadius: 20,
            background: "var(--accent-light)", color: "var(--accent)",
            fontSize: 12, fontWeight: 500,
          }}>
            {t.display_name}
            <button
              onClick={() => onRemove(t.industry_id)}
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--accent)", display: "flex", padding: 0, opacity: 0.7 }}
            >
              <X size={11} strokeWidth={3} />
            </button>
          </span>
        ))}
        {tags.length < 5 && (
          <input
            style={{ border: "none", outline: "none", background: "transparent", fontSize: 13, color: "var(--text)", fontFamily: "inherit", minWidth: 120, flex: 1 }}
            placeholder={tags.length === 0 ? "Type an industry, press Enter…" : "Add another…"}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            onBlur={commit}
          />
        )}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 5 }}>
        {tags.length}/5 industries · Press Enter to add
      </div>
    </div>
  );
}

// ── Preview card ──────────────────────────────────────────────────────────────

interface PreviewState {
  userName: string;
  userRole: string;
  ownCompany: string;
  region: string;
  offerings: ProductEntry[];
  industries: IndustryTarget[];
  companySizes: string[];
  targetTitles: string;
  pathPref: UserProfile["path_preference"];
}

function ProfilePreviewCard({ s }: { s: PreviewState }) {
  const modeMeta = PATH_MODES.find(m => m.value === s.pathPref) ?? PATH_MODES[0];
  const regionName = regionLabel(s.region);
  const titleLine = [s.userName, s.userRole].filter(Boolean).join(" · ");
  const companyLine = [s.ownCompany, regionName !== "Global" ? regionName : ""].filter(Boolean).join(" · ");
  const sizes = s.companySizes.filter(sz => sz !== "any");

  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 14, padding: "20px 18px", position: "sticky", top: 24,
      minWidth: 210,
    }}>
      <div style={{ fontSize: 9, fontWeight: 800, color: "var(--text-xmuted)", letterSpacing: "0.1em", marginBottom: 14 }}>
        PROFILE PREVIEW
      </div>

      {/* Identity */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", lineHeight: 1.4 }}>
          {titleLine || <span style={{ color: "var(--text-xmuted)", fontStyle: "italic" }}>Your name</span>}
        </div>
        {companyLine && (
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{companyLine}</div>
        )}
      </div>

      {/* Offerings */}
      {s.offerings.filter(o => o.name.trim()).length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-xmuted)", letterSpacing: "0.09em", marginBottom: 5 }}>OFFERINGS</div>
          {s.offerings.filter(o => o.name.trim()).map((o, i) => (
            <div key={i} style={{ marginBottom: i < s.offerings.length - 1 ? 6 : 0 }}>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text)" }}>{o.name}</div>
              {o.value_prop && <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1, lineHeight: 1.5 }}>{o.value_prop.slice(0, 60)}{o.value_prop.length > 60 ? "…" : ""}</div>}
            </div>
          ))}
        </div>
      )}

      {/* Targeting */}
      {(s.industries.length > 0 || sizes.length > 0 || s.targetTitles) && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-xmuted)", letterSpacing: "0.09em", marginBottom: 5 }}>TARGETING</div>
          {s.industries.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 5 }}>
              {s.industries.map(ind => (
                <span key={ind.industry_id} style={{ fontSize: 10, padding: "2px 7px", borderRadius: 10, background: "var(--accent-light)", color: "var(--accent)", fontWeight: 500 }}>
                  {ind.display_name}
                </span>
              ))}
            </div>
          )}
          {sizes.length > 0 && (
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 3 }}>
              {sizes.map(sz => SIZE_OPTIONS.find(o => o.value === sz)?.label ?? sz).join(", ")}
            </div>
          )}
          {s.targetTitles && (
            <div style={{ fontSize: 11, color: "var(--text-muted)", fontStyle: "italic" }}>
              {s.targetTitles.split(",").slice(0, 3).map(t => t.trim()).filter(Boolean).join(", ")}
            </div>
          )}
        </div>
      )}

      {/* Mode */}
      <div>
        <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-xmuted)", letterSpacing: "0.09em", marginBottom: 5 }}>MODE</div>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 9px", borderRadius: 10, background: modeMeta.bg, color: modeMeta.color, fontSize: 11, fontWeight: 600 }}>
          {modeMeta.icon && <span style={{ display: "flex" }}>{modeMeta.icon}</span>}
          {modeMeta.label}
        </div>
      </div>
    </div>
  );
}

// ── Profile Picker ────────────────────────────────────────────────────────────

function ProfilePicker({ profiles, onSelect, onNew }: {
  profiles: UserProfile[];
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", background: "var(--bg)", padding: 32 }}>
      <div style={{ width: "min(480px, 100%)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 40 }}>
          <div style={{ width: 28, height: 28, background: "var(--text)", borderRadius: 7, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Zap size={13} color="var(--bg)" strokeWidth={2.5} />
          </div>
          <span className="font-display" style={{ fontSize: 17, color: "var(--text)" }}>{APP_NAME}</span>
        </div>
        <h1 className="font-display" style={{ fontSize: 26, color: "var(--text)", letterSpacing: "-0.02em", marginBottom: 6 }}>
          Choose a profile
        </h1>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 28, lineHeight: 1.6 }}>
          Each profile configures how Harbinger scans, scores, and generates outreach.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
          {profiles.map((p) => {
            const mode = PATH_MODES.find((m) => m.value === p.path_preference) ?? PATH_MODES[0];
            return (
              <button
                key={p.profile_id}
                onClick={() => onSelect(p.profile_id)}
                style={{
                  display: "flex", alignItems: "center", gap: 14, padding: "14px 16px",
                  border: "1px solid var(--border)", borderRadius: 12, background: "var(--surface)",
                  cursor: "pointer", textAlign: "left", transition: "border-color 150ms, background 150ms",
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--accent)"; (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLElement).style.background = "var(--surface)"; }}
              >
                <div style={{ width: 38, height: 38, borderRadius: 9, background: mode.bg, color: mode.color, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  {mode.icon}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", marginBottom: 2 }}>{p.user_name}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                    {p.own_company || "No company"} · {mode.label}
                  </div>
                </div>
                <ArrowRight size={14} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
              </button>
            );
          })}
        </div>
        <button
          onClick={onNew}
          style={{
            width: "100%", padding: "11px", fontSize: 13, fontWeight: 500,
            border: "2px dashed var(--border)", borderRadius: 10,
            background: "transparent", color: "var(--text-muted)", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
          }}
        >
          <Plus size={14} /> Create new profile
        </button>
      </div>
    </div>
  );
}

// ── Main Wizard ───────────────────────────────────────────────────────────────

function OnboardingWizard({ onComplete }: { onComplete: (profileId: string, companyName?: string) => void }) {
  const [step, setStep] = useState<StepId>(1);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Step 1 — You
  const [userName, setUserName] = useState("");
  const [userRole, setUserRole] = useState("");
  const [ownCompany, setOwnCompany] = useState("");
  const [region, setRegion] = useState("global");

  // Step 2 — Offerings (products + services)
  const [offerings, setOfferings] = useState<ProductEntry[]>([emptyProduct()]);

  // Step 3 — Your buyers
  const [industries, setIndustries] = useState<IndustryTarget[]>([]);
  const [companySizes, setCompanySizes] = useState<string[]>(["any"]);
  const [targetTitles, setTargetTitles] = useState("");

  // Step 4 — Mode
  const [pathPref, setPathPref] = useState<UserProfile["path_preference"]>("auto");

  // Step 5 — Mode config (only non-auto)
  const [accountText, setAccountText] = useState("");
  const [reportTitle, setReportTitle] = useState("");
  const [reportSummary, setReportSummary] = useState("");

  // Step 6 — Outreach
  const [fromName, setFromName] = useState("");
  const [fromEmail, setFromEmail] = useState("");

  // Which steps are visible? Step 5 is conditional on non-auto mode.
  const hasStep5 = pathPref !== "auto";

  // Compute the visible step count
  const totalSteps = hasStep5 ? 6 : 5;

  // Map visual step number → logical step id
  function logicalToVisual(logical: StepId): number {
    if (!hasStep5 && logical >= 5) return logical - 1;
    return logical;
  }

  function visualToLogical(visual: number): StepId {
    if (!hasStep5 && visual >= 5) return (visual + 1) as StepId;
    return visual as StepId;
  }

  const visualStep = logicalToVisual(step);

  function addIndustry(name: string) {
    if (industries.length < 5) {
      setIndustries(prev => [...prev, makeIndustry(name)]);
    }
  }

  function removeIndustry(id: string) {
    setIndustries(prev => prev.filter(i => i.industry_id !== id));
  }

  function toggleSize(val: string) {
    if (val === "any") {
      setCompanySizes(["any"]);
      return;
    }
    setCompanySizes(prev => {
      const without = prev.filter(v => v !== "any");
      if (without.includes(val)) {
        const next = without.filter(v => v !== val);
        return next.length === 0 ? ["any"] : next;
      }
      return [...without, val];
    });
  }

  function goNext() {
    if (step === 1 && !userName.trim()) { setError("Your name is required"); return; }
    setError(null);
    // Skip step 5 if auto mode
    if (step === 4 && pathPref === "auto") {
      setStep(6);
      return;
    }
    setStep(s => Math.min(6, s + 1) as StepId);
  }

  function goBack() {
    setError(null);
    // Skip step 5 backward if auto mode
    if (step === 6 && pathPref === "auto") {
      setStep(4);
      return;
    }
    setStep(s => Math.max(1, s - 1) as StepId);
  }

  const isFinalStep = step === 6;

  async function handleSave() {
    if (!userName.trim()) { setError("Your name is required"); return; }

    const validOfferings = offerings
      .filter(o => o.name.trim())
      .map(o => ({
        ...o,
        name: o.name.trim(),
        value_prop: o.value_prop.trim(),
        target_roles: targetTitles.split(",").map(s => s.trim()).filter(Boolean),
        relevant_event_types: [],
        case_studies: [],
      }));

    const payload: CreateProfileRequest = {
      user_name: userName.trim(),
      own_company: ownCompany.trim(),
      region,
      path_preference: pathPref,
      min_lead_score: 0.5,
      target_industries: industries,
      own_products: validOfferings,
      account_list: accountText.split("\n").map(s => s.trim()).filter(Boolean),
      report_title: reportTitle.trim(),
      report_summary: reportSummary.trim(),
      contact_hierarchy: [],
      email_config: { from_name: fromName.trim(), from_email: fromEmail.trim() },
    };

    setSaving(true);
    setError(null);
    try {
      const saved = await createProfile(payload);
      onComplete(saved.profile_id, ownCompany);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  const preview: PreviewState = {
    userName, userRole, ownCompany, region,
    offerings,
    industries, companySizes, targetTitles,
    pathPref,
  };

  // Visible steps for sidebar (skipping step 5 when auto)
  const visibleSteps = STEPS.filter(s => hasStep5 || s.id !== 5);

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "var(--bg)" }}>

      {/* ── Left sidebar ── */}
      <div style={{
        width: 240, flexShrink: 0, background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        display: "flex", flexDirection: "column", padding: "32px 20px",
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 44 }}>
          <div style={{ width: 26, height: 26, background: "var(--text)", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Zap size={12} color="var(--bg)" strokeWidth={2.5} />
          </div>
          <span className="font-display" style={{ fontSize: 15, color: "var(--text)" }}>{APP_NAME}</span>
        </div>

        {/* Step list */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-xmuted)", letterSpacing: "0.08em", marginBottom: 10 }}>
            SETUP
          </div>
          {visibleSteps.map((s, idx) => {
            const vNum = idx + 1;
            const done = visualStep > vNum;
            const active = visualStep === vNum;
            return (
              <div
                key={s.id}
                onClick={() => done ? setStep(s.id) : undefined}
                style={{
                  display: "flex", alignItems: "center", gap: 11, padding: "9px 10px",
                  borderRadius: 9, cursor: done ? "pointer" : "default",
                  background: active ? "var(--accent-light)" : "transparent",
                  transition: "background 150ms",
                }}
              >
                <div style={{
                  width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                  border: `2px solid ${done || active ? "var(--accent)" : "var(--border)"}`,
                  background: done ? "var(--accent)" : active ? "var(--accent-light)" : "var(--surface)",
                  color: done ? "#fff" : active ? "var(--accent)" : "var(--text-muted)",
                  fontSize: 9, fontWeight: 700,
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  {done ? <Check size={10} strokeWidth={3} /> : vNum}
                </div>
                <div>
                  <div style={{ fontSize: 12, fontWeight: active ? 600 : 400, color: active ? "var(--accent)" : done ? "var(--text)" : "var(--text-muted)" }}>
                    {s.title}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{s.desc}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Progress */}
        <div style={{ marginTop: "auto", paddingTop: 20 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 6 }}>
            Step {visualStep} of {totalSteps}
          </div>
          <div style={{ height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${((visualStep - 1) / (totalSteps - 1)) * 100}%`, background: "var(--accent)", borderRadius: 2, transition: "width 300ms ease" }} />
          </div>
        </div>
      </div>

      {/* ── Right content: two-column ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <div style={{ flex: 1, display: "flex", overflowY: "auto" }}>

          {/* Form column */}
          <div style={{ flex: "0 0 auto", width: "min(520px, 60%)", padding: "48px 40px 48px 56px" }}>
            {error && (
              <div style={{ padding: "10px 14px", background: "var(--red-light)", color: "var(--red)", borderRadius: 8, fontSize: 12, marginBottom: 20, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                {error}
                <button onClick={() => setError(null)} style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 16, padding: 0, lineHeight: 1 }}>×</button>
              </div>
            )}

            {step === 1 && (
              <StepYou
                userName={userName} setUserName={setUserName}
                userRole={userRole} setUserRole={setUserRole}
                ownCompany={ownCompany} setOwnCompany={setOwnCompany}
                region={region} setRegion={setRegion}
              />
            )}
            {step === 2 && (
              <StepOfferings
                offerings={offerings}
                setOfferings={setOfferings}
              />
            )}
            {step === 3 && (
              <StepYourBuyers
                industries={industries} onAddIndustry={addIndustry} onRemoveIndustry={removeIndustry}
                companySizes={companySizes} onToggleSize={toggleSize}
                targetTitles={targetTitles} setTargetTitles={setTargetTitles}
              />
            )}
            {step === 4 && (
              <StepHowToFind
                pathPref={pathPref} setPathPref={setPathPref}
                hasIndustries={industries.length > 0}
                hasOfferings={offerings.some(o => o.name.trim())}
              />
            )}
            {step === 5 && (
              <StepConfigure
                pathPref={pathPref}
                industries={industries}
                accountText={accountText} setAccountText={setAccountText}
                reportTitle={reportTitle} setReportTitle={setReportTitle}
                reportSummary={reportSummary} setReportSummary={setReportSummary}
              />
            )}
            {step === 6 && (
              <StepOutreach
                fromName={fromName} setFromName={setFromName}
                fromEmail={fromEmail} setFromEmail={setFromEmail}
              />
            )}
          </div>

          {/* Preview column */}
          <div style={{ flex: 1, padding: "48px 40px 48px 24px", minWidth: 0, display: "flex", flexDirection: "column" }}>
            <ProfilePreviewCard s={preview} />
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: "14px 56px", borderTop: "1px solid var(--border)",
          background: "var(--surface)", display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <button
            onClick={goBack}
            disabled={step === 1}
            style={{
              display: "flex", alignItems: "center", gap: 5, padding: "9px 16px",
              fontSize: 13, fontWeight: 500, borderRadius: 8,
              border: "1px solid var(--border)", background: "var(--surface-raised)",
              color: step === 1 ? "var(--text-xmuted)" : "var(--text-secondary)",
              cursor: step === 1 ? "not-allowed" : "pointer",
            }}
          >
            <ChevronLeft size={14} /> Back
          </button>

          {!isFinalStep ? (
            <button
              onClick={goNext}
              style={{
                display: "flex", alignItems: "center", gap: 5, padding: "9px 22px",
                fontSize: 13, fontWeight: 600, borderRadius: 8,
                border: "none", background: "var(--accent)", color: "#fff", cursor: "pointer",
              }}
            >
              Continue <ChevronRight size={14} />
            </button>
          ) : (
            <button
              onClick={handleSave}
              disabled={saving}
              style={{
                display: "flex", alignItems: "center", gap: 6, padding: "9px 24px",
                fontSize: 13, fontWeight: 600, borderRadius: 8,
                border: "none",
                background: saving ? "var(--surface-raised)" : "var(--accent)",
                color: saving ? "var(--text-muted)" : "#fff",
                cursor: saving ? "not-allowed" : "pointer",
              }}
            >
              {saving ? "Creating…" : <><Check size={14} strokeWidth={3} /> Create Profile</>}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Role suggestions ─────────────────────────────────────────────────────────

const ROLE_SUGGESTIONS = [
  "VP Sales", "Head of BD", "CEO", "Founder", "CMO",
  "VP Marketing", "Director of Sales", "Account Executive",
  "Head of Growth", "CRO",
];

// ── Industry suggestions ────────────────────────────────────────────────────

const INDUSTRY_SUGGESTIONS = [
  "Fintech & BFSI", "IT & Software", "Manufacturing", "Healthcare & Pharma",
  "Logistics & Supply Chain", "Retail & FMCG", "EdTech", "SaaS",
  "Cybersecurity", "Clean Energy", "Real Estate", "Telecom",
];

// ── Step 1: You ───────────────────────────────────────────────────────────────

function StepYou({ userName, setUserName, userRole, setUserRole, ownCompany, setOwnCompany, region, setRegion }: {
  userName: string; setUserName: (v: string) => void;
  userRole: string; setUserRole: (v: string) => void;
  ownCompany: string; setOwnCompany: (v: string) => void;
  region: string; setRegion: (v: string) => void;
}) {
  // Parse comma-separated roles into tags
  const roleTags = userRole.split(",").map(s => s.trim()).filter(Boolean);

  function addRole(role: string) {
    const trimmed = role.trim();
    if (!trimmed) return;
    if (roleTags.includes(trimmed)) return;
    setUserRole([...roleTags, trimmed].join(", "));
  }

  function removeRole(role: string) {
    setUserRole(roleTags.filter(r => r !== role).join(", "));
  }

  const [roleInput, setRoleInput] = useState("");

  function commitRoleInput() {
    if (roleInput.trim()) {
      addRole(roleInput.trim());
      setRoleInput("");
    }
  }

  return (
    <div>
      <StepHeader step={1} title="Tell us about you" subtitle="This helps personalise outreach emails and target the right decision-makers." />
      <div style={fg}>
        <label style={lbl}>Your name *</label>
        <input style={inp} type="text" value={userName} onChange={e => setUserName(e.target.value)} placeholder="e.g. Arjun Sharma" autoFocus />
      </div>
      <div style={fg}>
        <label style={lbl}>Your role <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>(pick or type your own)</span></label>
        {/* Selected role tags */}
        {roleTags.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {roleTags.map(r => (
              <span key={r} style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "4px 10px", borderRadius: 20,
                background: "var(--accent-light)", color: "var(--accent)",
                fontSize: 12, fontWeight: 500,
              }}>
                {r}
                <button
                  onClick={() => removeRole(r)}
                  style={{ background: "none", border: "none", cursor: "pointer", color: "var(--accent)", display: "flex", padding: 0, opacity: 0.7 }}
                >
                  <X size={11} strokeWidth={3} />
                </button>
              </span>
            ))}
          </div>
        )}
        {/* Custom role input */}
        <input
          style={inp} type="text" value={roleInput}
          onChange={e => setRoleInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Enter") { e.preventDefault(); commitRoleInput(); }
            if (e.key === "Backspace" && !roleInput && roleTags.length > 0) {
              removeRole(roleTags[roleTags.length - 1]);
            }
          }}
          onBlur={commitRoleInput}
          placeholder={roleTags.length === 0 ? "Type a role and press Enter, or pick below" : "Add another role…"}
        />
        {/* Suggestion chips — always visible, filtered to hide already-picked */}
        {(() => {
          const remaining = ROLE_SUGGESTIONS.filter(r => !roleTags.includes(r));
          return remaining.length > 0 ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 8 }}>
              {remaining.map(r => (
                <button
                  key={r}
                  onClick={() => addRole(r)}
                  style={{
                    padding: "4px 10px", fontSize: 11, borderRadius: 14,
                    border: "1px solid var(--border)", background: "var(--surface)",
                    color: "var(--text-secondary)", cursor: "pointer",
                    transition: "border-color 150ms, background 150ms",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--accent)"; e.currentTarget.style.background = "var(--accent-light)"; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.background = "var(--surface)"; }}
                >
                  + {r}
                </button>
              ))}
            </div>
          ) : null;
        })()}
      </div>
      <div style={fg}>
        <label style={lbl}>Company</label>
        <input style={inp} type="text" value={ownCompany} onChange={e => setOwnCompany(e.target.value)} placeholder="e.g. Acme Consulting" />
      </div>
      <div style={fg}>
        <label style={lbl}>Primary market</label>
        <select style={{ ...inp, cursor: "pointer" }} value={region} onChange={e => setRegion(e.target.value)}>
          {REGION_OPTIONS.map(r => <option key={r.code} value={r.code}>{r.name}</option>)}
        </select>
      </div>
    </div>
  );
}

// ── Step 2: Your offerings (products & services) ─────────────────────────────

function StepOfferings({ offerings, setOfferings }: {
  offerings: ProductEntry[];
  setOfferings: (v: ProductEntry[]) => void;
}) {
  function updateAt(idx: number, field: keyof ProductEntry, val: string) {
    setOfferings(offerings.map((o, i) => i === idx ? { ...o, [field]: val } : o));
  }

  function addOffering() {
    if (offerings.length < 8) {
      setOfferings([...offerings, emptyProduct()]);
    }
  }

  function removeOffering(idx: number) {
    setOfferings(offerings.filter((_, i) => i !== idx));
  }

  return (
    <div>
      <StepHeader step={2} title="Your products" subtitle="The pipeline matches market signals to your products and uses them to personalise outreach emails." />

      {offerings.length === 0 && (
        <div style={{
          textAlign: "center", padding: "36px 24px",
          border: "1px dashed var(--border)", borderRadius: 10, marginBottom: 14,
          background: "var(--surface)",
        }}>
          <div style={{ fontSize: 22, marginBottom: 8 }}>+</div>
          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 4 }}>
            No products yet
          </div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
            Add a product so Harbinger can connect signals to what you sell.
          </div>
        </div>
      )}

      {offerings.map((o, idx) => (
        <div key={idx} style={{
          padding: "16px", marginBottom: 12,
          border: "1px solid var(--border)", borderRadius: 10, background: "var(--surface)",
        }}>
          {/* Header with inline name */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
            <div style={{
              width: 24, height: 24, borderRadius: 6,
              background: "var(--accent-light)", color: "var(--accent)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 11, fontWeight: 700, flexShrink: 0,
            }}>
              {idx + 1}
            </div>
            <input
              style={{
                ...inp, fontWeight: 600, fontSize: 14, padding: "6px 10px", flex: 1,
                border: o.name ? "1px solid transparent" : "1px solid var(--border)",
                background: o.name ? "transparent" : "var(--bg)",
              }}
              type="text"
              value={o.name}
              onChange={e => updateAt(idx, "name", e.target.value)}
              placeholder="Product or service name"
              autoFocus={idx === 0 && !o.name}
            />
            {offerings.length > 0 && (
              <button
                onClick={() => removeOffering(idx)}
                title="Remove"
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  color: "var(--text-muted)", padding: 4, display: "flex",
                  opacity: 0.5, transition: "opacity 150ms",
                }}
                onMouseEnter={e => { e.currentTarget.style.opacity = "1"; e.currentTarget.style.color = "var(--red)"; }}
                onMouseLeave={e => { e.currentTarget.style.opacity = "0.5"; e.currentTarget.style.color = "var(--text-muted)"; }}
              >
                <X size={13} strokeWidth={2.5} />
              </button>
            )}
          </div>

          {/* Value prop */}
          <div>
            <label style={{ ...lbl, fontSize: 10 }}>
              Value proposition
              <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0, marginLeft: 4, opacity: 0.7 }}>
                — how it helps your buyers
              </span>
            </label>
            <textarea
              style={{ ...inp, resize: "vertical", lineHeight: 1.6, minHeight: 52 } as React.CSSProperties}
              rows={2}
              value={o.value_prop}
              onChange={e => updateAt(idx, "value_prop", e.target.value)}
              placeholder="e.g. Reduces compliance audit time by 60% for mid-market banks"
            />
          </div>
        </div>
      ))}

      {offerings.length < 8 && (
        <button
          onClick={addOffering}
          style={{
            width: "100%", padding: "10px", fontSize: 12, fontWeight: 500,
            border: "1.5px dashed var(--border)", borderRadius: 8,
            background: "transparent", color: "var(--text-muted)", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            marginBottom: 12,
          }}
        >
          <Plus size={13} /> Add product
        </button>
      )}

      <div style={{ padding: "10px 12px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, color: "var(--text-muted)", lineHeight: 1.6 }}>
        You can add targeting details like decision-maker roles and buying triggers later from the profile editor.
      </div>
    </div>
  );
}

// ── Step 3: Your buyers ───────────────────────────────────────────────────────

function StepYourBuyers({ industries, onAddIndustry, onRemoveIndustry, companySizes, onToggleSize, targetTitles, setTargetTitles }: {
  industries: IndustryTarget[];
  onAddIndustry: (name: string) => void;
  onRemoveIndustry: (id: string) => void;
  companySizes: string[];
  onToggleSize: (val: string) => void;
  targetTitles: string;
  setTargetTitles: (v: string) => void;
}) {
  const addedNames = new Set(industries.map(i => i.display_name));
  const suggestions = INDUSTRY_SUGGESTIONS.filter(s => !addedNames.has(s));

  return (
    <div>
      <StepHeader step={3} title="Who are your buyers?" subtitle="Define the industries, company sizes, and job titles you want to reach." />
      <div style={fg}>
        <label style={lbl}>Target industries <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>(up to 5)</span></label>
        <IndustryTagInput tags={industries} onAdd={onAddIndustry} onRemove={onRemoveIndustry} />
        {industries.length < 5 && suggestions.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 8 }}>
            <span style={{ fontSize: 10, color: "var(--text-xmuted)", alignSelf: "center", marginRight: 2 }}>Quick add:</span>
            {suggestions.slice(0, 6).map(s => (
              <button
                key={s}
                onClick={() => onAddIndustry(s)}
                style={{
                  padding: "3px 9px", fontSize: 11, borderRadius: 12,
                  border: "1px dashed var(--border)", background: "transparent",
                  color: "var(--text-muted)", cursor: "pointer",
                }}
              >
                + {s}
              </button>
            ))}
          </div>
        )}
      </div>
      <div style={fg}>
        <label style={lbl}>Company size</label>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {SIZE_OPTIONS.map(opt => {
            const active = companySizes.includes(opt.value);
            return (
              <button
                key={opt.value}
                onClick={() => onToggleSize(opt.value)}
                style={{
                  padding: "7px 16px", borderRadius: 20, fontSize: 12, fontWeight: 500,
                  border: `1.5px solid ${active ? "var(--accent)" : "var(--border)"}`,
                  background: active ? "var(--accent-light)" : "var(--surface)",
                  color: active ? "var(--accent)" : "var(--text-secondary)",
                  cursor: "pointer", transition: "all 150ms",
                }}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </div>
      <div style={fg}>
        <label style={lbl}>Target job titles</label>
        <input
          style={inp}
          type="text"
          value={targetTitles}
          onChange={e => setTargetTitles(e.target.value)}
          placeholder="e.g. CFO, VP Finance, Controller"
        />
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 5 }}>
          Comma-separated. Used to prioritise contacts during outreach.
        </div>
      </div>
    </div>
  );
}

// ── Step 4: How to find ───────────────────────────────────────────────────────

function StepHowToFind({ pathPref, setPathPref, hasIndustries, hasOfferings }: {
  pathPref: UserProfile["path_preference"];
  setPathPref: (v: UserProfile["path_preference"]) => void;
  hasIndustries: boolean;
  hasOfferings: boolean;
}) {
  // Dynamic recommendation hint
  const hint = hasIndustries && hasOfferings
    ? "Based on your industries and offerings, Auto mode will use Industry-First scanning."
    : hasIndustries
    ? "You have industries set — Industry-First mode is a great fit."
    : "Auto mode works best when starting out. You can switch anytime.";

  return (
    <div>
      <StepHeader step={4} title="How should we find leads?" subtitle="Choose a pipeline mode. You can change this any time." />
      <div style={{ padding: "8px 12px", background: "var(--accent-light)", borderRadius: 8, fontSize: 12, color: "var(--accent)", marginBottom: 16, lineHeight: 1.5 }}>
        {hint}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {PATH_MODES.map(mode => {
          const active = pathPref === mode.value;
          return (
            <button
              key={mode.value}
              onClick={() => setPathPref(mode.value)}
              style={{
                display: "flex", alignItems: "flex-start", gap: 14, padding: "14px 16px",
                borderRadius: 10,
                border: `1.5px solid ${active ? mode.color : "var(--border)"}`,
                background: active ? mode.bg : "var(--surface)",
                cursor: "pointer", textAlign: "left", transition: "all 150ms",
              }}
            >
              <div style={{
                width: 36, height: 36, borderRadius: 9, flexShrink: 0,
                background: active ? mode.color : "var(--surface-raised)",
                color: active ? "#fff" : "var(--text-muted)",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                {mode.icon}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: active ? mode.color : "var(--text)" }}>{mode.label}</span>
                  {mode.recommended && (
                    <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 10, background: "var(--accent)", color: "#fff", letterSpacing: "0.02em" }}>
                      RECOMMENDED
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.4 }}>{mode.subtitle}</div>
              </div>
              <div style={{
                width: 18, height: 18, borderRadius: "50%", flexShrink: 0, marginTop: 2,
                border: `2px solid ${active ? mode.color : "var(--border)"}`,
                background: active ? mode.color : "transparent",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                {active && <Check size={9} strokeWidth={3} color="#fff" />}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ── Step 5: Configure (conditional) ──────────────────────────────────────────

function StepConfigure({ pathPref, industries, accountText, setAccountText, reportTitle, setReportTitle, reportSummary, setReportSummary }: {
  pathPref: UserProfile["path_preference"];
  industries: IndustryTarget[];
  accountText: string; setAccountText: (v: string) => void;
  reportTitle: string; setReportTitle: (v: string) => void;
  reportSummary: string; setReportSummary: (v: string) => void;
}) {
  return (
    <div>
      <StepHeader step={5} title="Configure targeting" subtitle="Fine-tune the inputs for your chosen pipeline mode." />

      {pathPref === "industry_first" && (
        <div>
          <div style={{ padding: "10px 12px", borderRadius: 8, background: "var(--green-light)", border: "1px solid rgba(45,106,79,0.15)", fontSize: 12, color: "var(--green)", marginBottom: 16, lineHeight: 1.6 }}>
            Harbinger will track trends and signals across these industries.
          </div>
          {industries.length > 0 ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginBottom: 12 }}>
              {industries.map(ind => (
                <span key={ind.industry_id} style={{ padding: "5px 12px", borderRadius: 20, background: "var(--accent-light)", color: "var(--accent)", fontSize: 12, fontWeight: 500 }}>
                  {ind.display_name}
                </span>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: 13, color: "var(--text-muted)", padding: "12px 0", marginBottom: 12 }}>
              No industries added yet — go back to Step 3 to add them.
            </div>
          )}
          <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.6 }}>
            These were set in "Your buyers". You can add more from your profile settings.
          </div>
        </div>
      )}

      {pathPref === "company_first" && (
        <div>
          <div style={fg}>
            <label style={lbl}>Account list — one company per line</label>
            <textarea
              style={{ ...inp, resize: "vertical", minHeight: 160, lineHeight: 1.6 } as React.CSSProperties}
              rows={8}
              value={accountText}
              onChange={e => setAccountText(e.target.value)}
              placeholder={"Infosys\nTCS\nWipro\nHCL Technologies"}
            />
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 5 }}>
              {accountText.split("\n").filter(Boolean).length} companies
            </div>
          </div>
        </div>
      )}

      {pathPref === "report_driven" && (
        <div>
          <div style={fg}>
            <label style={lbl}>Report title</label>
            <input style={inp} type="text" value={reportTitle} onChange={e => setReportTitle(e.target.value)} placeholder="e.g. Q2 2026 Fintech Landscape" />
          </div>
          <div style={fg}>
            <label style={lbl}>Research brief</label>
            <textarea
              style={{ ...inp, resize: "vertical", minHeight: 130, lineHeight: 1.6 } as React.CSSProperties}
              rows={5}
              value={reportSummary}
              onChange={e => setReportSummary(e.target.value)}
              placeholder="Describe the report topic, scope, and what companies or trends you want to surface…"
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Step 6: Outreach ──────────────────────────────────────────────────────────

function StepOutreach({ fromName, setFromName, fromEmail, setFromEmail }: {
  fromName: string; setFromName: (v: string) => void;
  fromEmail: string; setFromEmail: (v: string) => void;
}) {
  return (
    <div>
      <StepHeader step={6} title="Sender identity" subtitle="Outreach emails are sent via Brevo using your sender details. You can update these later in Settings." />
      <div style={fg}>
        <label style={lbl}>From name</label>
        <input style={inp} type="text" value={fromName} onChange={e => setFromName(e.target.value)} placeholder="e.g. Arjun from Acme" />
      </div>
      <div style={fg}>
        <label style={lbl}>From email</label>
        <input style={inp} type="email" value={fromEmail} onChange={e => setFromEmail(e.target.value)} placeholder="arjun@acme.com" />
      </div>
      <div style={{ padding: "10px 12px", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, color: "var(--text-muted)", lineHeight: 1.6 }}>
        Lead quality threshold is set to a balanced default (0.5). You can tune it in Settings after setup.
      </div>
    </div>
  );
}

// ── Shared step header ────────────────────────────────────────────────────────

function StepHeader({ step, title, subtitle }: { step: number; title: string; subtitle: string }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "var(--accent)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>
        Step {step}
      </div>
      <h2 style={{ fontSize: 26, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.025em", marginBottom: 8, lineHeight: 1.2 }}>
        {title}
      </h2>
      <p style={{ fontSize: 14, color: "var(--text-muted)", lineHeight: 1.65 }}>{subtitle}</p>
    </div>
  );
}

// ── Page entry ────────────────────────────────────────────────────────────────

function OnboardingContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const fromProfiles = searchParams.get("from") === "profiles";

  const [checking, setChecking] = useState(true);
  const [profiles, setProfiles] = useState<UserProfile[]>([]);
  const [showPicker, setShowPicker] = useState(false);
  const [showWizard, setShowWizard] = useState(false);

  useEffect(() => {
    listProfiles()
      .then(res => {
        const ps = res.profiles;
        setProfiles(ps);
        if (ps.length === 0) {
          setShowWizard(true);
        } else if (ps.length === 1 && !fromProfiles) {
          localStorage.setItem(STORAGE_KEY, ps[0].profile_id);
          router.replace("/dashboard");
        } else {
          setShowPicker(true);
        }
      })
      .catch(() => setShowWizard(true))
      .finally(() => setChecking(false));
  }, [router, fromProfiles]);

  const handleSelect = useCallback((id: string) => {
    localStorage.setItem(STORAGE_KEY, id);
    router.push(fromProfiles ? "/profiles" : "/dashboard");
  }, [router, fromProfiles]);

  const [settingUp, setSettingUp] = useState(false);
  const [setupStep, setSetupStep] = useState("");

  const handleComplete = useCallback(async (profileId: string, companyName?: string) => {
    localStorage.setItem(STORAGE_KEY, profileId);

    // If a company name was provided, enrich in background while showing loading
    if (companyName?.trim()) {
      setSettingUp(true);
      setSetupStep("Verifying your company…");
      try {
        const result = await api.searchCompanies(companyName.trim());
        const co = result.companies?.[0];
        if (co) {
          setSetupStep("Pulling products & industry data…");
          // Build enrichment patch from search results
          const patch: Record<string, unknown> = { profile_id: profileId };

          // Auto-fill products from enrichment if user didn't add any
          if (co.products_services && co.products_services.length > 0) {
            const existingProfile = await getProfile(profileId).catch(() => null);
            if (!existingProfile?.own_products?.length) {
              patch.own_products = co.products_services.slice(0, 5).map((name: string) => ({
                name,
                value_prop: "",
                case_studies: [],
                target_roles: [],
                relevant_event_types: [],
              }));
            }
          }

          // Auto-fill industry if none set
          if (co.industry) {
            const existingProfile = await getProfile(profileId).catch(() => null);
            if (!existingProfile?.target_industries?.length && co.industry) {
              patch.target_industries = [{
                industry_id: co.industry.toLowerCase().replace(/[^a-z0-9]+/g, "_"),
                display_name: co.industry,
                order: "both",
                first_order_description: "",
                second_order_description: "",
                use_builtin: false,
              }];
            }
          }

          // Only update if we have something to add
          if (Object.keys(patch).length > 1) {
            setSetupStep("Updating your profile…");
            await updateProfile(profileId, patch as Parameters<typeof updateProfile>[1]).catch(() => {});
          }
        }
      } catch {
        // Enrichment is best-effort — don't block the user
      }
      setSetupStep("Taking you to your dashboard…");
      await new Promise(r => setTimeout(r, 600));
    }

    router.push("/dashboard");
  }, [router]);

  if (settingUp) {
    return (
      <div style={{
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        minHeight: "100vh", background: "var(--bg)", gap: 20,
      }}>
        <div style={{
          width: 48, height: 48, borderRadius: 12,
          background: "var(--text)", display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Zap size={22} color="var(--bg)" strokeWidth={2.5} />
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: "var(--text)", marginBottom: 8, letterSpacing: "-0.02em" }}>
            Setting up your dashboard
          </div>
          <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 20 }}>
            {setupStep}
          </div>
          <div style={{
            width: 200, height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden", margin: "0 auto",
          }}>
            <div style={{
              height: "100%", width: "60%",
              background: "linear-gradient(90deg, var(--accent), var(--accent))",
              borderRadius: 2,
              animation: "progress-indeterminate 1.4s ease-in-out infinite",
            }} />
          </div>
        </div>
      </div>
    );
  }

  if (checking) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", background: "var(--bg)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text-muted)", fontSize: 13 }}>
          <div style={{ width: 16, height: 16, border: "2px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
          Loading…
        </div>
      </div>
    );
  }

  if (showWizard) {
    return (
      <div style={{ position: "relative" }}>
        <OnboardingWizard onComplete={handleComplete} />
        <button
          onClick={() => router.push("/dashboard")}
          style={{ position: "fixed", bottom: 20, left: "50%", transform: "translateX(-50%)", fontSize: 12, color: "var(--text-muted)", background: "none", border: "none", cursor: "pointer", zIndex: 10, textDecoration: "underline" }}
        >
          {profiles.length > 0 ? "Skip for now → go to dashboard" : "Skip setup → explore the app first"}
        </button>
      </div>
    );
  }

  if (showPicker) {
    return (
      <ProfilePicker
        profiles={profiles}
        onSelect={handleSelect}
        onNew={() => { setShowPicker(false); setShowWizard(true); }}
      />
    );
  }

  return null;
}

export default function OnboardingPage() {
  return (
    <Suspense>
      <OnboardingContent />
    </Suspense>
  );
}
