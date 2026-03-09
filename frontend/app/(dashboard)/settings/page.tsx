"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  Sun, Moon, Zap, RefreshCw, Check,
  CheckCircle, AlertCircle, XCircle,
  Wifi, WifiOff, Clock, Server,
  Search, Mail, Brain, Globe, Users, Newspaper,
} from "lucide-react";
import { api } from "@/lib/api";
import type { HealthResponse, ProviderHealth } from "@/lib/types";

type Tab = "appearance" | "pipeline" | "enrichment" | "contacts" | "news_email" | "connections";

const TABS: { id: Tab; label: string }[] = [
  { id: "appearance",  label: "Appearance" },
  { id: "pipeline",    label: "Pipeline" },
  { id: "enrichment",  label: "Enrichment" },
  { id: "contacts",    label: "Contacts" },
  { id: "news_email",  label: "News & Email" },
  { id: "connections",  label: "Connections" },
];

// ── All countries via Intl API ──────────────────────

const ALL_COUNTRIES: { code: string; name: string }[] = (() => {
  try {
    const regionNames = new Intl.DisplayNames(["en"], { type: "region" });
    const codes = [
      "AF","AL","DZ","AD","AO","AG","AR","AM","AU","AT","AZ","BS","BH","BD","BB",
      "BY","BE","BZ","BJ","BT","BO","BA","BW","BR","BN","BG","BF","BI","KH","CM",
      "CA","CV","CF","TD","CL","CN","CO","KM","CG","CR","HR","CU","CY","CZ","CD",
      "DK","DJ","DM","DO","EC","EG","SV","GQ","ER","EE","SZ","ET","FJ","FI","FR",
      "GA","GM","GE","DE","GH","GR","GD","GT","GN","GW","GY","HT","HN","HU","IS",
      "IN","ID","IR","IQ","IE","IL","IT","CI","JM","JP","JO","KZ","KE","KI","KW",
      "KG","LA","LV","LB","LS","LR","LY","LI","LT","LU","MG","MW","MY","MV","ML",
      "MT","MH","MR","MU","MX","FM","MD","MC","MN","ME","MA","MZ","MM","NA","NR",
      "NP","NL","NZ","NI","NE","NG","KP","MK","NO","OM","PK","PW","PS","PA","PG",
      "PY","PE","PH","PL","PT","QA","RO","RU","RW","KN","LC","VC","WS","SM","ST",
      "SA","SN","RS","SC","SL","SG","SK","SI","SB","SO","ZA","KR","SS","ES","LK",
      "SD","SR","SE","CH","SY","TW","TJ","TZ","TH","TL","TG","TO","TT","TN","TR",
      "TM","TV","UG","UA","AE","GB","US","UY","UZ","VU","VE","VN","YE","ZM","ZW",
    ];
    return codes
      .map((c) => ({ code: c, name: regionNames.of(c) ?? c }))
      .sort((a, b) => a.name.localeCompare(b.name));
  } catch {
    return [{ code: "IN", name: "India" }, { code: "US", name: "United States" }];
  }
})();

// ── Reusable primitives ─────────────────────────────

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-xmuted)", letterSpacing: "0.08em", marginBottom: 12, marginTop: 4 }}>
      {children}
    </div>
  );
}

function SettingRow({ label, description, children }: { label: string; description?: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 24, padding: "14px 0", borderBottom: "1px solid var(--border)" }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)", marginBottom: description ? 2 : 0 }}>{label}</div>
        {description && <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5 }}>{description}</div>}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </div>
  );
}

function StatusDot({ status }: { status: "online" | "ready" | "degraded" | "offline" }) {
  const color = status === "online" ? "var(--green)" : status === "ready" ? "var(--blue)" : status === "degraded" ? "var(--amber)" : "var(--red)";
  return <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block", flexShrink: 0 }} />;
}

function StatusBadge({ status }: { status: "online" | "ready" | "degraded" | "offline" }) {
  const map = {
    online:   { cls: "badge-green", label: "Online",   Icon: CheckCircle },
    ready:    { cls: "badge-blue",  label: "Ready",    Icon: CheckCircle },
    degraded: { cls: "badge-amber", label: "Degraded", Icon: AlertCircle },
    offline:  { cls: "badge-red",   label: "Offline",  Icon: XCircle },
  };
  const { cls, label, Icon } = map[status];
  return (
    <span className={`badge ${cls}`} style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 3 }}>
      <Icon size={10} />
      {label}
    </span>
  );
}

function effectiveStatus(info: ProviderHealth): "online" | "ready" | "degraded" | "offline" {
  if (info.failure_count === 0) return "online";
  if (info.backoff_until) {
    const expiry = new Date(info.backoff_until).getTime();
    if (Date.now() > expiry) return "ready";
  }
  if (info.status === "broken" || info.failure_count >= 10) return "offline";
  return "degraded";
}

function InfoNote({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ marginTop: 20, padding: "12px 14px", borderRadius: 8, background: "var(--accent-light)", border: "1px solid var(--border)", display: "flex", alignItems: "flex-start", gap: 8 }}>
      <Zap size={14} style={{ color: "var(--accent)", flexShrink: 0, marginTop: 1 }} />
      <span style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
        {children}
      </span>
    </div>
  );
}

function ToggleSwitch({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      style={{
        position: "relative", width: 36, height: 20, borderRadius: 10, padding: 0,
        border: "1px solid var(--border)",
        background: checked ? "var(--accent)" : "var(--surface-raised)",
        cursor: "pointer", transition: "background 200ms", flexShrink: 0,
      }}
    >
      <span
        style={{
          position: "absolute", top: 2, left: checked ? 17 : 2,
          width: 14, height: 14, borderRadius: "50%",
          background: checked ? "#fff" : "var(--text-muted)",
          transition: "left 200ms",
        }}
      />
    </button>
  );
}

// ── Stored preference helpers ───────────────────────

function getPref(key: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return localStorage.getItem(`harbinger_${key}`) ?? fallback;
}

function setPref(key: string, value: string) {
  if (typeof window !== "undefined") localStorage.setItem(`harbinger_${key}`, value);
}

// ── Settings sync hook ──────────────────────────────

function useSetting(key: string, healthVal: unknown, fallback: string) {
  const [value, setValue] = useState(() => getPref(key, fallback));
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (healthVal != null && !getPref(key, "")) {
      setValue(String(healthVal));
    }
  }, [healthVal, key]);

  const update = useCallback((newVal: string) => {
    setValue(newVal);
    setPref(key, newVal);
    setSaved(false);

    // Parse value for backend
    let parsed: unknown = newVal;
    if (newVal === "true") parsed = true;
    else if (newVal === "false") parsed = false;
    else if (/^\d+$/.test(newVal)) parsed = parseInt(newVal, 10);
    else if (/^\d+\.\d+$/.test(newVal)) parsed = parseFloat(newVal);

    api.updateSettings({ [key]: parsed })
      .then(() => { setSaved(true); setTimeout(() => setSaved(false), 1500); })
      .catch(() => {});
  }, [key]);

  return { value, update, saved };
}

/** Inline save indicator */
function SaveCheck({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return <Check size={12} style={{ color: "var(--green)", marginLeft: 4, transition: "opacity 300ms" }} />;
}

/** Slider with value display */
function SliderSetting({ value, min, max, step, onChange, unit }: {
  value: string; min: number; max: number; step?: number; onChange: (v: string) => void; unit?: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <input
        type="range" min={min} max={max} step={step ?? 1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{ width: 100, cursor: "pointer", accentColor: "var(--accent)" }}
      />
      <span className="num" style={{ fontSize: 14, color: "var(--text)", minWidth: 32, textAlign: "right" }}>
        {value}{unit ?? ""}
      </span>
    </div>
  );
}

/** Tag editor — visual chips with add/remove */
function TagEditor({ value, onChange, saved, placeholder }: {
  value: string; onChange: (v: string) => void; saved: boolean; placeholder?: string;
}) {
  const [input, setInput] = useState("");
  const tags = value.split(",").map((t) => t.trim()).filter(Boolean);

  const addTag = () => {
    const trimmed = input.trim();
    if (!trimmed || tags.includes(trimmed)) return;
    onChange([...tags, trimmed].join(","));
    setInput("");
  };

  const removeTag = (tag: string) => {
    onChange(tags.filter((t) => t !== tag).join(","));
  };

  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: tags.length ? 8 : 0 }}>
        {tags.map((tag) => (
          <span
            key={tag}
            style={{
              display: "inline-flex", alignItems: "center", gap: 4,
              padding: "4px 10px", borderRadius: 14, fontSize: 11, fontWeight: 500,
              background: "var(--surface-hover)", color: "var(--text)",
              border: "1px solid var(--border)",
            }}
          >
            {tag}
            <button
              onClick={() => removeTag(tag)}
              style={{
                display: "flex", alignItems: "center", justifyContent: "center",
                width: 14, height: 14, borderRadius: "50%", border: "none", padding: 0,
                background: "var(--text-xmuted)", color: "var(--surface)",
                cursor: "pointer", fontSize: 10, lineHeight: 1,
              }}
            >
              ×
            </button>
          </span>
        ))}
        <SaveCheck visible={saved} />
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addTag(); } }}
          placeholder={placeholder ?? "Add role and press Enter"}
          style={{
            padding: "5px 10px", borderRadius: 7, border: "1px solid var(--border)",
            background: "var(--surface)", fontSize: 12, color: "var(--text)",
            flex: 1, minWidth: 0, outline: "none",
          }}
        />
        <button
          onClick={addTag}
          disabled={!input.trim()}
          style={{
            padding: "5px 12px", borderRadius: 7, border: "1px solid var(--border)",
            background: input.trim() ? "var(--accent)" : "var(--surface)",
            color: input.trim() ? "#fff" : "var(--text-muted)",
            fontSize: 12, cursor: input.trim() ? "pointer" : "default",
            fontWeight: 500,
          }}
        >
          Add
        </button>
      </div>
    </div>
  );
}

/** Person intel source checkboxes */
const PERSON_INTEL_SOURCES = [
  { key: "linkedin",     label: "LinkedIn / Apollo",        desc: "Professional profile via Apollo data + web search" },
  { key: "medium",       label: "Medium blogs",             desc: "Articles and thought leadership posts" },
  { key: "substack",     label: "Substack newsletters",     desc: "Newsletter content and subscriber-facing articles" },
  { key: "company_bio",  label: "Company team pages",       desc: "Bio, achievements, and role details from company sites" },
  { key: "conferences",  label: "Conferences & podcasts",   desc: "Speaking topics, panel appearances, event mentions" },
  { key: "github",       label: "GitHub profile",           desc: "Open-source repos and contributions (best for tech roles)" },
];

function SourceCheckboxes({ value, onChange, saved }: {
  value: string; onChange: (v: string) => void; saved: boolean;
}) {
  const activeSources = new Set(value.split(",").map((s) => s.trim()).filter(Boolean));

  const toggle = (key: string) => {
    const next = new Set(activeSources);
    if (next.has(key)) next.delete(key); else next.add(key);
    onChange([...next].join(","));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      {PERSON_INTEL_SOURCES.map(({ key, label, desc }) => (
        <div
          key={key}
          onClick={() => toggle(key)}
          style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "10px 12px", cursor: "pointer",
            borderBottom: "1px solid var(--border)",
            background: activeSources.has(key) ? "var(--accent-light)" : "transparent",
            borderRadius: 0, transition: "background 150ms",
          }}
        >
          <div style={{
            width: 16, height: 16, borderRadius: 4, flexShrink: 0,
            border: activeSources.has(key) ? "none" : "1.5px solid var(--text-muted)",
            background: activeSources.has(key) ? "var(--accent)" : "transparent",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            {activeSources.has(key) && <Check size={11} style={{ color: "#fff" }} />}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text)" }}>{label}</div>
            <div style={{ fontSize: 10, color: "var(--text-muted)", lineHeight: 1.4 }}>{desc}</div>
          </div>
        </div>
      ))}
      <SaveCheck visible={saved} />
    </div>
  );
}

// ── Appearance Tab ──────────────────────────────────

function AppearanceTab() {
  const [dark, setDark] = useState(false);
  const [terminalVerbosity, setTerminalVerbosity] = useState(() => getPref("terminal_verbosity", "standard"));

  useEffect(() => {
    const stored = localStorage.getItem("harbinger_dark");
    setDark(stored === "true" || (!stored && document.documentElement.classList.contains("dark")));
  }, []);

  const toggleDark = (val: boolean) => {
    setDark(val);
    document.documentElement.classList.toggle("dark", val);
    localStorage.setItem("harbinger_dark", String(val));
  };

  return (
    <div style={{ maxWidth: 560 }}>
      <SectionTitle>THEME</SectionTitle>
      <SettingRow label="Colour mode" description="Switch between light and dark interface">
        <div style={{ display: "flex", gap: 6 }}>
          {[{ val: false, Icon: Sun, label: "Light" }, { val: true, Icon: Moon, label: "Dark" }].map(({ val, Icon, label }) => (
            <button
              key={label}
              onClick={() => toggleDark(val)}
              style={{
                display: "flex", alignItems: "center", gap: 5,
                padding: "6px 12px", borderRadius: 7, fontSize: 12, fontWeight: 500,
                cursor: "pointer", border: "1px solid var(--border)",
                background: dark === val ? "var(--surface-hover)" : "var(--surface)",
                color: dark === val ? "var(--text)" : "var(--text-muted)",
                transition: "background 150ms, color 150ms",
              }}
            >
              <Icon size={12} />
              {label}
            </button>
          ))}
        </div>
      </SettingRow>

      <div style={{ marginTop: 24 }} />
      <SectionTitle>TERMINAL</SectionTitle>
      <SettingRow label="Pipeline log detail" description="Controls how much detail the terminal shows during a run">
        <select
          value={terminalVerbosity}
          onChange={(e) => { setTerminalVerbosity(e.target.value); setPref("terminal_verbosity", e.target.value); }}
          style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
        >
          <option value="minimal">Minimal</option>
          <option value="standard">Standard</option>
          <option value="verbose">Verbose</option>
        </select>
      </SettingRow>
    </div>
  );
}

// ── Pipeline Tab ────────────────────────────────────

function PipelineTab({ health }: { health: HealthResponse | null }) {
  const cfg = health?.config ?? {} as Record<string, unknown>;
  const country = useSetting("country", cfg.country, "India");
  const maxTrends = useSetting("max_trends", cfg.max_trends, "12");
  const coherenceMin = useSetting("coherence_min", cfg.coherence_min, "0.48");
  const mergeThreshold = useSetting("merge_threshold", cfg.merge_threshold, "0.82");
  const minSynthConf = useSetting("min_synthesis_confidence", cfg.min_synthesis_confidence, "0.40");
  const companyMinRel = useSetting("company_min_relevance", cfg.company_min_relevance, "0.20");
  const synthTimeout = useSetting("engine_synthesis_timeout", cfg.engine_synthesis_timeout, "600");
  const causalTimeout = useSetting("engine_causal_timeout", cfg.engine_causal_timeout, "300");
  const leadGenTimeout = useSetting("lead_gen_timeout", cfg.lead_gen_timeout, "900");

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 48px" }}>
      <div>
        <SectionTitle>TARGET MARKET</SectionTitle>
        <SettingRow label="Country" description="Primary market for news scanning and company discovery">
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Globe size={13} style={{ color: "var(--text-muted)" }} />
            <select
              value={country.value}
              onChange={(e) => country.update(e.target.value)}
              style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none", minWidth: 160 }}
            >
              <option value="">Select country...</option>
              {ALL_COUNTRIES.map(({ code, name }) => <option key={code} value={name}>{name}</option>)}
            </select>
            <SaveCheck visible={country.saved} />
          </div>
        </SettingRow>

        <div style={{ marginTop: 24 }}>
          <SectionTitle>COLLECTION</SectionTitle>
          <SettingRow label="Max trends per run" description="Market signals to extract per pipeline run">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={maxTrends.value} min={3} max={30} onChange={maxTrends.update} />
              <SaveCheck visible={maxTrends.saved} />
            </div>
          </SettingRow>
        </div>

        <div style={{ marginTop: 24 }}>
          <SectionTitle>TIMEOUTS</SectionTitle>
          <SettingRow label="Trend synthesis" description="Max seconds per cluster analysis">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={synthTimeout.value} min={60} max={1800} step={30} onChange={synthTimeout.update} unit="s" />
              <SaveCheck visible={synthTimeout.saved} />
            </div>
          </SettingRow>
          <SettingRow label="Causal council" description="Max seconds for cross-trend debate">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={causalTimeout.value} min={60} max={1800} step={30} onChange={causalTimeout.update} unit="s" />
              <SaveCheck visible={causalTimeout.saved} />
            </div>
          </SettingRow>
          <SettingRow label="Lead generation" description="Max seconds for full lead gen per run">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={leadGenTimeout.value} min={120} max={3600} step={60} onChange={leadGenTimeout.update} unit="s" />
              <SaveCheck visible={leadGenTimeout.saved} />
            </div>
          </SettingRow>
        </div>
      </div>

      <div>
        <SectionTitle>QUALITY GATES</SectionTitle>
        <SettingRow label="Cluster coherence minimum" description="Trends below this are discarded as noise">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={coherenceMin.value} min={0} max={1} step={0.01} onChange={coherenceMin.update} />
            <SaveCheck visible={coherenceMin.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Merge similarity threshold" description="Clusters more similar than this are merged">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={mergeThreshold.value} min={0} max={1} step={0.01} onChange={mergeThreshold.update} />
            <SaveCheck visible={mergeThreshold.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Synthesis confidence floor" description="Trends below this confidence are dropped">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={minSynthConf.value} min={0} max={1} step={0.01} onChange={minSynthConf.update} />
            <SaveCheck visible={minSynthConf.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Company relevance floor" description="Companies below this are excluded">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={companyMinRel.value} min={0} max={1} step={0.01} onChange={companyMinRel.update} />
            <SaveCheck visible={companyMinRel.saved} />
          </div>
        </SettingRow>
      </div>
    </div>
  );
}

// ── Enrichment Tab ──────────────────────────────────

function EnrichmentTab({ health }: { health: HealthResponse | null }) {
  const cfg = health?.config ?? {} as Record<string, unknown>;
  const deepEnabled = useSetting("deep_enrichment_enabled", cfg.deep_enrichment_enabled, "true");
  const websiteScrape = useSetting("website_scrape_enabled", cfg.website_scrape_enabled, "true");
  const hiringSignals = useSetting("hiring_signals_enabled", cfg.hiring_signals_enabled, "true");
  const techIp = useSetting("tech_ip_analysis_enabled", cfg.tech_ip_analysis_enabled, "true");
  const sgModel = useSetting("scrapegraph_model", cfg.scrapegraph_model, "openai/gpt-4.1-mini");
  const sgMaxResults = useSetting("scrapegraph_max_results", cfg.scrapegraph_max_results, "3");
  const sgTimeout = useSetting("scrapegraph_timeout", cfg.scrapegraph_timeout, "90");
  const personDeep = useSetting("person_deep_intel_enabled", cfg.person_deep_intel_enabled, "true");
  const personSources = useSetting("person_intel_sources", cfg.person_intel_sources, "medium,substack,github,company_bio,conferences");
  const personStaleness = useSetting("person_intel_staleness_days", cfg.person_intel_staleness_days, "7");
  const personMaxUrls = useSetting("person_intel_max_urls", cfg.person_intel_max_urls, "5");
  const cacheDays = useSetting("company_cache_days", cfg.company_cache_days, "7");
  const cacheEnabled = useSetting("company_cache_enabled", cfg.company_cache_enabled, "true");

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 48px" }}>
      <div>
        <SectionTitle>COMPANY DEEP INTEL</SectionTitle>
        <SettingRow label="Background deep enrichment" description="Run ScrapeGraphAI, hiring analysis, and tech/IP signals in background">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={deepEnabled.value === "true"} onChange={(v) => deepEnabled.update(String(v))} />
            <SaveCheck visible={deepEnabled.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Website scraping" description="SmartScraper on company homepage, about, and products pages">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={websiteScrape.value === "true"} onChange={(v) => websiteScrape.update(String(v))} />
            <SaveCheck visible={websiteScrape.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Hiring signal analysis" description="Analyze job postings as buying signals">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={hiringSignals.value === "true"} onChange={(v) => hiringSignals.update(String(v))} />
            <SaveCheck visible={hiringSignals.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Tech & IP intelligence" description="Patent, tech stack, and partnership signals">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={techIp.value === "true"} onChange={(v) => techIp.update(String(v))} />
            <SaveCheck visible={techIp.saved} />
          </div>
        </SettingRow>

        <div style={{ marginTop: 24 }}>
          <SectionTitle>SCRAPEGRAPHAI</SectionTitle>
          <SettingRow label="Model" description="LLM used for web scraping and data extraction">
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <select
                value={sgModel.value}
                onChange={(e) => sgModel.update(e.target.value)}
                style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none", minWidth: 200 }}
              >
                <option value="openai/gpt-4.1-mini">GPT-4.1 Mini (fast, recommended)</option>
                <option value="openai/gpt-4.1">GPT-4.1 (highest quality)</option>
                <option value="openai/gpt-4o-mini">GPT-4o Mini</option>
                <option value="openai/gpt-4o">GPT-4o</option>
                <option value="openai/gpt-3.5-turbo">GPT-3.5 Turbo (cheapest)</option>
              </select>
              <SaveCheck visible={sgModel.saved} />
            </div>
          </SettingRow>
          <SettingRow label="Max search results" description="SearchGraph result count per query">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={sgMaxResults.value} min={1} max={10} onChange={sgMaxResults.update} />
              <SaveCheck visible={sgMaxResults.saved} />
            </div>
          </SettingRow>
          <SettingRow label="Timeout" description="Max seconds per ScrapeGraphAI operation">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={sgTimeout.value} min={30} max={300} step={10} onChange={sgTimeout.update} unit="s" />
              <SaveCheck visible={sgTimeout.saved} />
            </div>
          </SettingRow>
        </div>
      </div>

      <div>
        <SectionTitle>PERSON INTELLIGENCE</SectionTitle>
        <SettingRow label="Deep person research" description="Scrape public profiles, blogs, and bios to personalize outreach emails">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={personDeep.value === "true"} onChange={(v) => personDeep.update(String(v))} />
            <SaveCheck visible={personDeep.saved} />
          </div>
        </SettingRow>
        {personDeep.value === "true" && (
          <>
            <div style={{ marginTop: 12, marginBottom: 4 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", marginBottom: 8 }}>
                Research sources — which platforms to check for each contact
              </div>
              <SourceCheckboxes value={personSources.value} onChange={personSources.update} saved={personSources.saved} />
            </div>
          </>
        )}
        <SettingRow label="Re-enrich after" description="Days before re-scraping a person's deep intel">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={personStaleness.value} min={1} max={30} onChange={personStaleness.update} unit="d" />
            <SaveCheck visible={personStaleness.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Max URLs per person" description="How many URLs to scrape per contact">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={personMaxUrls.value} min={1} max={20} onChange={personMaxUrls.update} />
            <SaveCheck visible={personMaxUrls.saved} />
          </div>
        </SettingRow>

        <div style={{ marginTop: 24 }}>
          <SectionTitle>COMPANY CACHE</SectionTitle>
          <SettingRow label="Enable cache" description="Use DB cache for previously enriched companies">
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <ToggleSwitch checked={cacheEnabled.value === "true"} onChange={(v) => cacheEnabled.update(String(v))} />
              <SaveCheck visible={cacheEnabled.saved} />
            </div>
          </SettingRow>
          <SettingRow label="Cache TTL" description="Days before re-enriching a cached company">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={cacheDays.value} min={1} max={30} onChange={cacheDays.update} unit="d" />
              <SaveCheck visible={cacheDays.saved} />
            </div>
          </SettingRow>
        </div>
      </div>
    </div>
  );
}

// ── Contacts Tab ────────────────────────────────────

function ContactsTab({ health }: { health: HealthResponse | null }) {
  const cfg = health?.config ?? {} as Record<string, unknown>;
  const maxContacts = useSetting("max_contacts_per_company", cfg.max_contacts_per_company, "8");
  const roleInference = useSetting("contact_role_inference", cfg.contact_role_inference, "llm");
  const dmRoles = useSetting("default_dm_roles", cfg.default_dm_roles, "CEO,CTO,CFO,COO,VP Operations,Founder");
  const infRoles = useSetting("default_influencer_roles", cfg.default_influencer_roles, "VP Engineering,VP Product,Head of Strategy,VP Marketing,VP Sales");

  return (
    <div style={{ maxWidth: 600 }}>
      <SectionTitle>CONTACT DISCOVERY</SectionTitle>
      <SettingRow label="Max contacts per company" description="Total contacts to find (mix of DMs and influencers)">
        <div style={{ display: "flex", alignItems: "center" }}>
          <SliderSetting value={maxContacts.value} min={1} max={20} onChange={maxContacts.update} />
          <SaveCheck visible={maxContacts.saved} />
        </div>
      </SettingRow>
      <SettingRow label="Role inference mode" description="How to determine target roles for each company">
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <select
            value={roleInference.value}
            onChange={(e) => roleInference.update(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
          >
            <option value="llm">LLM (adaptive)</option>
            <option value="default">Default roles</option>
            <option value="manual">Manual (from request)</option>
          </select>
          <SaveCheck visible={roleInference.saved} />
        </div>
      </SettingRow>

      <div style={{ marginTop: 24 }}>
        <SectionTitle>FALLBACK ROLES (used when LLM is unavailable or mode = Default)</SectionTitle>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)", marginBottom: 4 }}>Decision-makers</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>C-suite and senior leadership — the people who sign off on purchases</div>
          <TagEditor value={dmRoles.value} onChange={dmRoles.update} saved={dmRoles.saved} placeholder="e.g. VP Finance — press Enter to add" />
        </div>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)", marginBottom: 4 }}>Influencers</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>VP/Director level — people who evaluate and recommend solutions</div>
          <TagEditor value={infRoles.value} onChange={infRoles.update} saved={infRoles.saved} placeholder="e.g. Director of Engineering — press Enter to add" />
        </div>
      </div>

      <InfoNote>
        <strong>LLM mode</strong> (recommended) — AI analyzes each company&apos;s industry, size, and news to pick the best 6-8 roles automatically.{" "}
        <strong>Default mode</strong> — always uses the roles listed above.{" "}
        <strong>Manual mode</strong> — uses roles you specify per API request.
      </InfoNote>
    </div>
  );
}

// ── News & Email Tab ────────────────────────────────

function NewsEmailTab({ health }: { health: HealthResponse | null }) {
  const cfg = health?.config ?? {} as Record<string, unknown>;
  const lookbackDays = useSetting("news_lookback_days", cfg.news_lookback_days, "7");
  const maxArticles = useSetting("news_max_articles", cfg.news_max_articles, "50");
  const relevanceThreshold = useSetting("news_relevance_threshold", cfg.news_relevance_threshold, "0.5");
  const historicalEnabled = useSetting("historical_news_enabled", cfg.historical_news_enabled, "true");
  const historicalMonths = useSetting("historical_news_months", cfg.historical_news_months, "5");
  const personDepth = useSetting("email_personalization_depth", cfg.email_personalization_depth, "deep");
  const emailMaxLen = useSetting("email_max_length", cfg.email_max_length, "300");
  const emailSending = useSetting("email_sending_enabled", cfg.email_sending_enabled, "false");
  const emailTestMode = useSetting("email_test_mode", cfg.email_test_mode, "true");

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 48px" }}>
      <div>
        <SectionTitle>NEWS COLLECTION</SectionTitle>
        <SettingRow label="Fresh news window" description="Days of recent news to fetch via RSS + Tavily">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={lookbackDays.value} min={1} max={30} onChange={lookbackDays.update} unit="d" />
            <SaveCheck visible={lookbackDays.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Max articles per company" description="Total articles to collect across all sources">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={maxArticles.value} min={5} max={200} step={5} onChange={maxArticles.update} />
            <SaveCheck visible={maxArticles.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Relevance threshold" description="LLM relevance score cutoff for article filtering">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={relevanceThreshold.value} min={0} max={1} step={0.05} onChange={relevanceThreshold.update} />
            <SaveCheck visible={relevanceThreshold.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Historical news" description="Enable gnews library for 1-5 month historical articles">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={historicalEnabled.value === "true"} onChange={(v) => historicalEnabled.update(String(v))} />
            <SaveCheck visible={historicalEnabled.saved} />
          </div>
        </SettingRow>
        {historicalEnabled.value === "true" && (
          <SettingRow label="Historical months" description="How many months of history to fetch">
            <div style={{ display: "flex", alignItems: "center" }}>
              <SliderSetting value={historicalMonths.value} min={1} max={12} onChange={historicalMonths.update} unit="mo" />
              <SaveCheck visible={historicalMonths.saved} />
            </div>
          </SettingRow>
        )}
      </div>

      <div>
        <SectionTitle>EMAIL OUTREACH</SectionTitle>
        <SettingRow label="Personalization depth" description="How much research goes into each outreach email">
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <select
              value={personDepth.value}
              onChange={(e) => personDepth.update(e.target.value)}
              style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none", minWidth: 260 }}
            >
              <option value="basic">Basic — job title + company info only</option>
              <option value="deep">Deep — LinkedIn, blogs, articles, achievements</option>
            </select>
            <SaveCheck visible={personDepth.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Max email length" description="Approximate word limit for generated emails">
          <div style={{ display: "flex", alignItems: "center" }}>
            <SliderSetting value={emailMaxLen.value} min={50} max={1000} step={25} onChange={emailMaxLen.update} unit="w" />
            <SaveCheck visible={emailMaxLen.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Email sending" description="Enable real email sending via Brevo">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={emailSending.value === "true"} onChange={(v) => emailSending.update(String(v))} />
            <SaveCheck visible={emailSending.saved} />
          </div>
        </SettingRow>
        <SettingRow label="Test mode" description="Only send to test recipients (safety guard)">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <ToggleSwitch checked={emailTestMode.value === "true"} onChange={(v) => emailTestMode.update(String(v))} />
            <SaveCheck visible={emailTestMode.saved} />
          </div>
        </SettingRow>
      </div>
    </div>
  );
}

// ── Connections Tab ──────────────────────────────────

const POLL_INTERVAL_MS = 15_000;

function ConnectionsTab({ health, loading, onRefresh }: { health: HealthResponse | null; loading: boolean; onRefresh: () => void }) {
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [disabledProviders, setDisabledProviders] = useState<Set<string>>(() => getDisabledProviders());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setLastUpdated(new Date());
    intervalRef.current = setInterval(() => {
      onRefresh();
      setLastUpdated(new Date());
    }, POLL_INTERVAL_MS);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [onRefresh]);

  useEffect(() => {
    if (health) setLastUpdated(new Date());
  }, [health]);

  const toggleProvider = useCallback((name: string, enabled: boolean) => {
    setDisabledProviders((prev) => {
      const next = new Set(prev);
      if (enabled) next.delete(name); else next.add(name);
      saveDisabledProviders(next);
      return next;
    });
  }, []);

  const providers = useMemo(() => {
    if (!health) return [];
    return Object.entries(health.providers).map(([name, info]) => ({
      name, info, status: effectiveStatus(info),
    }));
  }, [health]);

  const onlineCount = providers.filter((p) => !disabledProviders.has(p.name) && (p.status === "online" || p.status === "ready")).length;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <SectionTitle>LLM PROVIDERS</SectionTitle>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {lastUpdated && <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>Updated {lastUpdated.toLocaleTimeString()}</span>}
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: loading ? "var(--amber)" : "var(--green)", animation: loading ? "pulse 1s ease-in-out infinite" : "none" }} title={loading ? "Refreshing..." : "Live"} />
        </div>
      </div>

      {loading && !health ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {[0, 1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 72, borderRadius: 10 }} />)}
        </div>
      ) : providers.length > 0 ? (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", background: onlineCount > 0 ? "var(--green-light)" : "var(--red-light)", borderRadius: 8, marginBottom: 14 }}>
            {onlineCount > 0 ? <Wifi size={14} style={{ color: "var(--green)" }} /> : <WifiOff size={14} style={{ color: "var(--red)" }} />}
            <span style={{ fontSize: 12, fontWeight: 600, color: onlineCount > 0 ? "var(--green)" : "var(--red)" }}>
              {onlineCount} of {providers.length} providers available
            </span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {providers.map(({ name, info, status }) => (
              <ProviderCard key={name} name={name} info={info} status={status} enabled={!disabledProviders.has(name)} onToggle={(enabled) => toggleProvider(name, enabled)} />
            ))}
          </div>
        </>
      ) : (
        <div style={{ padding: 20, textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
          Unable to reach backend.
        </div>
      )}

      <div style={{ marginTop: 28, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 48px" }}>
        <div>
          <SectionTitle>SEARCH SERVICES</SectionTitle>
          <ServiceRow icon={Search} name="Tavily" description="Primary AI search" status={health?.config?.tavily_enabled ? "connected" : "not configured"} />
          <ServiceRow icon={Search} name="DuckDuckGo" description="Fallback search" status={health?.config?.use_ddg_fallback !== false ? "active" : "disabled"} />
        </div>
        <div>
          <SectionTitle>CONTACT ENRICHMENT</SectionTitle>
          <ServiceRow icon={Mail} name="Apollo" description="B2B contacts" status="configured" />
          <ServiceRow icon={Mail} name="Hunter" description="Email verification" status="configured" />
        </div>
      </div>

      <div style={{ marginTop: 28 }}>
        <SectionTitle>EMBEDDINGS</SectionTitle>
        <div style={{ maxWidth: "50%" }}>
          <ServiceRow icon={Brain} name="NVIDIA Embeddings" description="Semantic similarity for dedup and clustering" status="configured" />
        </div>
      </div>
    </div>
  );
}

const PROVIDER_LABELS: Record<string, string> = {
  GeminiDirect: "Gemini (Google AI)",
  GeminiDirectLite: "Gemini Flash Lite",
  VertexLlama: "Llama 4 (Vertex AI)",
  VertexDeepSeek: "DeepSeek (Vertex AI)",
  Groq: "Groq Cloud",
  NVIDIA: "NVIDIA NIM",
  OpenRouter: "OpenRouter",
  Ollama: "Ollama (Local)",
};

function getDisabledProviders(): Set<string> {
  try {
    const raw = localStorage.getItem("harbinger_disabled_providers");
    return raw ? new Set(JSON.parse(raw)) : new Set();
  } catch { return new Set(); }
}

function saveDisabledProviders(disabled: Set<string>) {
  localStorage.setItem("harbinger_disabled_providers", JSON.stringify([...disabled]));
}

function ProviderCard({ name, info, status, enabled, onToggle }: {
  name: string; info: ProviderHealth; status: "online" | "ready" | "degraded" | "offline"; enabled: boolean; onToggle: (enabled: boolean) => void;
}) {
  const label = PROVIDER_LABELS[name] ?? name;
  let detail = "";
  if (!enabled) detail = "Disabled";
  else if (status === "online") detail = "No errors";
  else if (status === "ready") detail = `${info.failure_count} past errors — ready`;
  else if (status === "degraded" && info.backoff_until) {
    const remaining = Math.max(0, Math.round((new Date(info.backoff_until).getTime() - Date.now()) / 1000));
    detail = remaining > 60 ? `${info.failure_count} errors — ${Math.ceil(remaining / 60)}m cooldown` : `${info.failure_count} errors — ${remaining}s cooldown`;
  } else if (info.failure_count > 0) {
    detail = `${info.failure_count} errors`;
  }

  const eff = enabled ? status : "offline";
  return (
    <div className="card" style={{ padding: "14px 16px", display: "flex", alignItems: "flex-start", gap: 10, opacity: enabled ? 1 : 0.55, transition: "opacity 200ms" }}>
      <StatusDot status={eff} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{label}</span>
          <StatusBadge status={eff} />
        </div>
        {detail && <div style={{ fontSize: 11, color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}><Clock size={10} />{detail}</div>}
      </div>
      <ToggleSwitch checked={enabled} onChange={onToggle} />
    </div>
  );
}

function ServiceRow({ icon: Icon, name, description, status }: { icon: React.ElementType; name: string; description: string; status: string }) {
  const isOk = status === "configured" || status === "connected" || status === "active";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 0", borderBottom: "1px solid var(--border)" }}>
      <Icon size={14} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text)" }}>{name}</span>
          {isOk
            ? <span className="badge badge-green" style={{ fontSize: 9, display: "inline-flex", alignItems: "center", gap: 3 }}><CheckCircle size={9} />{status}</span>
            : <span className="badge badge-muted" style={{ fontSize: 9, display: "inline-flex", alignItems: "center", gap: 3 }}><AlertCircle size={9} />{status}</span>}
        </div>
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{description}</div>
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>("appearance");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    api.health()
      .then(setHealth)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return (
    <>
      <div style={{ padding: "16px 24px 0", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Settings
          </h1>
          <button
            onClick={refresh}
            disabled={loading}
            style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 5, padding: "5px 12px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", cursor: loading ? "not-allowed" : "pointer", fontSize: 12, color: "var(--text-secondary)", opacity: loading ? 0.6 : 1 }}
          >
            <RefreshCw size={12} style={{ animation: loading ? "spin 1s linear infinite" : "none" }} />
            Refresh
          </button>
        </div>

        <div style={{ display: "flex", gap: 0, overflowX: "auto" }}>
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              style={{
                padding: "8px 16px", background: "none", border: "none",
                borderBottom: activeTab === id ? "2px solid var(--accent)" : "2px solid transparent",
                cursor: "pointer", fontSize: 13, whiteSpace: "nowrap",
                fontWeight: activeTab === id ? 600 : 400,
                color: activeTab === id ? "var(--text)" : "var(--text-muted)",
                transition: "color 150ms, border-color 150ms",
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "24px" }}>
        {activeTab === "appearance" && <AppearanceTab />}
        {activeTab === "pipeline" && <PipelineTab health={health} />}
        {activeTab === "enrichment" && <EnrichmentTab health={health} />}
        {activeTab === "contacts" && <ContactsTab health={health} />}
        {activeTab === "news_email" && <NewsEmailTab health={health} />}
        {activeTab === "connections" && <ConnectionsTab health={health} loading={loading} onRefresh={refresh} />}
      </div>
    </>
  );
}
