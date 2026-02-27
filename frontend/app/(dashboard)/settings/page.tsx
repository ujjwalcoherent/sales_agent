"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  Sun, Moon, Zap, RefreshCw,
  CheckCircle, AlertCircle, XCircle,
  Wifi, WifiOff, Clock, Server,
  Search, Mail, Brain, Globe,
} from "lucide-react";
import { api } from "@/lib/api";
import type { HealthResponse, ProviderHealth } from "@/lib/types";

type Tab = "appearance" | "pipeline" | "connections";

const TABS: { id: Tab; label: string }[] = [
  { id: "appearance",  label: "Appearance"  },
  { id: "pipeline",    label: "Pipeline"    },
  { id: "connections", label: "Connections" },
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

/** Determine effective status: if cooldown has expired, provider is "ready" even with historical failures */
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

function ReadOnlyValue({ value }: { value: string }) {
  return (
    <span className="num" style={{ fontSize: 13, color: "var(--text-secondary)", background: "var(--surface-raised)", padding: "4px 10px", borderRadius: 6 }}>
      {value}
    </span>
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

// ── Appearance Tab (single column) ──────────────────

function AppearanceTab() {
  const [dark, setDark] = useState(false);
  const [density, setDensity] = useState(() => getPref("density", "comfortable"));
  const [terminalVerbosity, setTerminalVerbosity] = useState(() => getPref("terminal_verbosity", "standard"));
  const [refreshInterval, setRefreshInterval] = useState(() => getPref("refresh_interval", "off"));

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
      <SectionTitle>LAYOUT</SectionTitle>
      <SettingRow label="Interface density" description="Controls padding and spacing across all dashboard pages">
        <select
          value={density}
          onChange={(e) => { setDensity(e.target.value); setPref("density", e.target.value); }}
          style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
        >
          <option value="comfortable">Comfortable</option>
          <option value="compact">Compact</option>
        </select>
      </SettingRow>

      <div style={{ marginTop: 24 }} />
      <SectionTitle>TERMINAL</SectionTitle>
      <SettingRow label="Pipeline log detail" description="Controls how much detail the terminal shows during a run">
        <select
          value={terminalVerbosity}
          onChange={(e) => { setTerminalVerbosity(e.target.value); setPref("terminal_verbosity", e.target.value); }}
          style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
        >
          <option value="minimal">Minimal — steps only</option>
          <option value="standard">Standard — steps + metrics</option>
          <option value="verbose">Verbose — full technical detail</option>
        </select>
      </SettingRow>

      <div style={{ marginTop: 24 }} />
      <SectionTitle>DATA</SectionTitle>
      <SettingRow label="Dashboard auto-refresh" description="Automatically reload dashboard data at a regular interval">
        <select
          value={refreshInterval}
          onChange={(e) => { setRefreshInterval(e.target.value); setPref("refresh_interval", e.target.value); }}
          style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
        >
          <option value="off">Off</option>
          <option value="15">Every 15 seconds</option>
          <option value="30">Every 30 seconds</option>
          <option value="60">Every minute</option>
          <option value="300">Every 5 minutes</option>
        </select>
      </SettingRow>
    </div>
  );
}

// ── Pipeline Tab ────────────────────────────────────

function PipelineTab({ health }: { health: HealthResponse | null }) {
  const [country, setCountry] = useState(() => getPref("country", ""));
  const [maxTrends, setMaxTrends] = useState(() => getPref("max_trends", ""));

  useEffect(() => {
    if (!health) return;
    if (!country) setCountry(String(health.config.country ?? "India"));
    if (!maxTrends) setMaxTrends(String(health.config.max_trends ?? 8));
  }, [health, country, maxTrends]);

  const save = (key: string, value: string, setter: (v: string) => void) => {
    setter(value);
    setPref(key, value);
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 48px" }}>
      {/* Left column */}
      <div>
        <SectionTitle>TARGET MARKET</SectionTitle>
        <SettingRow label="Country" description="Primary market for news scanning and company discovery">
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Globe size={13} style={{ color: "var(--text-muted)" }} />
            <select
              value={country}
              onChange={(e) => save("country", e.target.value, setCountry)}
              style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none", minWidth: 160 }}
            >
              <option value="">Select country...</option>
              {ALL_COUNTRIES.map(({ code, name }) => (
                <option key={code} value={name}>{name}</option>
              ))}
            </select>
          </div>
        </SettingRow>

        <div style={{ marginTop: 24 }}>
          <SectionTitle>COLLECTION</SectionTitle>
          <SettingRow label="Max trends per run" description="How many market signals to extract from each pipeline run">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input
                type="range" min={3} max={20}
                value={maxTrends}
                onChange={(e) => save("max_trends", e.target.value, setMaxTrends)}
                style={{ width: 100, cursor: "pointer", accentColor: "var(--accent)" }}
              />
              <span className="num" style={{ fontSize: 14, color: "var(--text)", minWidth: 24, textAlign: "right" }}>
                {maxTrends}
              </span>
            </div>
          </SettingRow>

          <SettingRow label="RSS lookback window" description="How far back to scan for articles">
            <ReadOnlyValue value={`${health?.config?.rss_hours_ago ?? 72} hours`} />
          </SettingRow>

          <SettingRow label="Max articles per source" description="Cap per RSS feed to avoid noise from a single source">
            <ReadOnlyValue value={String(health?.config?.rss_max_per_source ?? 25)} />
          </SettingRow>
        </div>
      </div>

      {/* Right column */}
      <div>
        <SectionTitle>QUALITY GATES</SectionTitle>

        <SettingRow label="Cluster coherence minimum" description="Trends below this score are discarded as noise">
          <ReadOnlyValue value={String(health?.config?.coherence_min ?? 0.48)} />
        </SettingRow>

        <SettingRow label="Merge similarity threshold" description="Clusters more similar than this are merged into one trend">
          <ReadOnlyValue value={String(health?.config?.merge_threshold ?? 0.82)} />
        </SettingRow>

        <SettingRow label="Minimum synthesis confidence" description="Trends below this confidence are dropped before analysis">
          <ReadOnlyValue value={String(health?.config?.min_synthesis_confidence ?? 0.40)} />
        </SettingRow>

        <SettingRow label="Company relevance floor" description="Companies scoring below this are excluded from leads">
          <ReadOnlyValue value={String(health?.config?.company_min_relevance ?? 0.20)} />
        </SettingRow>

        <div style={{ marginTop: 24 }}>
          <SectionTitle>TIMEOUTS</SectionTitle>

          <SettingRow label="Trend analysis" description="Maximum time for trend synthesis per cluster">
            <ReadOnlyValue value={`${health?.config?.engine_synthesis_timeout ?? 180}s`} />
          </SettingRow>

          <SettingRow label="Causal council" description="Maximum time for the AI council to debate cross-trend impacts">
            <ReadOnlyValue value={`${health?.config?.engine_causal_timeout ?? 60}s`} />
          </SettingRow>
        </div>
      </div>

      <div style={{ gridColumn: "1 / -1" }}>
        <InfoNote>
          Country and max trends are saved locally and sent with each pipeline run. All other values are server-side defaults configured in <code style={{ fontSize: 10, background: "var(--surface-raised)", padding: "1px 5px", borderRadius: 4 }}>.env</code>. Adaptive thresholds may override manual values after sufficient run history.
        </InfoNote>
      </div>
    </div>
  );
}

// ── Connections Tab (auto-refreshing) ───────────────

const POLL_INTERVAL_MS = 15_000; // refresh provider health every 15 seconds

function ConnectionsTab({ health, loading, onRefresh }: { health: HealthResponse | null; loading: boolean; onRefresh: () => void }) {
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [disabledProviders, setDisabledProviders] = useState<Set<string>>(() => getDisabledProviders());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-refresh every 15s while this tab is visible
  useEffect(() => {
    setLastUpdated(new Date());
    intervalRef.current = setInterval(() => {
      onRefresh();
      setLastUpdated(new Date());
    }, POLL_INTERVAL_MS);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [onRefresh]);

  // Update timestamp when health data arrives
  useEffect(() => {
    if (health) setLastUpdated(new Date());
  }, [health]);

  const toggleProvider = useCallback((name: string, enabled: boolean) => {
    setDisabledProviders((prev) => {
      const next = new Set(prev);
      if (enabled) {
        next.delete(name);
      } else {
        next.add(name);
      }
      saveDisabledProviders(next);
      return next;
    });
  }, []);

  const providers = useMemo(() => {
    if (!health) return [];
    return Object.entries(health.providers).map(([name, info]) => ({
      name,
      info,
      status: effectiveStatus(info),
    }));
  }, [health]);

  const onlineCount = providers.filter((p) => !disabledProviders.has(p.name) && (p.status === "online" || p.status === "ready")).length;

  return (
    <div>
      {/* LLM Providers */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <SectionTitle>LLM PROVIDERS</SectionTitle>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {lastUpdated && (
            <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <div
            style={{
              width: 6, height: 6, borderRadius: "50%",
              background: loading ? "var(--amber)" : "var(--green)",
              animation: loading ? "pulse 1s ease-in-out infinite" : "none",
            }}
            title={loading ? "Refreshing..." : "Live — refreshes every 15s"}
          />
        </div>
      </div>

      {loading && !health ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {[0, 1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 72, borderRadius: 10 }} />)}
        </div>
      ) : providers.length > 0 ? (
        <>
          {/* Summary bar */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", background: onlineCount > 0 ? "var(--green-light)" : "var(--red-light)", borderRadius: 8, marginBottom: 14 }}>
            {onlineCount > 0 ? <Wifi size={14} style={{ color: "var(--green)" }} /> : <WifiOff size={14} style={{ color: "var(--red)" }} />}
            <span style={{ fontSize: 12, fontWeight: 600, color: onlineCount > 0 ? "var(--green)" : "var(--red)" }}>
              {onlineCount} of {providers.length} providers available
            </span>
            <span style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: 4 }}>
              — providers reset automatically at the start of each pipeline run
            </span>
          </div>

          {/* Provider grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {providers.map(({ name, info, status }) => (
              <ProviderCard
                key={name}
                name={name}
                info={info}
                status={status}
                enabled={!disabledProviders.has(name)}
                onToggle={(enabled) => toggleProvider(name, enabled)}
              />
            ))}
          </div>
        </>
      ) : (
        <div style={{ padding: 20, textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
          Unable to reach backend — check that the API server is running.
        </div>
      )}

      {/* Search & Data */}
      <div style={{ marginTop: 28, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 48px" }}>
        <div>
          <SectionTitle>SEARCH SERVICES</SectionTitle>
          <ServiceRow icon={Search} name="SearXNG" description="Self-hosted meta-search for trend sourcing" status={health?.config?.searxng_enabled ? "connected" : "not configured"} />
          <ServiceRow icon={Search} name="DuckDuckGo" description="Fallback web search for company discovery" status={health?.config?.use_ddg_fallback !== false ? "active" : "disabled"} />
        </div>

        <div>
          <SectionTitle>CONTACT ENRICHMENT</SectionTitle>
          <ServiceRow icon={Mail} name="Apollo" description="B2B contact finder and company data enrichment" status="configured" />
          <ServiceRow icon={Mail} name="Hunter" description="Email verification and domain-level contact discovery" status="configured" />
        </div>
      </div>

      <div style={{ marginTop: 28 }}>
        <SectionTitle>EMBEDDINGS</SectionTitle>
        <div style={{ maxWidth: "50%" }}>
          <ServiceRow icon={Brain} name="NVIDIA Embeddings" description="Semantic similarity for article dedup and trend clustering" status="configured" />
        </div>
      </div>

      <InfoNote>
        Provider health refreshes automatically every 15 seconds. API keys and connection URLs are stored server-side in <code style={{ fontSize: 10, background: "var(--surface-raised)", padding: "1px 5px", borderRadius: 4 }}>.env</code> and are never sent to the browser. Providers with expired cooldowns show as &quot;Ready&quot; — they will reconnect on the next pipeline run.
      </InfoNote>
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

/** Toggle switch for enabling/disabling providers */
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

/** Read disabled providers from localStorage */
function getDisabledProviders(): Set<string> {
  try {
    const raw = localStorage.getItem("harbinger_disabled_providers");
    return raw ? new Set(JSON.parse(raw)) : new Set();
  } catch { return new Set(); }
}

/** Save disabled providers to localStorage */
function saveDisabledProviders(disabled: Set<string>) {
  localStorage.setItem("harbinger_disabled_providers", JSON.stringify([...disabled]));
}

function ProviderCard({ name, info, status, enabled, onToggle }: {
  name: string;
  info: ProviderHealth;
  status: "online" | "ready" | "degraded" | "offline";
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}) {
  const label = PROVIDER_LABELS[name] ?? name;

  let detail = "";
  if (!enabled) {
    detail = "Disabled — will be skipped in pipeline runs";
  } else if (status === "online") {
    detail = "No errors";
  } else if (status === "ready") {
    detail = `${info.failure_count} past errors — cooldown expired, ready to retry`;
  } else if (status === "degraded" && info.backoff_until) {
    const remaining = Math.max(0, Math.round((new Date(info.backoff_until).getTime() - Date.now()) / 1000));
    if (remaining > 60) {
      detail = `${info.failure_count} errors — cooldown ${Math.ceil(remaining / 60)}m remaining`;
    } else if (remaining > 0) {
      detail = `${info.failure_count} errors — cooldown ${remaining}s remaining`;
    } else {
      detail = `${info.failure_count} errors — cooldown expired`;
    }
  } else if (info.failure_count > 0 && info.last_failure_time) {
    const ago = Math.round((Date.now() - new Date(info.last_failure_time).getTime()) / 60000);
    detail = ago < 60
      ? `${info.failure_count} errors — last ${ago}m ago`
      : `${info.failure_count} errors — last ${Math.round(ago / 60)}h ago`;
  }

  const effectiveStatus = enabled ? status : "offline";

  return (
    <div className="card" style={{
      padding: "14px 16px", display: "flex", alignItems: "flex-start", gap: 10,
      opacity: enabled ? 1 : 0.55, transition: "opacity 200ms",
    }}>
      <StatusDot status={effectiveStatus} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{label}</span>
          <StatusBadge status={effectiveStatus} />
        </div>
        {detail && (
          <div style={{ fontSize: 11, color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
            <Clock size={10} />
            {detail}
          </div>
        )}
      </div>
      <ToggleSwitch checked={enabled} onChange={onToggle} />
    </div>
  );
}

function ServiceRow({ icon: Icon, name, description, status }: {
  icon: React.ElementType;
  name: string;
  description: string;
  status: string;
}) {
  const isOk = status === "configured" || status === "connected" || status === "active";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 0", borderBottom: "1px solid var(--border)" }}>
      <Icon size={14} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text)" }}>{name}</span>
          {isOk
            ? <span className="badge badge-green" style={{ fontSize: 9, display: "inline-flex", alignItems: "center", gap: 3 }}><CheckCircle size={9} />{status}</span>
            : <span className="badge badge-muted" style={{ fontSize: 9, display: "inline-flex", alignItems: "center", gap: 3 }}><AlertCircle size={9} />{status}</span>
          }
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
      {/* Header */}
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

        <div style={{ display: "flex", gap: 0 }}>
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              style={{
                padding: "8px 16px", background: "none", border: "none",
                borderBottom: activeTab === id ? "2px solid var(--accent)" : "2px solid transparent",
                cursor: "pointer", fontSize: 13,
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

      {/* Tab body — full width */}
      <div style={{ flex: 1, overflow: "auto", padding: "24px" }}>
        {activeTab === "appearance" && <AppearanceTab />}
        {activeTab === "pipeline" && <PipelineTab health={health} />}
        {activeTab === "connections" && <ConnectionsTab health={health} loading={loading} onRefresh={refresh} />}
      </div>
    </>
  );
}
