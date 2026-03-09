"use client";

import { useState, useEffect, useRef, use } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft, Globe, ExternalLink, Newspaper, Users,
  Mail, User, Loader2, Zap, Check, RefreshCw, Send,
  Linkedin, FileText, MapPin, Calendar, TrendingUp, ChevronRight, ChevronDown,
} from "lucide-react";
import Link from "next/link";
import { CompanyLogo } from "@/components/ui/company-logo";
import { PageLoader, PageError } from "@/components/ui/page-states";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import { confidenceColor, formatDate } from "@/lib/utils";
import type { SavedCompany, PersonRecord, LeadRecord, GenerateLeadsResponse, CompanyNewsArticle } from "@/lib/types";

type Tab = "overview" | "news" | "leads";

export default function CompanyDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: rawId } = use(params);
  const router = useRouter();
  const [company, setCompany] = useState<SavedCompany | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [genLoading, setGenLoading] = useState(false);
  const [genStatus, setGenStatus] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  // Resolved company ID (may differ from rawId if we did a name-based search)
  const [resolvedId, setResolvedId] = useState(rawId);

  const autoEnrichTriggered = useRef(false);

  // Must call hooks before any conditional returns
  const { leads: allPipelineLeads } = usePipelineContext();

  useEffect(() => { loadCompany(); }, [rawId]);

  async function loadCompany() {
    setLoading(true);
    setError(null);

    // Decode — rawId could be a hash ID or a URL-encoded company name
    const decoded = decodeURIComponent(rawId);
    const isHashId = /^[a-f0-9]{8,16}$/.test(decoded);

    try {
      // First, try direct lookup by ID
      const data = await api.getSavedCompany(decoded);
      setCompany(data);
      setResolvedId(data.id);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "";
      if (isHashId) {
        // Distinguish "not found" (404) from "can't connect" (network error)
        if (msg.includes("Failed to fetch") || msg.includes("fetch")) {
          setError("Cannot reach the backend server. Make sure the API is running on port 8000.");
        } else {
          setError("Company not found");
        }
        setLoading(false);
        return;
      }
      // Not a hash ID — treat as company name and auto-search to save it
      try {
        const searchRes = await api.searchCompanies(decoded);
        if (searchRes.companies.length > 0) {
          const match = searchRes.companies[0];
          // Now load the saved company (search auto-persists)
          try {
            const data = await api.getSavedCompany(match.id);
            setCompany(data);
            setResolvedId(match.id);
          } catch {
            // Search saved it but we can still construct from search result
            setCompany({
              id: match.id,
              company_name: match.company_name,
              domain: match.domain,
              website: match.website,
              industry: match.industry,
              description: match.description || "",
              headquarters: match.headquarters || "",
              employee_count: match.employee_count || "",
              founded_year: match.founded_year || null,
              stock_ticker: match.stock_ticker || "",
              ceo: match.ceo || "",
              funding_stage: match.funding_stage || "",
              wikidata_id: "",
              reason_relevant: match.reason_relevant,
              article_count: match.article_count,
              recent_articles: match.recent_articles,
              live_news: match.live_news,
              search_query: decoded,
              search_type: "company",
              contacts: [],
              contacts_reasoning: "",
              contacts_generated_at: null,
              last_searched_at: new Date().toISOString(),
              created_at: new Date().toISOString(),
            });
            setResolvedId(match.id);
          }
        } else {
          setError(`No results found for "${decoded}"`);
        }
      } catch (searchErr) {
        setError(`Could not find company "${decoded}"`);
      }
    }
    setLoading(false);
  }

  // Auto-enrich: trigger refresh if metadata or news/intelligence data is missing
  useEffect(() => {
    if (!company || autoEnrichTriggered.current || refreshing) return;
    const hasBasicData = !!(company.description || company.headquarters || company.employee_count || company.ceo);
    const hasIntelData = (company.live_news?.length ?? 0) > 0
      || (company.article_count ?? 0) > 0
      || (company.recent_articles?.length ?? 0) > 0;
    // Refresh if missing metadata OR missing news/intel
    // (campaign-added companies have metadata but skip the news/article fetch)
    if ((!hasBasicData || !hasIntelData) && company.company_name) {
      autoEnrichTriggered.current = true;
      handleRefreshSearch();
    }
  }, [company]);

  async function handleGenLeads() {
    if (!company) return;
    setGenLoading(true);
    setError(null);
    setGenStatus("Finding contacts & generating outreach...");
    try {
      const genResult = await api.generateLeads(company.id, company.company_name, company.domain);
      setGenStatus("");

      // Try reloading from DB first (contacts persisted server-side)
      try {
        const updated = await api.getSavedCompany(resolvedId);
        // If DB reload returned 0 contacts but the response had some, use response directly
        if ((updated.contacts?.length ?? 0) === 0 && (genResult.contacts?.length ?? 0) > 0) {
          setCompany({ ...updated, contacts: genResult.contacts });
        } else {
          setCompany(updated);
        }
      } catch {
        // DB reload failed — fall back to contacts from the response
        if ((genResult.contacts?.length ?? 0) > 0) {
          setCompany(prev => prev ? { ...prev, contacts: genResult.contacts } : prev);
        }
      }
      setActiveTab("leads");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Lead generation failed";
      // Distinguish network errors from backend errors
      if (msg === "Failed to fetch" || msg.startsWith("NetworkError")) {
        setError("Cannot reach the backend server. Make sure the API is running on port 8000.");
      } else if (msg.includes("timed out")) {
        setError(msg); // timeout message already clear
      } else {
        // Show actual API error (e.g. "API 500: ...")
        setError(`Lead gen failed: ${msg.slice(0, 120)}`);
      }
      // Even on error, try to reload — backend may have saved contacts before the error
      try {
        const updated = await api.getSavedCompany(resolvedId);
        if ((updated.contacts?.length ?? 0) > 0) { setCompany(updated); setActiveTab("leads"); }
      } catch { /* ignore reload failure */ }
    }
    setGenLoading(false);
    setGenStatus("");
  }

  async function handleRefreshSearch() {
    if (!company) return;
    setRefreshing(true);
    try {
      // Re-search to refresh news and articles
      await api.searchCompanies(company.company_name);
      const updated = await api.getSavedCompany(resolvedId);
      setCompany(updated);
    } catch {
      // silent fail on refresh
    }
    setRefreshing(false);
  }

  if (loading) return <PageLoader message="Loading company..." />;
  if (error || !company) return <PageError message={error || "Company not found"} onBack={() => router.push("/companies")} backLabel="Back to Companies" />;

  const pipelineLeadsForCompany = allPipelineLeads.filter(
    (l) => l.company_name.toLowerCase() === company.company_name.toLowerCase()
  );
  const hasNews = (company.live_news?.length ?? 0) > 0;
  const hasArticles = (company.recent_articles?.length ?? 0) > 0;
  const hasContacts = (company.contacts?.length ?? 0) > 0;
  const totalSignals = (company.live_news?.length ?? 0) + (company.recent_articles?.length ?? 0);
  const totalPeople = (company.contacts?.length ?? 0) + pipelineLeadsForCompany.length;

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "overview", label: "Overview" },
    { id: "news", label: "News & Intel", count: totalSignals },
    { id: "leads", label: "People & Leads", count: totalPeople },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Header */}
      <div style={{ padding: "16px 24px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        {/* Back + Company info */}
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
          <button onClick={() => router.push("/companies")} style={{ width: 32, height: 32, borderRadius: 8, border: "1px solid var(--border)", background: "var(--surface)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", flexShrink: 0 }}>
            <ArrowLeft size={14} />
          </button>
          <CompanyLogo domain={company.domain} size={48} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1 className="font-display" style={{ fontSize: 22, color: "var(--text)", letterSpacing: "-0.02em", marginBottom: 3 }}>
              {company.company_name}
            </h1>
            <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, color: "var(--text-muted)", flexWrap: "wrap" }}>
              {company.stock_ticker && (
                <span style={{ fontSize: 10, fontWeight: 700, color: "var(--green)", background: "var(--green-light)", padding: "1px 6px", borderRadius: 4 }}>
                  {company.stock_ticker}
                </span>
              )}
              {company.domain && (
                <a href={`https://${company.domain}`} target="_blank" rel="noopener noreferrer" style={{ color: "var(--blue)", display: "flex", alignItems: "center", gap: 4, textDecoration: "none" }}>
                  <Globe size={12} /> {company.domain}
                </a>
              )}
              {company.industry && <span>{company.industry}</span>}
              {company.headquarters && (
                <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                  <MapPin size={11} /> {company.headquarters}
                </span>
              )}
              {company.last_searched_at && (
                <span style={{ color: "var(--text-xmuted)", fontSize: 10 }}>
                  Searched {formatDate(company.last_searched_at, "medium")}
                </span>
              )}
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
            <button
              onClick={handleRefreshSearch}
              disabled={refreshing}
              style={{ display: "flex", alignItems: "center", gap: 4, padding: "6px 12px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 11, fontWeight: 500, color: "var(--text-secondary)", cursor: refreshing ? "not-allowed" : "pointer" }}
            >
              <RefreshCw size={12} style={refreshing ? { animation: "spin 1s linear infinite" } : undefined} />
              Refresh
            </button>
            <button
              onClick={handleGenLeads}
              disabled={genLoading}
              style={{ display: "flex", alignItems: "center", gap: 4, padding: "6px 14px", borderRadius: 7, border: "1px solid var(--accent)", background: genLoading ? "var(--surface)" : "var(--accent)", fontSize: 11, fontWeight: 600, color: genLoading ? "var(--text-secondary)" : "#fff", cursor: genLoading ? "not-allowed" : "pointer" }}
            >
              {genLoading ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> : <Zap size={12} />}
              {hasContacts ? "Refresh Leads" : "Generate Leads"}
            </button>
          </div>
        </div>

        {/* Lead gen progress banner */}
        {genLoading && genStatus && (
          <div style={{
            display: "flex", alignItems: "center", gap: 8, padding: "8px 16px",
            background: "var(--accent-light)", borderRadius: 8, fontSize: 12,
            color: "var(--accent)", fontWeight: 500, marginTop: 8,
          }}>
            <Loader2 size={13} style={{ animation: "spin 1s linear infinite" }} />
            {genStatus}
            <span style={{ color: "var(--text-muted)", fontWeight: 400, marginLeft: "auto", fontSize: 11 }}>
              This typically takes 1-3 minutes
            </span>
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: "flex", gap: 0, borderBottom: "none" }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: "8px 16px", fontSize: 12, fontWeight: 500, border: "none", cursor: "pointer",
                background: "transparent",
                color: activeTab === tab.id ? "var(--accent)" : "var(--text-muted)",
                borderBottom: activeTab === tab.id ? "2px solid var(--accent)" : "2px solid transparent",
                transition: "color 150ms, border-color 150ms",
                display: "flex", alignItems: "center", gap: 5,
              }}
            >
              {tab.label}
              {tab.count !== undefined && tab.count > 0 && (
                <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 10, background: activeTab === tab.id ? "var(--accent-light)" : "var(--surface-raised)", color: activeTab === tab.id ? "var(--accent)" : "var(--text-muted)" }}>
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content — always mounted to preserve state (avoid re-fetching on tab switch) */}
      <div style={{ flex: 1, overflow: "auto", padding: "20px 24px" }}>
        <div style={{ display: activeTab === "overview" ? "block" : "none" }}><OverviewTab company={company} /></div>
        <div style={{ display: activeTab === "news" ? "block" : "none" }}><NewsTab company={company} /></div>
        <div style={{ display: activeTab === "leads" ? "block" : "none" }}><LeadsTab company={company} genLoading={genLoading} onGenLeads={handleGenLeads} /></div>
      </div>
    </div>
  );
}

function OverviewTab({ company }: { company: SavedCompany }) {
  const [descExpanded, setDescExpanded] = useState(false);
  const hasNews = (company.live_news?.length ?? 0) > 0;
  const hasContacts = (company.contacts?.length ?? 0) > 0;
  const desc = company.description || company.reason_relevant || "";
  const descTruncLimit = 320;
  const descTruncated = desc.length > descTruncLimit && !descExpanded;

  const metaFields = [
    company.industry && { label: "Industry", value: company.industry, icon: TrendingUp as typeof MapPin },
    company.headquarters && { label: "Headquarters", value: company.headquarters, icon: MapPin as typeof MapPin },
    company.ceo && { label: "CEO", value: company.ceo, icon: User as typeof MapPin },
    company.stock_ticker && { label: "Ticker", value: company.stock_ticker, icon: TrendingUp as typeof MapPin, ticker: true },
    company.funding_stage && { label: "Stage", value: company.funding_stage, icon: null },
    company.domain && { label: "Website", value: company.domain, icon: Globe as typeof MapPin, link: true },
  ].filter(Boolean) as { label: string; value: string; icon: typeof MapPin | null; ticker?: boolean; link?: boolean }[];

  const hasEnrichment = (company.sub_industries?.length ?? 0) + (company.products_services?.length ?? 0)
    + (company.tech_stack?.length ?? 0) + (company.competitors?.length ?? 0)
    + (company.investors?.length ?? 0) + (company.revenue ? 1 : 0) + (company.total_funding ? 1 : 0)
    + (company.employee_count ? 1 : 0) + (company.founded_year ? 1 : 0) > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

      {/* About + Enrichment — side by side */}
      <div style={{ display: "grid", gridTemplateColumns: hasEnrichment ? "1fr 1fr" : "1fr", gap: 14 }}>
        {/* About */}
        <div className="card" style={{ padding: 0 }}>
          <div style={{ padding: "11px 16px 9px", borderBottom: "1px solid var(--border)" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              About {company.company_name}
            </div>
          </div>
          <div style={{ padding: "14px 16px" }}>
            {desc && (
              <div style={{ marginBottom: metaFields.length > 0 ? 14 : 0 }}>
                {!company.description && company.reason_relevant && (
                  <div style={{ fontSize: 9, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.04em", marginBottom: 4 }}>WHY RELEVANT</div>
                )}
                <p style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.75, margin: 0 }}>
                  {descTruncated ? desc.slice(0, descTruncLimit) + "…" : desc}
                </p>
                {desc.length > descTruncLimit && (
                  <button onClick={() => setDescExpanded(v => !v)} style={{ marginTop: 6, fontSize: 11, color: "var(--accent)", background: "none", border: "none", cursor: "pointer", padding: 0 }}>
                    {descExpanded ? "Show less" : "Read more →"}
                  </button>
                )}
              </div>
            )}
            {metaFields.length > 0 && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 16px" }}>
                {metaFields.map(f => (
                  <div key={f.label}>
                    <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.05em", marginBottom: 3 }}>{f.label.toUpperCase()}</div>
                    <div style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
                      {f.icon && <f.icon size={11} style={{ color: "var(--text-muted)", flexShrink: 0 }} />}
                      {f.link ? (
                        <a href={`https://${f.value}`} target="_blank" rel="noopener noreferrer" style={{ color: "var(--blue)", textDecoration: "none" }}>{f.value}</a>
                      ) : f.ticker ? (
                        <span style={{ color: "var(--green)", fontWeight: 700, padding: "1px 5px", background: "var(--green-light)", borderRadius: 4 }}>{f.value}</span>
                      ) : f.value}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!desc && metaFields.length === 0 && (
              <div style={{ color: "var(--text-muted)", fontSize: 12, textAlign: "center", padding: "12px 0" }}>
                No data yet — click <strong>Refresh</strong> to enrich.
              </div>
            )}
          </div>
        </div>

        {/* Enrichment Data */}
        {hasEnrichment && <OverviewEnrichment company={company} inline />}
      </div>

      {/* Key People — full-width grid that fills available horizontal space */}
      {((company.key_people?.length ?? 0) > 0) && (() => {
        const people = (company.key_people ?? []).filter(p => (typeof p === "string" ? p : p.name));
        if (people.length === 0) return null;
        return (
          <div className="card" style={{ padding: 0 }}>
            <div style={{ padding: "11px 16px 9px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "flex", alignItems: "center", gap: 5 }}>
                <Users size={11} /> Key People ({people.length})
              </div>
            </div>
            <div style={{ padding: "12px 16px", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))", gap: 8 }}>
              {people.map((p, i) => {
                const name = typeof p === "string" ? p : p.name;
                const role = typeof p === "string" ? "" : (p.role || (p as { title?: string }).title || "");
                const linkedin = typeof p === "string" ? "" : (p.linkedin_url || "");
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 12px", background: "var(--surface-raised)", borderRadius: 8, border: "1px solid var(--border)" }}>
                    <div style={{ width: 32, height: 32, borderRadius: "50%", background: "var(--accent-light)", color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, flexShrink: 0 }}>
                      {name[0]?.toUpperCase() ?? "?"}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</div>
                      {role && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{role}</div>}
                    </div>
                    {linkedin && (
                      <a href={linkedin} target="_blank" rel="noopener noreferrer" style={{ flexShrink: 0, color: "var(--blue)", opacity: 0.7 }} title="LinkedIn">
                        <Linkedin size={13} />
                      </a>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}

      {/* Live news */}
      {hasNews && (
        <div className="card" style={{ padding: 0 }}>
          <div style={{ padding: "11px 16px 9px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              <Newspaper size={11} /> Live News
            </div>
            <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{company.live_news!.length} articles · <button onClick={() => {}} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--accent)", fontSize: 10, padding: 0 }}>View all in News tab</button></span>
          </div>
          <div style={{ padding: "8px 8px" }}>
            {company.live_news!.slice(0, 4).map((n, i) => (
              <a key={i} href={n.url} target="_blank" rel="noopener noreferrer"
                style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "9px 10px", borderRadius: 7, textDecoration: "none", transition: "background 100ms" }}
                onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-raised)")}
                onMouseLeave={e => (e.currentTarget.style.background = "")}
              >
                <ExternalLink size={10} style={{ color: "var(--text-xmuted)", flexShrink: 0, marginTop: 2 }} />
                <span style={{ fontSize: 12, color: "var(--text)", lineHeight: 1.4, flex: 1 }}>{n.title}</span>
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Pipeline Articles — richer than live_news (has summary, source, date) */}
      {(company.recent_articles?.length ?? 0) > 0 && (
        <div className="card" style={{ padding: 0 }}>
          <div style={{ padding: "11px 16px 9px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              <FileText size={11} /> Pipeline Articles
            </div>
            <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>{company.recent_articles!.length} from last run</span>
          </div>
          <div style={{ padding: "8px 8px" }}>
            {company.recent_articles!.slice(0, 3).map((a, i) => (
              <a key={i} href={a.url} target="_blank" rel="noopener noreferrer"
                style={{ display: "flex", gap: 10, padding: "10px 10px", borderRadius: 7, textDecoration: "none", transition: "background 100ms" }}
                onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-raised)")}
                onMouseLeave={e => (e.currentTarget.style.background = "")}
              >
                <ExternalLink size={10} style={{ color: "var(--text-xmuted)", flexShrink: 0, marginTop: 3 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, lineHeight: 1.4, marginBottom: 3 }}>{a.title}</div>
                  {a.summary && <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{a.summary}</div>}
                  <div style={{ display: "flex", gap: 6, marginTop: 4, fontSize: 10, color: "var(--text-xmuted)" }}>
                    {a.source_name && <span>{a.source_name}</span>}
                    {a.published_at && <span>· {formatDate(a.published_at, "short")}</span>}
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Intelligence stats */}
      <div className="card" style={{ padding: 0 }}>
        <div style={{ padding: "11px 16px 9px", borderBottom: "1px solid var(--border)" }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Data Intelligence</div>
        </div>
        <div style={{ padding: "12px 16px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 10 }}>
            <IntelStat label="Live News" value={company.live_news?.length ?? 0} color="var(--blue)" />
            <IntelStat label="Pipeline Articles" value={company.recent_articles?.length ?? 0} color="var(--accent)" />
            <IntelStat label="Cached Articles" value={company.article_count} color="var(--accent)" />
            <IntelStat label="Contacts" value={company.contacts?.length ?? 0} color="var(--green)" />
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
            {company.wikidata_id && (
              <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "4px 8px", background: "var(--green-light)", borderRadius: 5, border: "1px solid var(--green)33" }}>
                <Check size={10} style={{ color: "var(--green)" }} />
                <span style={{ fontSize: 10, color: "var(--green)", fontWeight: 600 }}>Verified Entity</span>
              </div>
            )}
            {company.contacts_generated_at && (
              <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>Contacts enriched {formatDate(company.contacts_generated_at, "medium")}</span>
            )}
            {company.validation_source && (
              <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>via {company.validation_source}</span>
            )}
          </div>
          {company.contacts_reasoning && (
            <div style={{ marginTop: 10, fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.6, padding: "8px 12px", background: "var(--surface-raised)", borderRadius: 7 }}>
              <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>Why these contacts: </span>
              {company.contacts_reasoning}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function IntelStat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ padding: "10px 14px", borderRadius: 8, background: "var(--surface-raised)", border: "1px solid var(--border)", textAlign: "center" }}>
      <div className="num" style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3 }}>{label}</div>
    </div>
  );
}

function OverviewEnrichment({ company, inline }: { company: SavedCompany; inline?: boolean }) {
  const sections: { label: string; items?: string[]; color?: string }[] = [
    { label: "Sub-industries", items: company.sub_industries },
    { label: "Products & Services", items: company.products_services },
    { label: "Competitors", items: company.competitors },
    { label: "Investors", items: company.investors, color: "var(--green)" },
    { label: "Tech Stack", items: company.tech_stack, color: "var(--blue)" },
  ].filter((s) => s.items && s.items.length > 0);

  const hasFinancials = !!(company.revenue || company.total_funding || company.employee_count || company.founded_year);
  // Key people now rendered as a full-width section in OverviewTab — skip guard check
  if (sections.length === 0 && !hasFinancials) return null;

  return (
    <div className="card" style={{ padding: 0, ...(inline ? {} : { gridColumn: "1 / -1" }) }}>
      <div style={{ padding: "11px 16px 9px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          Enrichment Data
        </div>
      </div>
      <div style={{ padding: "14px 16px" }}>
        {/* Key metrics row */}
        {hasFinancials && (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: sections.length > 0 ? 14 : 0 }}>
            {company.employee_count && (
              <div style={{ padding: "7px 12px", background: "var(--amber-light)", borderRadius: 7, border: "1px solid var(--amber)33" }}>
                <div style={{ fontSize: 9, color: "var(--amber)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>EMPLOYEES</div>
                <div className="num" style={{ fontSize: 13, color: "var(--amber)", fontWeight: 700 }}>{company.employee_count}</div>
              </div>
            )}
            {company.founded_year && (
              <div style={{ padding: "7px 12px", background: "var(--accent-light)", borderRadius: 7, border: "1px solid var(--accent)33" }}>
                <div style={{ fontSize: 9, color: "var(--accent)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>FOUNDED</div>
                <div className="num" style={{ fontSize: 13, color: "var(--accent)", fontWeight: 700 }}>{company.founded_year}</div>
              </div>
            )}
            {company.revenue && (
              <div style={{ padding: "7px 12px", background: "var(--surface-raised)", borderRadius: 7, border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>REVENUE</div>
                <div className="num" style={{ fontSize: 13, color: "var(--text)", fontWeight: 700 }}>{company.revenue}</div>
              </div>
            )}
            {company.total_funding && (
              <div style={{ padding: "7px 12px", background: "var(--surface-raised)", borderRadius: 7, border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>TOTAL FUNDING</div>
                <div className="num" style={{ fontSize: 13, color: "var(--green)", fontWeight: 700 }}>{company.total_funding}</div>
              </div>
            )}
          </div>
        )}

        {/* Tag groups */}
        <div style={{ display: "grid", gridTemplateColumns: sections.length > 2 ? "1fr 1fr" : "1fr", gap: "12px 24px" }}>
          {sections.map((s) => (
            <div key={s.label}>
              <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 5 }}>{s.label.toUpperCase()}</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {s.items!.map((item) => (
                  <span
                    key={item}
                    style={{
                      fontSize: 10, padding: "2px 7px", borderRadius: 4,
                      background: "var(--surface-raised)", color: s.color ?? "var(--text-secondary)",
                      border: "1px solid var(--border)",
                    }}
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Key People — removed, now rendered as its own full-width section in OverviewTab */}
        {false && keyPeople.length > 0 && (
          <div style={{ marginTop: sections.length > 0 || hasFinancials ? 14 : 0 }}>
            <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 8 }}>KEY PEOPLE</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {keyPeople.map((p, i) => {
                const name = typeof p === "string" ? p : p.name;
                const role = typeof p === "string" ? "" : (p.role || p.title || "");
                const linkedin = typeof p === "string" ? "" : (p.linkedin_url || "");
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 10px", background: "var(--surface-raised)", borderRadius: 7, border: "1px solid var(--border)" }}>
                    <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--accent-light)", color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, flexShrink: 0 }}>
                      {name[0]?.toUpperCase() ?? "?"}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{name}</div>
                      {role && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>{role}</div>}
                    </div>
                    {linkedin && (
                      <a href={linkedin} target="_blank" rel="noopener noreferrer" style={{ flexShrink: 0, color: "var(--blue)", opacity: 0.7 }} title="LinkedIn">
                        <Linkedin size={13} />
                      </a>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {company.validation_source && (
          <div style={{ marginTop: 10, fontSize: 9, color: "var(--text-xmuted)" }}>
            Source: {company.validation_source}
          </div>
        )}
      </div>
    </div>
  );
}

function NewsTab({ company }: { company: SavedCompany }) {
  const [articles, setArticles] = useState<CompanyNewsArticle[]>([]);
  const [totalArticles, setTotalArticles] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [collecting, setCollecting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const perPage = 20;

  useEffect(() => {
    loadNews(1);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [company.id]);

  async function loadNews(p: number) {
    setLoading(true);
    try {
      const res = await api.getCompanyNews(company.id, p, perPage);
      if (p === 1) {
        setArticles(res.articles);
        // Start polling if backend is still collecting (< 5 articles)
        if (res.total < 5 && !pollRef.current) {
          setCollecting(true);
          let polls = 0;
          pollRef.current = setInterval(async () => {
            polls++;
            try {
              const fresh = await api.getCompanyNews(company.id, 1, perPage);
              if (fresh.total >= 5 || polls >= 6) {
                // Articles arrived or 30s timeout
                setArticles(fresh.articles);
                setTotalArticles(fresh.total);
                setCollecting(false);
                if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
              }
            } catch { /* keep polling */ }
          }, 5000);
        }
      } else {
        setArticles((prev) => [...prev, ...res.articles]);
      }
      setTotalArticles(res.total);
      setPage(p);
    } catch {
      // Fall back to embedded articles if API fails
      if (p === 1) {
        const embedded: CompanyNewsArticle[] = [
          ...(company.live_news || []).map((n) => ({
            title: n.title, summary: n.content || "", source_name: "live",
            published_at: "", url: n.url, sentiment_score: 0, source_type: "live" as const,
          })),
          ...(company.recent_articles || []).map((a) => ({
            title: a.title, summary: a.summary, source_name: a.source_name,
            published_at: a.published_at, url: a.url, sentiment_score: 0, source_type: "cached" as const,
          })),
        ];
        setArticles(embedded);
        setTotalArticles(embedded.length);
      }
    }
    setLoading(false);
  }

  // B6: Dedup articles by URL before rendering
  const dedupedArticles = (() => {
    const seen = new Set<string>();
    return articles.filter((a) => {
      const key = a.url?.toLowerCase().split("?")[0] || a.title;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  })();

  // Group articles by month
  const grouped = dedupedArticles.reduce<Record<string, CompanyNewsArticle[]>>((acc, a) => {
    let key = "Unknown Date";
    if (a.published_at) {
      try {
        const d = new Date(a.published_at);
        if (!isNaN(d.getTime())) {
          key = d.toLocaleDateString("en-US", { month: "long", year: "numeric" });
        }
      } catch { /* use default */ }
    }
    (acc[key] ||= []).push(a);
    return acc;
  }, {});

  if (loading && articles.length === 0) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 0", gap: 8 }}>
        <Loader2 size={16} style={{ color: "var(--accent)", animation: "spin 1s linear infinite" }} />
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Loading news from article cache...</span>
      </div>
    );
  }

  if (articles.length === 0) {
    return (
      <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
        {collecting ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
            <Loader2 size={18} style={{ color: "var(--accent)", animation: "spin 1s linear infinite" }} />
            <span>Collecting news in background...</span>
            <button
              onClick={() => loadNews(1)}
              style={{ fontSize: 12, color: "var(--accent)", background: "none", border: "1px solid var(--border)", borderRadius: 6, padding: "4px 12px", cursor: "pointer" }}
            >
              Check now
            </button>
          </div>
        ) : (
          "No news intelligence yet. Run the pipeline to populate the article cache, or use Refresh to fetch latest news."
        )}
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 800 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          {totalArticles} articles
        </div>
      </div>

      {/* Grouped by month */}
      {Object.entries(grouped).map(([month, monthArticles]) => (
        <div key={month} style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", marginBottom: 10, padding: "4px 0", borderBottom: "1px solid var(--border)" }}>
            {month}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {monthArticles.map((a, i) => (
              <div key={`${month}-${i}`} className="card" style={{ padding: "12px 16px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: a.source_type === "live" ? "var(--blue)" : "var(--accent)" }}>
                    {a.source_name || (a.source_type === "live" ? "Live" : "Cached")}
                  </span>
                  {a.source_type === "live" && (
                    <span style={{ fontSize: 8, padding: "1px 5px", borderRadius: 3, background: "var(--blue)", color: "#fff", fontWeight: 600 }}>LIVE</span>
                  )}
                  {a.published_at && (
                    <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>
                      {formatDate(a.published_at, "long")}
                    </span>
                  )}
                  {a.url && (
                    <a href={a.url} target="_blank" rel="noopener noreferrer" style={{ marginLeft: "auto", fontSize: 10, color: "var(--blue)", display: "flex", alignItems: "center", gap: 3, textDecoration: "none" }}>
                      Source <ExternalLink size={9} />
                    </a>
                  )}
                </div>
                <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)", lineHeight: 1.4, marginBottom: 4 }}>{a.title}</div>
                {a.summary && (
                  <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>{a.summary}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Load more button */}
      {articles.length < totalArticles && (
        <div style={{ textAlign: "center", padding: "12px 0" }}>
          <button
            onClick={() => loadNews(page + 1)}
            disabled={loading}
            style={{
              display: "inline-flex", alignItems: "center", gap: 6,
              padding: "8px 20px", borderRadius: 7,
              border: "1px solid var(--border)", background: "var(--surface)",
              fontSize: 12, fontWeight: 500, color: "var(--text-secondary)",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> : <ChevronDown size={12} />}
            Load more ({totalArticles - articles.length} remaining)
          </button>
        </div>
      )}
    </div>
  );
}

function LeadsTab({ company, genLoading, onGenLeads }: { company: SavedCompany; genLoading: boolean; onGenLeads: () => void }) {
  const { leads: contextLeads } = usePipelineContext();
  const [apiLeads, setApiLeads] = useState<LeadRecord[]>([]);
  const hasContacts = (company.contacts?.length ?? 0) > 0;

  // Fetch leads from API on mount (context may be empty if no active run)
  useEffect(() => {
    api.getLeads({ limit: 200 }).then(({ leads }) => {
      const matching = leads.filter(
        (l) => l.company_name.toLowerCase() === company.company_name.toLowerCase()
      );
      setApiLeads(matching);
    }).catch(() => {});
  }, [company.company_name]);

  // Merge context + API leads, dedup by company+contact
  const allLeads = contextLeads.length > 0 ? contextLeads : apiLeads;
  const pipelineLeads = allLeads.filter(
    (l) => l.company_name.toLowerCase() === company.company_name.toLowerCase()
  );

  const totalPeople = (company.contacts?.length ?? 0) + pipelineLeads.length;

  return (
    <div style={{ maxWidth: 800 }}>
      {/* Header with generate button */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            {totalPeople > 0 ? `${totalPeople} People & Leads` : "No Contacts Yet"}
          </div>
          {pipelineLeads.length > 0 && (
            <Link
              href={`/leads?company=${encodeURIComponent(company.company_name)}`}
              style={{ fontSize: 10, color: "var(--accent)", textDecoration: "none", display: "flex", alignItems: "center", gap: 3, padding: "2px 8px", borderRadius: 5, border: "1px solid var(--accent)33", background: "var(--accent-light)" }}
            >
              View in Leads <ExternalLink size={9} />
            </Link>
          )}
        </div>
        <button
          onClick={onGenLeads}
          disabled={genLoading}
          style={{ display: "flex", alignItems: "center", gap: 4, padding: "6px 14px", borderRadius: 7, border: "1px solid var(--accent)", background: genLoading ? "var(--surface)" : "var(--accent)", fontSize: 11, fontWeight: 600, color: genLoading ? "var(--text-secondary)" : "#fff", cursor: genLoading ? "not-allowed" : "pointer" }}
        >
          {genLoading ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> : <Zap size={12} />}
          {hasContacts ? "Generate More" : "Generate Leads"}
        </button>
      </div>

      {company.contacts_reasoning && (
        <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 14, padding: "8px 12px", background: "var(--surface-raised)", borderRadius: 7, border: "1px solid var(--border)" }}>
          {company.contacts_reasoning}
        </div>
      )}

      {/* Pipeline leads section */}
      {pipelineLeads.length > 0 && (
        <div style={{ marginBottom: hasContacts ? 20 : 0 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
            Pipeline Leads ({pipelineLeads.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {pipelineLeads.map((lead, i) => (
              <PipelineLeadCard key={lead.id ?? i} lead={lead} />
            ))}
          </div>
        </div>
      )}

      {/* Generated contacts section */}
      {hasContacts && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
            Generated Contacts ({company.contacts.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {company.contacts.map((c: PersonRecord, i: number) => (
              <ContactCard key={i} contact={c} />
            ))}
          </div>
        </div>
      )}

      {totalPeople === 0 && (
        <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
          Click &quot;Generate Leads&quot; to find contacts at {company.company_name} using Apollo + Tavily and create personalized outreach.
        </div>
      )}
    </div>
  );
}

/** Pipeline lead card — links to /leads/[id] */
function PipelineLeadCard({ lead }: { lead: LeadRecord }) {
  const cc = confidenceColor(lead.confidence);
  return (
    <Link href={`/leads/${lead.id ?? 0}`} style={{ textDecoration: "none" }}>
      <div className="card card-hover" style={{ padding: "12px 16px", display: "flex", alignItems: "center", gap: 12, cursor: "pointer" }}>
        {/* Confidence badge */}
        <div style={{ width: 38, height: 38, borderRadius: 8, background: cc.bg, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <span className="num" style={{ fontSize: 14, fontWeight: 600, color: cc.text, lineHeight: 1 }}>
            {Math.round(lead.confidence * 100)}
          </span>
        </div>
        {/* Lead info */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {lead.trend_title}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            {lead.contact_name && <span style={{ display: "flex", alignItems: "center", gap: 3 }}><User size={10} /> {lead.contact_name}</span>}
            {lead.contact_role && <span>{lead.contact_role}</span>}
            {lead.contact_email && <span style={{ color: "var(--blue)" }}>{lead.contact_email}</span>}
          </div>
        </div>
        {/* Meta badges */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3, flexShrink: 0 }}>
          <span className={`badge ${lead.lead_type === "pain" ? "badge-red" : lead.lead_type === "opportunity" ? "badge-green" : lead.lead_type === "risk" ? "badge-amber" : "badge-muted"}`} style={{ fontSize: 9 }}>
            {lead.lead_type}
          </span>
          <span style={{ fontSize: 10, color: "var(--text-xmuted)" }}>
            H{lead.hop} · {lead.urgency_weeks}w
          </span>
        </div>
        <ChevronRight size={13} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
      </div>
    </Link>
  );
}

/** Full contact card with outreach email */
function ContactCard({ contact }: { contact: PersonRecord }) {
  const [showEmail, setShowEmail] = useState(false);
  const c = contact;

  return (
    <div className="card" style={{ padding: 0 }}>
      {/* Contact header */}
      <div style={{ padding: "14px 16px", display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 40, height: 40, borderRadius: 10, background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <User size={18} style={{ color: "var(--accent)" }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", marginBottom: 2 }}>{c.person_name}</div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{c.role}</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3, flexShrink: 0 }}>
          {c.seniority_tier && (
            <span className={`badge ${c.seniority_tier === "decision_maker" ? "badge-green" : c.seniority_tier === "influencer" ? "badge-blue" : "badge-muted"}`} style={{ fontSize: 9 }}>
              {c.seniority_tier.replace("_", " ")}
            </span>
          )}
          {c.reach_score > 0 && (
            <span style={{ fontSize: 10, color: "var(--text-muted)" }}>Reach: {c.reach_score}</span>
          )}
        </div>
      </div>

      {/* Contact details */}
      <div style={{ padding: "0 16px 12px", display: "flex", flexWrap: "wrap", gap: 10 }}>
        {c.email && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Mail size={11} style={{ color: "var(--text-muted)" }} />
            <a href={`mailto:${c.email}`} style={{ fontSize: 12, color: "var(--blue)", textDecoration: "none" }}>{c.email}</a>
            {c.email_confidence > 0 && (
              <span style={{ fontSize: 9, fontWeight: 600, color: c.email_confidence >= 80 ? "var(--green)" : "var(--text-muted)", padding: "1px 5px", borderRadius: 4, background: c.email_confidence >= 80 ? "var(--green-light)" : "var(--surface-raised)" }}>
                {c.email_confidence}%
              </span>
            )}
          </div>
        )}
        {c.linkedin_url && (
          <a href={c.linkedin_url} target="_blank" rel="noopener noreferrer" style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 12, color: "var(--blue)", textDecoration: "none" }}>
            <Linkedin size={11} /> LinkedIn
          </a>
        )}
      </div>

      {/* Outreach email toggle */}
      {c.outreach_subject && (
        <>
          <div style={{ borderTop: "1px solid var(--border)", padding: "8px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <button
              onClick={() => setShowEmail(!showEmail)}
              style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, fontWeight: 500, color: "var(--accent)", background: "none", border: "none", cursor: "pointer", padding: 0 }}
            >
              <Send size={11} />
              {showEmail ? "Hide Outreach Email" : "View Outreach Email"}
            </button>
            {c.outreach_tone && (
              <span className="badge badge-muted" style={{ fontSize: 9 }}>{c.outreach_tone}</span>
            )}
          </div>
          {showEmail && (
            <div style={{ padding: "0 16px 14px" }}>
              <div style={{ background: "var(--surface-raised)", borderRadius: 8, padding: "12px 14px", border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                  <strong>Subject:</strong> <span style={{ color: "var(--text)" }}>{c.outreach_subject}</span>
                </div>
                <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
                  {c.outreach_body}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Compact contact row for overview tab — clickable to leads page */
function ContactRow({ contact, companyName }: { contact: PersonRecord; companyName?: string }) {
  const c = contact;
  const searchQuery = c.person_name || companyName || "";
  return (
    <Link href={`/leads?company=${encodeURIComponent(searchQuery)}`} style={{ textDecoration: "none" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", background: "var(--surface-raised)", borderRadius: 7, cursor: "pointer", transition: "background 150ms" }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "var(--surface-hover)")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "var(--surface-raised)")}
      >
        <div style={{ width: 28, height: 28, borderRadius: 7, background: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <User size={13} style={{ color: "var(--accent)" }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{c.person_name}</div>
          <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{c.role}</div>
        </div>
        {c.email && (
          <div style={{ display: "flex", alignItems: "center", gap: 3, flexShrink: 0 }}>
            <Mail size={10} style={{ color: "var(--text-xmuted)" }} />
            <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{c.email}</span>
          </div>
        )}
        <ChevronRight size={11} style={{ color: "var(--text-xmuted)", flexShrink: 0 }} />
      </div>
    </Link>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ textAlign: "center", padding: "10px 8px", background: "var(--surface-raised)", borderRadius: 8, border: "1px solid var(--border)" }}>
      <div className="num" style={{ fontSize: 22, color, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>{label}</div>
    </div>
  );
}
