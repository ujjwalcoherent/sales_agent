"use client";

import { Suspense, useState, useMemo, useEffect, useRef, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Search, X, Globe, Loader2, Newspaper, RefreshCw, ArrowRight, MapPin, Users, Calendar, TrendingUp, SlidersHorizontal, ArrowUpDown, Mail } from "lucide-react";
import { CompanyLogo } from "@/components/ui/company-logo";
import { CompanyDetailPanel } from "@/components/dashboard/company-detail-panel";
import { api } from "@/lib/api";
import type { SavedCompany, CompanySearchResult } from "@/lib/types";

/* ── Helpers ─────────────────────────────────────────────── */

/** Parse employee_count string → number (handles "11,528", "51‑200", "over 600,000 professionals") */
function parseEmployeeCount(raw: string): number {
  if (!raw) return 0;
  const nums = raw.replace(/[,\s]/g, "").match(/\d+/g);
  if (!nums) return 0;
  return Math.max(...nums.map(Number));
}

const SIZE_BUCKETS = [
  { label: "1–50", min: 1, max: 50 },
  { label: "51–200", min: 51, max: 200 },
  { label: "201–1K", min: 201, max: 1000 },
  { label: "1K–10K", min: 1001, max: 10000 },
  { label: "10K+", min: 10001, max: Infinity },
] as const;

/**
 * Adaptive industry normalizer — no hardcoded map needed.
 *
 * Rules (applied in order):
 * 1. Strip trailing noise: "industry", "sector", "services", "solutions", "market"
 * 2. Collapse known patterns: "information technology" → "IT", "x & y" → keep first
 * 3. Capitalize the result
 *
 * This handles any industry string from the DB automatically.
 */
const _STRIP_SUFFIXES = /\s+(industry|sector|services|solutions|market|companies|firms)$/i;
const _IT_PATTERN = /^information[- ]?technology/i;
const _EDTECH_PATTERN = /^edu?tech$/i;

function normalizeIndustry(raw: string): string {
  if (!raw) return "";
  let s = raw.trim();

  // Collapse multi-part descriptions (take before first comma or dash-with-spaces)
  if (s.includes(",")) s = s.split(",")[0].trim();
  if (s.includes(" - ")) s = s.split(" - ")[0].trim();

  // IT variations
  if (_IT_PATTERN.test(s)) return "IT Services";
  if (/^it\b/i.test(s) && s.length < 30) return "IT Services";

  // EdTech variations
  if (_EDTECH_PATTERN.test(s)) return "Edtech";

  // Strip trailing noise words
  s = s.replace(_STRIP_SUFFIXES, "").trim();
  if (!s) return raw.trim();

  // Capitalize first letter of each word (title case)
  return s.replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Known city name aliases → canonical form */
const CITY_ALIASES: Record<string, string> = {
  bangalore: "Bengaluru",
  bengaluru: "Bengaluru",
  mumbai: "Mumbai",
  "new delhi": "Delhi",
  delhi: "Delhi",
  noida: "Noida",
  gurugram: "Gurugram",
  gurgaon: "Gurugram",
  hyderabad: "Hyderabad",
  chennai: "Chennai",
  pune: "Pune",
  kolkata: "Kolkata",
  bhubaneswar: "Bhubaneswar",
  "santa clara": "Santa Clara",
  redmond: "Redmond",
  austin: "Austin",
  hawthorne: "Hawthorne",
  "san francisco": "San Francisco",
  "sector 62": "Noida",  // Known Noida tech hub area
  "sector 63": "Noida",
  starbase: "Starbase",
};

/** Extract city name from a full HQ address */
function extractCity(hq: string): string {
  if (!hq) return "";
  const lower = hq.toLowerCase();

  // First pass: check if any known city name appears anywhere in the string
  for (const [alias, canonical] of Object.entries(CITY_ALIASES)) {
    if (lower.includes(alias)) return canonical;
  }

  // Second pass: split by comma, skip parts that look like street addresses or zip codes
  const parts = hq.split(",").map((s) => s.trim());
  for (const p of parts) {
    if (/\d/.test(p) && /(floor|rd|main|cross|layout|sector|plot|marg|no\.?|sco|phase)/i.test(p)) continue;
    if (/^\d{5,6}$/.test(p.replace(/\s/g, ""))) continue;
    if (/^(above|near|behind|opposite|next to)/i.test(p)) continue;
    if (p.length >= 3 && p.length <= 35 && /^[A-Z]/.test(p)) return p;
  }

  return hq.slice(0, 25);
}

type SortKey = "name" | "employees" | "articles" | "newest";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "name", label: "A–Z" },
  { key: "employees", label: "Employees" },
  { key: "articles", label: "Articles" },
  { key: "newest", label: "Founded" },
];

/* ── Page ────────────────────────────────────────────────── */

export default function CompaniesPage() {
  return (
    <Suspense fallback={<div style={{ padding: 24, color: "var(--text-muted)", fontSize: 13 }}>Loading...</div>}>
      <CompaniesPageInner />
    </Suspense>
  );
}

function CompaniesPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialSearchDone = useRef(false);

  // Saved companies (loaded on mount)
  const [savedCompanies, setSavedCompanies] = useState<SavedCompany[]>([]);
  const [loadingSaved, setLoadingSaved] = useState(true);

  // Web search results (loaded on demand)
  const [webResults, setWebResults] = useState<CompanySearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchInfo, setSearchInfo] = useState("");
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchSlow, setSearchSlow] = useState(false);
  const slowTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Filter / sort state
  const [query, setQuery] = useState("");
  const [industryFilter, setIndustryFilter] = useState("all");
  const [sizeFilter, setSizeFilter] = useState("all");
  const [locationFilter, setLocationFilter] = useState("all");
  const [sortBy, setSortBy] = useState<SortKey>("name");
  const [panelCompany, setPanelCompany] = useState<CompanySearchResult | null>(null);
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 24;

  // Load saved companies on mount
  useEffect(() => {
    api.getSavedCompanies(200)
      .then((res) => setSavedCompanies(res.companies))
      .catch(() => {})
      .finally(() => setLoadingSaved(false));
  }, []);

  // Auto-search when navigated with ?search=CompanyName
  useEffect(() => {
    if (initialSearchDone.current) return;
    const q = searchParams.get("search");
    if (q) {
      initialSearchDone.current = true;
      setQuery(q);
      setTimeout(() => handleWebSearch(q), 0);
    }
    // ?industry=X → set industry filter (from trends page links)
    const ind = searchParams.get("industry");
    if (ind) {
      setIndustryFilter(ind);
    }
  }, [searchParams]);

  /* ── Derived filter options (computed from full savedCompanies) ── */

  const industryOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of savedCompanies) {
      const norm = normalizeIndustry(c.industry);
      if (norm) counts.set(norm, (counts.get(norm) || 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([label, count]) => ({ label, count }));
  }, [savedCompanies]);

  const sizeOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of savedCompanies) {
      const n = parseEmployeeCount(c.employee_count || "");
      if (n <= 0) continue;
      for (const b of SIZE_BUCKETS) {
        if (n >= b.min && n <= b.max) { counts.set(b.label, (counts.get(b.label) || 0) + 1); break; }
      }
    }
    return SIZE_BUCKETS.filter((b) => counts.has(b.label)).map((b) => ({ label: b.label, count: counts.get(b.label) || 0 }));
  }, [savedCompanies]);

  const locationOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of savedCompanies) {
      const city = extractCity(c.headquarters);
      if (city) counts.set(city, (counts.get(city) || 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([label, count]) => ({ label, count }));
  }, [savedCompanies]);

  const activeFilterCount =
    (industryFilter !== "all" ? 1 : 0) +
    (sizeFilter !== "all" ? 1 : 0) +
    (locationFilter !== "all" ? 1 : 0);

  const clearAllFilters = useCallback(() => {
    setIndustryFilter("all");
    setSizeFilter("all");
    setLocationFilter("all");
  }, []);

  /* ── Filtered + sorted list ──────────────────────────── */

  const filteredSaved = useMemo(() => {
    let list = savedCompanies;

    // Text search (includes reason_relevant so industry-search saves match)
    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter((c) =>
        c.company_name.toLowerCase().includes(q) ||
        (c.industry || "").toLowerCase().includes(q) ||
        (c.domain || "").toLowerCase().includes(q) ||
        (c.description || "").toLowerCase().includes(q) ||
        (c.headquarters || "").toLowerCase().includes(q) ||
        (c.ceo || "").toLowerCase().includes(q) ||
        (c.reason_relevant || "").toLowerCase().includes(q) ||
        (c.sub_industries || []).some((s) => s.toLowerCase().includes(q))
      );
    }

    // Industry filter — exact match on normalized name, or substring match for URL params
    if (industryFilter !== "all") {
      list = list.filter((c) => {
        const norm = normalizeIndustry(c.industry);
        if (norm === industryFilter) return true;
        // Fuzzy match: check if filter term appears in industry, description, or sub_industries
        const fq = industryFilter.toLowerCase();
        return (
          (c.industry || "").toLowerCase().includes(fq) ||
          (c.description || "").toLowerCase().includes(fq) ||
          (c.sub_industries || []).some((s) => s.toLowerCase().includes(fq)) ||
          (c.reason_relevant || "").toLowerCase().includes(fq)
        );
      });
    }

    // Size bucket
    if (sizeFilter !== "all") {
      const bucket = SIZE_BUCKETS.find((b) => b.label === sizeFilter);
      if (bucket) {
        list = list.filter((c) => {
          const n = parseEmployeeCount(c.employee_count || "");
          return n >= bucket.min && n <= bucket.max;
        });
      }
    }

    // Location
    if (locationFilter !== "all") {
      list = list.filter((c) => extractCity(c.headquarters) === locationFilter);
    }

    // Sort
    list = [...list].sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.company_name.localeCompare(b.company_name);
        case "employees":
          return parseEmployeeCount(b.employee_count || "") - parseEmployeeCount(a.employee_count || "");
        case "articles":
          return (b.article_count || 0) - (a.article_count || 0);
        case "newest":
          return (b.founded_year || 0) - (a.founded_year || 0);
        default:
          return 0;
      }
    });

    // Dedup by company name (case-insensitive) — safety net for DB duplicates
    const seen = new Set<string>();
    list = list.filter((c) => {
      const key = c.company_name.toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    return list;
  }, [savedCompanies, query, industryFilter, sizeFilter, locationFilter, sortBy]);

  // Reset to page 1 when filters change
  useEffect(() => { setPage(1); }, [filteredSaved.length, query, industryFilter, sizeFilter, locationFilter, sortBy]);

  // Dedup web results against saved companies + internal dedup by name
  const dedupedWebResults = useMemo(() => {
    const savedNames = new Set(savedCompanies.map((c) => c.company_name.toLowerCase()));
    const seen = new Set<string>();
    return webResults.filter((c) => {
      const key = c.company_name.toLowerCase();
      if (savedNames.has(key) || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [webResults, savedCompanies]);

  // Show "Search Web" button when query is 3+ chars and no saved match
  const showWebSearchButton = query.trim().length >= 3 && !searchLoading;

  async function handleWebSearch(q?: string) {
    const searchTerm = (q || query).trim();
    if (!searchTerm) return;
    setSearchLoading(true);
    setSearchSlow(false);
    setSearchError(null);
    setWebResults([]);
    setSearchInfo("");

    if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
    slowTimerRef.current = setTimeout(() => setSearchSlow(true), 30000);

    try {
      const res = await api.searchCompanies(searchTerm);
      setWebResults(res.companies);
      setSearchInfo(
        `${res.companies.length} results · ${res.search_type} search · ${(res.search_duration_ms / 1000).toFixed(1)}s` +
        (res.cached_articles_used > 0 ? ` · ${res.cached_articles_used} cached articles` : "")
      );
      // Refresh saved companies — web search saves new companies to DB
      api.getSavedCompanies(200)
        .then((r) => setSavedCompanies(r.companies))
        .catch(() => {});
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      const isTimeout = msg.toLowerCase().includes("timeout") || msg.includes("504");
      setSearchError(isTimeout ? "Search timed out. Try again." : `Search failed: ${msg}`);
      setSearchInfo("");
    }
    if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
    setSearchSlow(false);
    setSearchLoading(false);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    handleWebSearch();
  }

  const totalDisplayed = filteredSaved.length + dedupedWebResults.length;

  return (
    <>
      {/* Header */}
      <div style={{ padding: "16px 24px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 12 }}>
          <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Companies
          </h1>
          <span style={{ fontSize: 12, color: searchError ? "var(--red)" : "var(--text-muted)", display: "flex", alignItems: "center", gap: 6 }}>
            {searchError ? (
              <>
                {searchError}
                <button
                  onClick={() => handleWebSearch()}
                  style={{ background: "none", border: "1px solid var(--red)", borderRadius: 5, padding: "2px 8px", fontSize: 11, color: "var(--red)", cursor: "pointer", display: "flex", alignItems: "center", gap: 3 }}
                >
                  <RefreshCw size={10} /> Retry
                </button>
              </>
            ) : searchInfo ? (
              searchInfo
            ) : loadingSaved ? (
              "Loading..."
            ) : (
              `${filteredSaved.length}` + (filteredSaved.length !== savedCompanies.length ? ` of ${savedCompanies.length}` : "") + ` companies` + (query ? ` matching "${query}"` : ` saved`)
            )}
          </span>
        </div>

        {/* Search bar */}
        <form onSubmit={handleSubmit} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <div style={{ position: "relative", flex: "1 1 240px", maxWidth: 480 }}>
            <Search size={13} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }} />
            <input
              type="text"
              placeholder="Search companies or type to search web..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{ width: "100%", padding: "7px 10px 7px 30px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text)", outline: "none" }}
            />
            {query && (
              <button
                type="button"
                onClick={() => { setQuery(""); setWebResults([]); setSearchInfo(""); setSearchError(null); }}
                style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", display: "flex" }}
              >
                <X size={12} />
              </button>
            )}
          </div>

          {showWebSearchButton && (
            <button
              type="submit"
              disabled={searchLoading}
              style={{ padding: "6px 14px", borderRadius: 7, border: "1px solid var(--accent)", background: "var(--accent)", color: "#fff", fontSize: 12, fontWeight: 500, cursor: searchLoading ? "not-allowed" : "pointer", opacity: searchLoading ? 0.7 : 1, display: "flex", alignItems: "center", gap: 4 }}
            >
              {searchLoading ? <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> : <Globe size={12} />}
              Search Web
            </button>
          )}
        </form>

        {/* Filter chips row */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <SlidersHorizontal size={12} style={{ color: "var(--text-muted)", flexShrink: 0 }} />

          {/* Industry filter */}
          {industryOptions.length > 1 && (
            <FilterChipSelect
              label="Industry"
              value={industryFilter}
              onChange={setIndustryFilter}
              options={industryOptions}
            />
          )}

          {/* Size filter */}
          {sizeOptions.length > 0 && (
            <FilterChipSelect
              label="Size"
              value={sizeFilter}
              onChange={setSizeFilter}
              options={sizeOptions}
            />
          )}

          {/* Location filter */}
          {locationOptions.length > 1 && (
            <FilterChipSelect
              label="Location"
              value={locationFilter}
              onChange={setLocationFilter}
              options={locationOptions}
            />
          )}

          {/* Divider */}
          <span style={{ width: 1, height: 16, background: "var(--border)", margin: "0 2px" }} />

          {/* Sort */}
          <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <ArrowUpDown size={11} style={{ color: "var(--text-muted)" }} />
            {SORT_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                onClick={() => setSortBy(opt.key)}
                style={{
                  padding: "3px 8px",
                  borderRadius: 5,
                  border: "none",
                  background: sortBy === opt.key ? "var(--accent)" : "transparent",
                  color: sortBy === opt.key ? "#fff" : "var(--text-muted)",
                  fontSize: 11,
                  fontWeight: sortBy === opt.key ? 600 : 400,
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Clear all */}
          {activeFilterCount > 0 && (
            <button
              onClick={clearAllFilters}
              style={{
                marginLeft: "auto",
                display: "flex", alignItems: "center", gap: 3,
                padding: "3px 8px", borderRadius: 5,
                border: "1px solid var(--red)", background: "var(--red-light)",
                color: "var(--red)", fontSize: 10, fontWeight: 500, cursor: "pointer",
              }}
            >
              <X size={9} /> Clear {activeFilterCount} filter{activeFilterCount > 1 ? "s" : ""}
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        {loadingSaved ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="card" style={{ padding: "14px 16px" }}>
                <div style={{ display: "flex", gap: 10, marginBottom: 10 }}>
                  <div className="skeleton" style={{ width: 36, height: 36, borderRadius: 8 }} />
                  <div style={{ flex: 1 }}>
                    <div className="skeleton" style={{ height: 13, width: "70%", marginBottom: 6 }} />
                    <div className="skeleton" style={{ height: 11, width: "40%" }} />
                  </div>
                </div>
                <div className="skeleton" style={{ height: 11, width: "90%", marginBottom: 10 }} />
                <div className="skeleton" style={{ height: 20, width: "60%" }} />
              </div>
            ))}
          </div>
        ) : totalDisplayed === 0 && !searchLoading ? (
          <div style={{ padding: "50px 24px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
            {savedCompanies.length === 0 && !query
              ? "No saved companies yet — search for a company or run the pipeline."
              : query
                ? <>No companies match "{query}".{" "}
                    {!searchLoading && <button onClick={() => handleWebSearch()} style={{ color: "var(--accent)", background: "none", border: "none", cursor: "pointer", fontSize: 13 }}>Search the web</button>}
                  </>
                : null}
          </div>
        ) : (
          <>
            {/* Saved companies section */}
            {filteredSaved.length > 0 && (() => {
              const totalPages = Math.ceil(filteredSaved.length / PAGE_SIZE);
              const pagedSaved = filteredSaved.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
              return (
                <>
                  {(dedupedWebResults.length > 0 || searchInfo) && (
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Saved ({filteredSaved.length})
                    </div>
                  )}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12, marginBottom: dedupedWebResults.length > 0 ? 20 : 0 }}>
                    {pagedSaved.map((company) => (
                      <SavedCompanyCard
                        key={company.id}
                        company={company}
                        onClick={() => router.push(`/companies/${company.id}`)}
                      />
                    ))}
                  </div>
                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, padding: "16px 0", marginBottom: dedupedWebResults.length > 0 ? 8 : 0 }}>
                      <button
                        onClick={() => setPage(p => Math.max(1, p - 1))}
                        disabled={page === 1}
                        style={{ padding: "5px 12px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: page === 1 ? "var(--text-xmuted)" : "var(--text-secondary)", cursor: page === 1 ? "not-allowed" : "pointer" }}
                      >
                        ← Prev
                      </button>
                      {Array.from({ length: totalPages }, (_, i) => i + 1)
                        .filter(p => p === 1 || p === totalPages || Math.abs(p - page) <= 2)
                        .reduce<(number | "...")[]>((acc, p, i, arr) => {
                          if (i > 0 && p - (arr[i - 1] as number) > 1) acc.push("...");
                          acc.push(p);
                          return acc;
                        }, [])
                        .map((p, i) => p === "..." ? (
                          <span key={`ellipsis-${i}`} style={{ fontSize: 12, color: "var(--text-xmuted)", padding: "0 4px" }}>…</span>
                        ) : (
                          <button
                            key={p}
                            onClick={() => setPage(p as number)}
                            style={{ minWidth: 32, padding: "5px 8px", borderRadius: 7, border: "1px solid var(--border)", background: page === p ? "var(--accent)" : "var(--surface)", fontSize: 12, fontWeight: page === p ? 700 : 400, color: page === p ? "#fff" : "var(--text-secondary)", cursor: "pointer" }}
                          >
                            {p}
                          </button>
                        ))
                      }
                      <button
                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                        disabled={page === totalPages}
                        style={{ padding: "5px 12px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: page === totalPages ? "var(--text-xmuted)" : "var(--text-secondary)", cursor: page === totalPages ? "not-allowed" : "pointer" }}
                      >
                        Next →
                      </button>
                      <span style={{ fontSize: 11, color: "var(--text-xmuted)", marginLeft: 4 }}>
                        {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, filteredSaved.length)} of {filteredSaved.length}
                      </span>
                    </div>
                  )}
                </>
              );
            })()}

            {/* Web search loading indicator */}
            {searchLoading && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 0", gap: 12 }}>
                <Loader2 size={24} style={{ color: "var(--accent)", animation: "spin 1s linear infinite" }} />
                <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
                  {searchSlow ? "Taking longer than expected... still searching." : "Searching web + enriching..."}
                </span>
              </div>
            )}

            {/* Web results section */}
            {dedupedWebResults.length > 0 && (
              <>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Web Results ({dedupedWebResults.length})
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
                  {dedupedWebResults.map((company) => (
                    <SearchCompanyCard key={company.id} company={company} onClick={() => setPanelCompany(company)} />
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </div>

      <CompanyDetailPanel company={panelCompany} onClose={() => setPanelCompany(null)} />
    </>
  );
}

/** Saved company card — from DB, click navigates to detail page */
function SavedCompanyCard({ company, onClick }: { company: SavedCompany; onClick: () => void }) {
  return (
    <div className="card card-hover" onClick={onClick} style={{ padding: "14px 16px", cursor: "pointer" }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 8 }}>
        <CompanyLogo domain={company.domain} size={36} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {company.company_name}
            </span>
            {company.stock_ticker && (
              <span style={{ fontSize: 9, fontWeight: 600, color: "var(--green)", background: "var(--green-light)", padding: "1px 5px", borderRadius: 4, flexShrink: 0 }}>
                {company.stock_ticker}
              </span>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-muted)", flexWrap: "wrap" }}>
            {company.domain && (
              <span style={{ color: "var(--blue)", display: "flex", alignItems: "center", gap: 3 }}>
                <Globe size={10} /> {company.domain}
              </span>
            )}
            {company.industry && <span style={{ color: "var(--text-xmuted)" }}>{company.industry}</span>}
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
          {(company.contacts?.length ?? 0) > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "3px 8px", background: "var(--green-light)", borderRadius: 7 }}>
              <Mail size={11} style={{ color: "var(--green)" }} />
              <span className="num" style={{ fontSize: 12, color: "var(--green)" }}>{company.contacts.length}</span>
            </div>
          )}
          {company.article_count > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "3px 8px", background: "var(--accent-light)", borderRadius: 7 }}>
              <Newspaper size={11} style={{ color: "var(--accent)" }} />
              <span className="num" style={{ fontSize: 12, color: "var(--accent)" }}>{company.article_count}</span>
            </div>
          )}
        </div>
      </div>

      {company.description && (
        <p style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 6, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
          {company.description}
        </p>
      )}

      {(company.headquarters || company.employee_count || company.founded_year || company.funding_stage) && (
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
          {company.headquarters && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--text-muted)", background: "var(--surface-raised)", padding: "2px 6px", borderRadius: 4, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={company.headquarters}>
              <MapPin size={9} style={{ flexShrink: 0 }} /> {extractCity(company.headquarters)}
            </span>
          )}
          {company.employee_count && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--text-muted)", background: "var(--surface-raised)", padding: "2px 6px", borderRadius: 4 }}>
              <Users size={9} /> {company.employee_count}
            </span>
          )}
          {company.founded_year && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--text-muted)", background: "var(--surface-raised)", padding: "2px 6px", borderRadius: 4 }}>
              <Calendar size={9} /> {company.founded_year}
            </span>
          )}
          {company.funding_stage && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--accent)", background: "var(--accent-light)", padding: "2px 6px", borderRadius: 4 }}>
              <TrendingUp size={9} /> {company.funding_stage}
            </span>
          )}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
        {normalizeIndustry(company.industry) && (
          <span className="badge badge-blue" style={{ fontSize: 9 }}>{normalizeIndustry(company.industry)}</span>
        )}
        {company.ceo && <span className="badge badge-muted" style={{ fontSize: 9 }}>CEO: {company.ceo}</span>}
      </div>
    </div>
  );
}

/** Search company card — from web search, click opens sidebar */
function SearchCompanyCard({ company, onClick }: { company: CompanySearchResult; onClick: () => void }) {
  const router = useRouter();
  const totalSignals = (company.live_news?.length ?? 0) + (company.recent_articles?.length ?? 0);

  return (
    <div className="card card-hover" onClick={onClick} style={{ padding: "14px 16px", cursor: "pointer" }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 8 }}>
        <CompanyLogo domain={company.domain} size={36} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {company.company_name}
            </span>
            {company.stock_ticker && (
              <span style={{ fontSize: 9, fontWeight: 600, color: "var(--green)", background: "var(--green-light)", padding: "1px 5px", borderRadius: 4, flexShrink: 0 }}>
                {company.stock_ticker}
              </span>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-muted)", flexWrap: "wrap" }}>
            {company.domain && (
              <span style={{ color: "var(--blue)", display: "flex", alignItems: "center", gap: 3 }}>
                <Globe size={10} /> {company.domain}
              </span>
            )}
            {company.industry && <span style={{ color: "var(--text-xmuted)" }}>{company.industry}</span>}
          </div>
        </div>
        {totalSignals > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "3px 8px", background: "var(--accent-light)", borderRadius: 7, flexShrink: 0 }}>
            <Newspaper size={11} style={{ color: "var(--accent)" }} />
            <span className="num" style={{ fontSize: 12, color: "var(--accent)" }}>{totalSignals}</span>
          </div>
        )}
      </div>

      {company.description && (
        <p style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 6, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
          {company.description}
        </p>
      )}

      {(company.headquarters || company.employee_count || company.founded_year || company.funding_stage) && (
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
          {company.headquarters && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--text-muted)", background: "var(--surface-raised)", padding: "2px 6px", borderRadius: 4, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={company.headquarters}>
              <MapPin size={9} style={{ flexShrink: 0 }} /> {extractCity(company.headquarters)}
            </span>
          )}
          {company.employee_count && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--text-muted)", background: "var(--surface-raised)", padding: "2px 6px", borderRadius: 4 }}>
              <Users size={9} /> {company.employee_count}
            </span>
          )}
          {company.founded_year && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--text-muted)", background: "var(--surface-raised)", padding: "2px 6px", borderRadius: 4 }}>
              <Calendar size={9} /> {company.founded_year}
            </span>
          )}
          {company.funding_stage && (
            <span style={{ display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--accent)", background: "var(--accent-light)", padding: "2px 6px", borderRadius: 4 }}>
              <TrendingUp size={9} /> {company.funding_stage}
            </span>
          )}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
        {company.article_count > 0 && (
          <span className="badge badge-muted" style={{ fontSize: 9 }}>{company.article_count} cached</span>
        )}
        {(company.live_news?.length ?? 0) > 0 && (
          <span className="badge badge-blue" style={{ fontSize: 9 }}>{company.live_news!.length} live news</span>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); router.push(`/companies/${company.id}`); }}
          style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 3, fontSize: 10, color: "var(--accent)", background: "none", border: "none", cursor: "pointer", padding: "2px 0" }}
        >
          Company Page <ArrowRight size={10} />
        </button>
      </div>
    </div>
  );
}

/* ── Filter Chip Select ──────────────────────────────────── */

function FilterChipSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { label: string; count: number }[];
}) {
  const isActive = value !== "all";
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        padding: "3px 8px",
        paddingRight: 20,
        borderRadius: 5,
        border: isActive ? "1px solid var(--accent)" : "1px solid var(--border)",
        background: isActive ? "var(--accent-light)" : "var(--surface)",
        color: isActive ? "var(--accent)" : "var(--text-muted)",
        fontSize: 11,
        fontWeight: isActive ? 600 : 400,
        cursor: "pointer",
        outline: "none",
        appearance: "none",
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%238A8878' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E")`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 5px center",
      }}
    >
      <option value="all">{label}</option>
      {options.map((o) => (
        <option key={o.label} value={o.label}>
          {o.label} ({o.count})
        </option>
      ))}
    </select>
  );
}
