"use client";

import { useState, useMemo, useEffect } from "react";
import { Search, Building2, X } from "lucide-react";
import { LeadDetailPanel } from "@/components/dashboard/lead-detail-panel";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import type { LeadRecord } from "@/lib/types";

const TYPE_CLASSES: Record<string, string> = {
  pain: "badge-red",
  opportunity: "badge-green",
  risk: "badge-amber",
  intelligence: "badge-blue",
};

/** Group leads by company — keep highest-confidence lead per company */
function groupByCompany(leads: LeadRecord[]): LeadRecord[] {
  const map = new Map<string, LeadRecord>();
  for (const lead of leads) {
    const existing = map.get(lead.company_name);
    if (!existing || lead.confidence > existing.confidence) {
      map.set(lead.company_name, lead);
    }
  }
  return Array.from(map.values()).sort((a, b) => b.confidence - a.confidence);
}

function confidenceColor(c: number) {
  if (c >= 0.75) return { text: "var(--green)",  bg: "var(--green-light)"  };
  if (c >= 0.50) return { text: "var(--accent)", bg: "var(--amber-light)"  };
  return               { text: "var(--text-muted)", bg: "var(--surface-raised)" };
}

export default function CompaniesPage() {
  const { leads: contextLeads, initialLoading } = usePipelineContext();
  const [allLeads, setAllLeads] = useState<LeadRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sizeFilter, setSizeFilter] = useState("all");
  const [panelLead, setPanelLead] = useState<LeadRecord | null>(null);

  useEffect(() => {
    if (contextLeads.length > 0) {
      setAllLeads(contextLeads);
      setLoading(false);
      return;
    }
    if (initialLoading) return;
    api.getLeads({ limit: 200 })
      .then(({ leads }) => { if (leads.length > 0) setAllLeads(leads); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [contextLeads, initialLoading]);

  const companies = useMemo(() => groupByCompany(allLeads), [allLeads]);
  const sizeOptions = useMemo(
    () => Array.from(new Set(companies.map((l) => l.company_size_band).filter(Boolean))).sort(),
    [companies],
  );

  const filtered = useMemo(() => {
    return companies.filter((lead) => {
      if (sizeFilter !== "all" && lead.company_size_band !== sizeFilter) return false;
      if (search) {
        const q = search.toLowerCase();
        return (
          lead.company_name.toLowerCase().includes(q) ||
          lead.trend_title.toLowerCase().includes(q) ||
          lead.company_city.toLowerCase().includes(q) ||
          lead.company_state.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [companies, search, sizeFilter]);

  return (
    <>
      {/* Header */}
      <div style={{ padding: "16px 24px 14px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 14 }}>
          <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Companies
          </h1>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {filtered.length} of {companies.length} matched
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <div style={{ position: "relative", flex: "1 1 200px", maxWidth: 300 }}>
            <Search size={13} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }} />
            <input
              type="text"
              placeholder="Search companies, cities..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ width: "100%", padding: "7px 10px 7px 30px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text)", outline: "none" }}
            />
            {search && (
              <button onClick={() => setSearch("")} style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", display: "flex" }}>
                <X size={12} />
              </button>
            )}
          </div>

          <select
            value={sizeFilter}
            onChange={(e) => setSizeFilter(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
          >
            <option value="all">All sizes</option>
            {sizeOptions.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      {/* Company grid */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        {loading ? (
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
        ) : filtered.length === 0 ? (
          <div style={{ padding: "50px 24px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
            {companies.length === 0
              ? "No companies yet — run the pipeline to discover targets."
              : <>No companies match your filters.{" "}<button onClick={() => { setSearch(""); setSizeFilter("all"); }} style={{ color: "var(--accent)", background: "none", border: "none", cursor: "pointer", fontSize: 13 }}>Clear filters</button></>}
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
            {filtered.map((lead) => (
              <CompanyCard key={lead.company_name} lead={lead} onClick={() => setPanelLead(lead)} />
            ))}
          </div>
        )}
      </div>

      <LeadDetailPanel lead={panelLead} onClose={() => setPanelLead(null)} />
    </>
  );
}

function CompanyCard({ lead, onClick }: { lead: LeadRecord; onClick: () => void }) {
  const { text, bg } = confidenceColor(lead.confidence);
  const location = [lead.company_city, lead.company_state].filter(Boolean).join(", ");

  return (
    <div
      className="card card-hover"
      onClick={onClick}
      style={{ padding: "14px 16px", cursor: "pointer" }}
    >
      {/* Top row */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 10 }}>
        <div style={{ width: 36, height: 36, borderRadius: 8, background: "var(--surface-raised)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, border: "1px solid var(--border)" }}>
          <Building2 size={16} style={{ color: "var(--text-secondary)" }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 2 }}>
            {lead.company_name}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {lead.company_size_band}{location ? ` · ${location}` : ""}
          </div>
        </div>
        <div style={{ background: bg, borderRadius: 7, padding: "3px 8px", flexShrink: 0 }}>
          <span className="num" style={{ fontSize: 14, color: text, lineHeight: 1 }}>
            {Math.round(lead.confidence * 100)}
          </span>
        </div>
      </div>

      {/* Trigger trend */}
      <p style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: 10, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
        {lead.trend_title}
      </p>

      <div style={{ height: 1, background: "var(--border)", marginBottom: 10 }} />

      {/* Footer */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span className={`badge ${TYPE_CLASSES[lead.lead_type] ?? "badge-muted"}`} style={{ fontSize: 9 }}>
          {lead.lead_type}
        </span>
        <span className="badge badge-blue" style={{ fontSize: 9 }}>H{lead.hop}</span>
        {lead.contact_role && (
          <span style={{ fontSize: 10, color: "var(--text-xmuted)", marginLeft: "auto" }}>{lead.contact_role}</span>
        )}
      </div>
    </div>
  );
}
