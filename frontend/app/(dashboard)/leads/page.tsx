"use client";

import { Suspense, useState, useMemo, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Search, ArrowUpDown, X } from "lucide-react";
import { LeadsPanel } from "@/components/dashboard/leads-panel";
import { usePipelineContext } from "@/contexts/pipeline-context";
import { api } from "@/lib/api";
import type { LeadRecord } from "@/lib/types";

type SortKey = "confidence" | "company" | "urgency";

const TYPE_OPTIONS = ["all", "pain", "opportunity", "risk", "intelligence"] as const;
const HOP_OPTIONS = ["all", "1", "2", "3"] as const;

export default function LeadsPage() {
  return (
    <Suspense>
      <LeadsContent />
    </Suspense>
  );
}

function LeadsContent() {
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("selected") ?? undefined;
  const { leads: contextLeads, initialLoading } = usePipelineContext();

  const [leads, setLeads] = useState<LeadRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [leadType, setLeadType] = useState("all");
  const [hop, setHop] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("confidence");
  const [sortAsc, setSortAsc] = useState(false);

  // Load leads: prefer context, fall back to API
  useEffect(() => {
    if (contextLeads.length > 0) {
      setLeads(contextLeads);
      setLoading(false);
      return;
    }
    if (initialLoading) return; // wait for context to finish loading first
    api.getLeads({ limit: 200 })
      .then(({ leads: fresh }) => { if (fresh.length > 0) setLeads(fresh); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [contextLeads, initialLoading]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((v) => !v);
    else { setSortKey(key); setSortAsc(false); }
  };

  const filtered: LeadRecord[] = useMemo(() => {
    let list = leads.filter((l) => {
      if (leadType !== "all" && l.lead_type !== leadType) return false;
      if (hop !== "all" && l.hop !== Number(hop)) return false;
      if (search) {
        const q = search.toLowerCase();
        return (
          l.company_name.toLowerCase().includes(q) ||
          l.trend_title.toLowerCase().includes(q) ||
          l.pain_point.toLowerCase().includes(q) ||
          l.event_type.toLowerCase().includes(q)
        );
      }
      return true;
    });

    list = list.sort((a, b) => {
      let diff = 0;
      if (sortKey === "confidence") diff = a.confidence - b.confidence;
      else if (sortKey === "company") diff = a.company_name.localeCompare(b.company_name);
      else if (sortKey === "urgency") diff = a.urgency_weeks - b.urgency_weeks;
      return sortAsc ? diff : -diff;
    });

    return list;
  }, [leads, search, leadType, hop, sortKey, sortAsc]);

  return (
    <>
      {/* Header */}
      <div
        style={{
          padding: "16px 24px 14px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 14 }}>
          <h1 className="font-display" style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}>
            Leads
          </h1>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {filtered.length} of {leads.length} leads
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {/* Search */}
          <div style={{ position: "relative", flex: "1 1 200px", maxWidth: 300 }}>
            <Search size={13} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }} />
            <input
              type="text"
              placeholder="Search companies, trends, pain points..."
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

          {/* Lead type filter */}
          <select
            value={leadType}
            onChange={(e) => setLeadType(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
          >
            {TYPE_OPTIONS.map((t) => <option key={t} value={t}>{t === "all" ? "All types" : t.charAt(0).toUpperCase() + t.slice(1)}</option>)}
          </select>

          {/* Hop filter */}
          <select
            value={hop}
            onChange={(e) => setHop(e.target.value)}
            style={{ padding: "6px 10px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", fontSize: 12, color: "var(--text-secondary)", cursor: "pointer", outline: "none" }}
          >
            {HOP_OPTIONS.map((h) => <option key={h} value={h}>{h === "all" ? "All hops" : `Hop ${h}`}</option>)}
          </select>

          {/* Sort */}
          <div style={{ display: "flex", gap: 4 }}>
            {([["confidence", "Score"], ["company", "Company"], ["urgency", "Urgency"]] as [SortKey, string][]).map(([key, label]) => (
              <button
                key={key}
                onClick={() => toggleSort(key)}
                style={{
                  display: "flex", alignItems: "center", gap: 4,
                  padding: "5px 10px", borderRadius: 7, fontSize: 11, fontWeight: 500, cursor: "pointer",
                  border: "1px solid var(--border)",
                  background: sortKey === key ? "var(--surface-raised)" : "var(--surface)",
                  color: sortKey === key ? "var(--text)" : "var(--text-muted)",
                }}
              >
                <ArrowUpDown size={10} />
                {label}
                {sortKey === key && <span style={{ fontSize: 9 }}>{sortAsc ? "↑" : "↓"}</span>}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Leads panel */}
      <div style={{ flex: 1, overflow: "hidden", minHeight: 0 }}>
        {loading ? (
          <div style={{ padding: "18px 24px", display: "flex", flexDirection: "column", gap: 10 }}>
            {[0, 1, 2, 3, 4].map((i) => (
              <div key={i} className="card" style={{ padding: "14px 16px" }}>
                <div className="skeleton" style={{ height: 14, width: "50%", marginBottom: 8 }} />
                <div className="skeleton" style={{ height: 12, width: "80%", marginBottom: 4 }} />
                <div className="skeleton" style={{ height: 12, width: "40%" }} />
              </div>
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: "50px 24px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
            {leads.length === 0
              ? "No leads yet — run the pipeline to generate."
              : <>No leads match your filters.{" "}<button onClick={() => { setSearch(""); setLeadType("all"); setHop("all"); }} style={{ color: "var(--accent)", background: "none", border: "none", cursor: "pointer", fontSize: 13 }}>Clear filters</button></>}
          </div>
        ) : (
          <LeadsPanel leads={filtered} selectedId={selectedId} />
        )}
      </div>
    </>
  );
}
