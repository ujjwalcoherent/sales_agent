"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  X, ArrowRight, Globe, ExternalLink, Newspaper, Clock,
  MapPin, Users, Calendar, User, TrendingUp,
} from "lucide-react";
import { CompanyLogo } from "@/components/ui/company-logo";
import { InlineTagList } from "@/components/ui/tag-list";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import type { CompanySearchResult, CompanyNewsArticle } from "@/lib/types";

interface Props {
  company: CompanySearchResult | null;
  onClose: () => void;
}

export function CompanyDetailPanel({ company, onClose }: Props) {
  const router = useRouter();
  const [visible, setVisible] = useState(false);
  const [chromaArticles, setChromaArticles] = useState<CompanyNewsArticle[]>([]);
  const [chromaTotal, setChromaTotal] = useState(0);

  useEffect(() => { setVisible(!!company); }, [company]);

  // Fetch ChromaDB cached articles when company changes
  useEffect(() => {
    if (!company?.id) { setChromaArticles([]); setChromaTotal(0); return; }
    api.getCompanyNews(company.id, 1, 10).then((res) => {
      setChromaArticles(res.articles);
      setChromaTotal(res.total);
    }).catch(() => { /* fall back to embedded data */ });
  }, [company?.id]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  if (!company) return null;

  const hasNews = (company.live_news?.length ?? 0) > 0;
  const hasArticles = (company.recent_articles?.length ?? 0) > 0;
  const totalSignals = (company.live_news?.length ?? 0) + (company.recent_articles?.length ?? 0);

  const viewFull = () => {
    router.push(`/companies/${company.id}`);
    onClose();
  };

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed", inset: 0, background: "rgba(0,0,0,0.25)",
          zIndex: 200, opacity: visible ? 1 : 0, transition: "opacity 250ms ease",
        }}
      />

      {/* Drawer */}
      <div
        style={{
          position: "fixed", top: 0, right: 0, bottom: 0,
          width: 540, maxWidth: "90vw",
          background: "var(--surface)", borderLeft: "1px solid var(--border)",
          zIndex: 201, display: "flex", flexDirection: "column",
          boxShadow: "var(--shadow-lg)",
          transform: visible ? "translateX(0)" : "translateX(100%)",
          transition: "transform 280ms cubic-bezier(0.23, 1, 0.32, 1)",
        }}
      >
        {/* Header */}
        <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
          <CompanyLogo domain={company.domain} size={40} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {company.company_name}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--text-muted)" }}>
              {company.domain && (
                <a href={`https://${company.domain}`} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} style={{ color: "var(--blue)", display: "flex", alignItems: "center", gap: 3, textDecoration: "none" }}>
                  <Globe size={10} /> {company.domain}
                </a>
              )}
              {company.industry && <span>{company.industry}</span>}
            </div>
          </div>
          {totalSignals > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "3px 8px", background: "var(--accent-light)", borderRadius: 7 }}>
              <Newspaper size={11} style={{ color: "var(--accent)" }} />
              <span className="num" style={{ fontSize: 12, color: "var(--accent)" }}>{totalSignals}</span>
            </div>
          )}
          <button
            onClick={onClose}
            style={{ width: 28, height: 28, borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", flexShrink: 0 }}
          >
            <X size={13} />
          </button>
        </div>

        {/* Scrollable body */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          {/* ABOUT section — enriched data from web intelligence */}
          {(company.description || company.headquarters || company.employee_count || company.ceo) && (
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.06em", marginBottom: 8 }}>ABOUT</div>
              {company.description && (
                <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 10 }}>{company.description}</p>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 16px" }}>
                {company.industry && (
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>INDUSTRY</div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{company.industry}</div>
                  </div>
                )}
                {company.headquarters && (
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>HQ</div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 3 }}>
                      <MapPin size={10} /> {company.headquarters}
                    </div>
                  </div>
                )}
                {company.employee_count && (
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>EMPLOYEES</div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 3 }}>
                      <Users size={10} /> {company.employee_count}
                    </div>
                  </div>
                )}
                {company.founded_year && (
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>FOUNDED</div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 3 }}>
                      <Calendar size={10} /> {company.founded_year}
                    </div>
                  </div>
                )}
                {company.ceo && (
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>CEO</div>
                    <div style={{ fontSize: 11, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 3 }}>
                      <User size={10} /> {company.ceo}
                    </div>
                  </div>
                )}
                {company.stock_ticker && (
                  <div>
                    <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 2 }}>TICKER</div>
                    <div style={{ fontSize: 11, color: "var(--green)", fontWeight: 600 }}>{company.stock_ticker}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Enrichment fields */}
          <EnrichmentSection company={company} />

          {/* Reason */}
          {company.reason_relevant && (
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.06em", marginBottom: 6 }}>WHY RELEVANT</div>
              <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>{company.reason_relevant}</p>
            </div>
          )}

          {/* Live News */}
          {hasNews && (
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                <Newspaper size={11} /> Live News ({company.live_news!.length})
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {company.live_news!.map((n, i) => (
                  <a key={i} href={n.url} target="_blank" rel="noopener noreferrer" style={{ display: "block", padding: "8px 10px", background: "var(--surface-raised)", borderRadius: 7, border: "1px solid var(--border)", textDecoration: "none" }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                      <ExternalLink size={10} style={{ color: "var(--text-xmuted)", flexShrink: 0, marginTop: 2 }} />
                      <div>
                        <div style={{ fontSize: 11.5, fontWeight: 500, color: "var(--text)", lineHeight: 1.4, marginBottom: 3 }}>{n.title}</div>
                        {n.content && (
                          <div style={{ fontSize: 10.5, color: "var(--text-muted)", lineHeight: 1.5, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{n.content}</div>
                        )}
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* Cached Articles */}
          {hasArticles && (
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                <Clock size={11} /> Cached Intelligence ({company.recent_articles!.length})
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {company.recent_articles!.map((a, i) => (
                  <div key={i} style={{ padding: "8px 10px", background: "var(--surface-raised)", borderRadius: 7, border: "1px solid var(--border)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                      <span style={{ fontSize: 10, fontWeight: 600, color: "var(--accent)" }}>{a.source_name}</span>
                      {a.published_at && <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>{formatDate(a.published_at, "short")}</span>}
                    </div>
                    <div style={{ fontSize: 11, fontWeight: 500, color: "var(--text)", lineHeight: 1.4 }}>{a.title}</div>
                    {a.summary && (
                      <div style={{ fontSize: 10.5, color: "var(--text-muted)", lineHeight: 1.5, marginTop: 2, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{a.summary}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ChromaDB cached articles */}
          {chromaArticles.length > 0 && (
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <Newspaper size={11} /> Cached Intelligence ({chromaTotal})
                </div>
                <button
                  onClick={() => { router.push(`/companies/${company!.id}`); onClose(); }}
                  style={{ fontSize: 10, color: "var(--blue)", background: "none", border: "none", cursor: "pointer", fontWeight: 500 }}
                >
                  View all &rarr;
                </button>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {chromaArticles.slice(0, 5).map((a, i) => (
                  <div key={i} style={{ padding: "8px 10px", background: "var(--surface-raised)", borderRadius: 7, border: "1px solid var(--border)" }}>
                    <div style={{ fontSize: 11, fontWeight: 500, color: "var(--text)", lineHeight: 1.4 }}>{a.title}</div>
                    {a.summary && (
                      <div style={{ fontSize: 10.5, color: "var(--text-muted)", lineHeight: 1.5, marginTop: 2, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{a.summary}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No signals */}
          {!hasNews && !hasArticles && chromaArticles.length === 0 && (
            <div style={{ padding: "20px 16px", textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
              No news intelligence yet. Visit the full company page to collect articles.
            </div>
          )}
        </div>

        {/* Footer — View Full */}
        <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)", flexShrink: 0 }}>
          <button
            onClick={viewFull}
            style={{
              width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
              padding: "9px 16px", borderRadius: 8,
              border: "1px solid var(--border)", background: "var(--surface-raised)",
              cursor: "pointer", fontSize: 12, fontWeight: 500, color: "var(--text-secondary)",
              transition: "background 150ms, color 150ms",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)"; (e.currentTarget as HTMLElement).style.color = "var(--text)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)"; }}
          >
            View Full Company Page <ArrowRight size={13} />
          </button>
        </div>
      </div>
    </>
  );
}

// ── Enrichment fields (sub_industries, key_people, etc.) ────────


function EnrichmentRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, padding: "4px 0" }}>
      <span style={{ fontSize: 10, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: 11, color: "var(--text-secondary)", textAlign: "right" }}>{value}</span>
    </div>
  );
}

function EnrichmentSection({ company }: { company: CompanySearchResult }) {
  const hasFinancials = !!(company.revenue || company.total_funding);
  const hasLists = !!(
    (company.sub_industries?.length) ||
    (company.key_people?.length) ||
    (company.products_services?.length) ||
    (company.competitors?.length) ||
    (company.investors?.length) ||
    (company.tech_stack?.length)
  );

  if (!hasFinancials && !hasLists) return null;

  return (
    <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.06em", marginBottom: 8 }}>ENRICHMENT</div>

      {/* Financial row */}
      {hasFinancials && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px", marginBottom: hasLists ? 10 : 0 }}>
          {company.revenue && <EnrichmentRow label="REVENUE" value={company.revenue} />}
          {company.total_funding && <EnrichmentRow label="FUNDING" value={company.total_funding} />}
        </div>
      )}

      {/* Tag lists */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {(company.sub_industries?.length ?? 0) > 0 && (
          <div>
            <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 4 }}>SUB-INDUSTRIES</div>
            <InlineTagList items={company.sub_industries!} />
          </div>
        )}
        {(company.products_services?.length ?? 0) > 0 && (
          <div>
            <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 4 }}>PRODUCTS & SERVICES</div>
            <InlineTagList items={company.products_services!} />
          </div>
        )}
        {(company.key_people?.length ?? 0) > 0 && (
          <div>
            <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 4 }}>KEY PEOPLE</div>
            <InlineTagList items={company.key_people!.map(p => typeof p === 'string' ? p : `${p.name}${p.role ? ` (${p.role})` : ''}`)} color="var(--text)" />
          </div>
        )}
        {(company.competitors?.length ?? 0) > 0 && (
          <div>
            <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 4 }}>COMPETITORS</div>
            <InlineTagList items={company.competitors!} />
          </div>
        )}
        {(company.investors?.length ?? 0) > 0 && (
          <div>
            <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 4 }}>INVESTORS</div>
            <InlineTagList items={company.investors!} color="var(--green)" />
          </div>
        )}
        {(company.tech_stack?.length ?? 0) > 0 && (
          <div>
            <div style={{ fontSize: 9, color: "var(--text-xmuted)", fontWeight: 600, letterSpacing: "0.04em", marginBottom: 4 }}>TECH STACK</div>
            <InlineTagList items={company.tech_stack!} color="var(--blue)" />
          </div>
        )}
      </div>

      {company.validation_source && (
        <div style={{ marginTop: 6, fontSize: 9, color: "var(--text-xmuted)" }}>
          Source: {company.validation_source}
        </div>
      )}
    </div>
  );
}
