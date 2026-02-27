"use client";

import { useState, useEffect } from "react";
import {
  RefreshCw, Brain, Target, Gauge, Database, MessageSquare, Building2,
  TrendingUp, TrendingDown, Minus, Info, Activity, Zap, Shield,
} from "lucide-react";
import { api } from "@/lib/api";
import type { LearningStatus } from "@/lib/types";

export default function LearningPage() {
  const [status, setStatus] = useState<LearningStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = () => {
    setLoading(true);
    api.getLearningStatus()
      .then(setStatus)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { refresh(); }, []);

  return (
    <>
      {/* Header */}
      <div style={{ padding: "14px 24px", display: "flex", alignItems: "center", gap: 10, borderBottom: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
        <h1 className="font-display" style={{ fontSize: 19, color: "var(--text)", letterSpacing: "-0.02em" }}>
          Learning System
        </h1>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>6 self-learning loops</span>
        <button
          onClick={refresh}
          disabled={loading}
          style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 5, padding: "5px 12px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", cursor: loading ? "not-allowed" : "pointer", fontSize: 12, color: "var(--text-secondary)", opacity: loading ? 0.6 : 1 }}
        >
          <RefreshCw size={12} style={{ animation: loading ? "spin 1s linear infinite" : "none" }} />
          Refresh
        </button>
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>
        {/* Explanation banner */}
        <div style={{ padding: "12px 16px", background: "var(--accent-light)", borderRadius: 8, border: "1px solid var(--accent)", marginBottom: 18, display: "flex", gap: 10, alignItems: "flex-start" }}>
          <Info size={14} style={{ color: "var(--accent)", marginTop: 2, flexShrink: 0 }} />
          <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            The learning system improves itself after every pipeline run. It tracks which news sources produce the best leads,
            which company segments respond best, and tunes signal weights automatically.
            <strong style={{ color: "var(--text)" }}> More pipeline runs = smarter system.</strong>
          </div>
        </div>

        {!status ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="card" style={{ padding: 20 }}>
                <div className="skeleton" style={{ height: 14, width: "50%", marginBottom: 12 }} />
                <div className="skeleton" style={{ height: 80, width: "100%" }} />
              </div>
            ))}
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <SourceBanditCard data={status.source_bandit} />
            <WeightLearnerCard data={status.weight_learner} />
            <CompanyBanditCard data={status.company_bandit} />
            <AdaptiveThresholdsCard data={status.adaptive_thresholds} />
            <TrendMemoryCard data={status.trend_memory} />
            <FeedbackCard data={status.feedback} />
          </div>
        )}
      </div>
    </>
  );
}

// ── Loop Card Shell ─────────────────────────────────────────────────────────

function LoopCard({
  title,
  icon: Icon,
  accent,
  description,
  healthLabel,
  healthColor,
  children,
}: {
  title: string;
  icon: React.ElementType;
  accent: string;
  description: string;
  healthLabel?: string;
  healthColor?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card" style={{ padding: 0, overflow: "hidden" }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 28, height: 28, borderRadius: 7, background: "var(--surface-raised)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <Icon size={13} style={{ color: accent }} />
        </div>
        <div style={{ flex: 1 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{title}</span>
          <div style={{ fontSize: 10, color: "var(--text-xmuted)", lineHeight: 1.3 }}>{description}</div>
        </div>
        {healthLabel && (
          <span className="badge" style={{ fontSize: 9, background: healthColor === "green" ? "var(--green-light)" : healthColor === "amber" ? "var(--amber-light)" : "var(--surface-raised)", color: healthColor === "green" ? "var(--green)" : healthColor === "amber" ? "var(--amber)" : "var(--text-muted)" }}>
            {healthLabel}
          </span>
        )}
      </div>
      <div style={{ padding: "14px 16px" }}>{children}</div>
    </div>
  );
}

// ── Source Bandit ─────────────────────────────────────────────────────────

function SourceBanditCard({ data }: { data: LearningStatus["source_bandit"] }) {
  const sources = data.top_sources ?? [];
  const totalArms = data.total_arms ?? 0;
  const topMean = sources.length > 0 ? sources[0].mean : 0;
  const health = totalArms > 30 ? "green" : totalArms > 10 ? "amber" : undefined;
  const healthLabel = totalArms > 30 ? "Exploring" : totalArms > 10 ? "Learning" : undefined;

  return (
    <LoopCard
      title="Source Bandit"
      icon={Target}
      accent="var(--accent)"
      description="Ranks news sources by lead quality using Thompson Sampling"
      healthLabel={healthLabel || `${totalArms} arms`}
      healthColor={health}
    >
      {/* KPI row */}
      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
        <KpiMini label="Sources tracked" value={totalArms} color="var(--accent)" />
        <KpiMini label="Top source score" value={`${(topMean * 100).toFixed(0)}%`} color="var(--green)" />
        <KpiMini label="Total pulls" value={sources.reduce((s, x) => s + (x.pulls || 0), 0)} color="var(--text-secondary)" />
      </div>

      {sources.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.05em", marginBottom: 2 }}>TOP PERFORMING SOURCES</div>
          {sources.slice(0, 8).map((s, i) => {
            const barWidth = topMean > 0 ? (s.mean / topMean) * 100 : 0;
            return (
              <div key={s.source} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="num" style={{ fontSize: 10, color: "var(--text-xmuted)", minWidth: 14, textAlign: "right" }}>
                  {i + 1}.
                </span>
                <span style={{ fontSize: 11, color: i < 3 ? "var(--text)" : "var(--text-secondary)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: i < 3 ? 500 : 400 }}>
                  {s.source}
                </span>
                <div style={{ width: 80, height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${barWidth}%`, background: i === 0 ? "var(--green)" : "var(--accent)", borderRadius: 3, transition: "width 300ms" }} />
                </div>
                <span className="num" style={{ fontSize: 10, color: i === 0 ? "var(--green)" : "var(--text-muted)", minWidth: 32, textAlign: "right" }}>
                  {(s.mean * 100).toFixed(1)}%
                </span>
                <span style={{ fontSize: 9, color: "var(--text-xmuted)", minWidth: 20, textAlign: "right" }}>
                  {s.pulls || 0}x
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <EmptyState>No data yet — run a pipeline first</EmptyState>
      )}
    </LoopCard>
  );
}

// ── Weight Learner ─────────────────────────────────────────────────────────

function WeightLearnerCard({ data }: { data: LearningStatus["weight_learner"] }) {
  const weights = data.weights ?? {};
  const categories = Object.entries(weights);
  const dataCount = data.data_count ?? 0;
  const health = dataCount > 50 ? "green" : dataCount > 10 ? "amber" : undefined;

  return (
    <LoopCard
      title="Weight Learner"
      icon={Gauge}
      accent="var(--blue)"
      description="Tunes signal weights based on pipeline feedback"
      healthLabel={dataCount > 0 ? `${dataCount} samples` : undefined}
      healthColor={health}
    >
      {categories.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {categories.map(([category, subWeights]) => {
            if (typeof subWeights === "number") {
              return <WeightRow key={category} label={category} value={subWeights} />;
            }
            const entries = Object.entries(subWeights as Record<string, number>);
            const total = entries.reduce((s, [, v]) => s + (typeof v === "number" ? v : 0), 0);
            return (
              <div key={category}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)", textTransform: "capitalize" }}>
                    {category.replace(/_/g, " ")}
                  </span>
                  <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>
                    ({entries.length} signals, total {total.toFixed(2)})
                  </span>
                </div>
                {/* Stacked bar for this category */}
                <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", marginBottom: 6, background: "var(--border)" }}>
                  {entries.map(([key, val], idx) => {
                    const pct = total > 0 ? ((typeof val === "number" ? val : 0) / total) * 100 : 0;
                    const colors = ["var(--blue)", "var(--accent)", "var(--green)", "var(--amber)", "var(--red)", "var(--text-muted)"];
                    return (
                      <div
                        key={key}
                        title={`${key}: ${typeof val === "number" ? val.toFixed(3) : val}`}
                        style={{ width: `${pct}%`, background: colors[idx % colors.length], minWidth: pct > 0 ? 2 : 0, transition: "width 300ms" }}
                      />
                    );
                  })}
                </div>
                {/* Legend */}
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {entries.map(([key, val], idx) => {
                    const colors = ["var(--blue)", "var(--accent)", "var(--green)", "var(--amber)", "var(--red)", "var(--text-muted)"];
                    return (
                      <div key={key} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                        <div style={{ width: 6, height: 6, borderRadius: 2, background: colors[idx % colors.length] }} />
                        <span style={{ fontSize: 9, color: "var(--text-muted)" }}>{key.replace(/_/g, " ")}</span>
                        <span className="num" style={{ fontSize: 9, color: "var(--text-xmuted)" }}>{typeof val === "number" ? val.toFixed(2) : String(val)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <EmptyState>No learned weights yet</EmptyState>
      )}
    </LoopCard>
  );
}

function WeightRow({ label, value }: { label: string; value: number }) {
  const pct = Math.min(value * 100, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontSize: 11, color: "var(--text-secondary)", flex: 1, textTransform: "capitalize" }}>
        {label.replace(/_/g, " ")}
      </span>
      <div style={{ width: 80, height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: "var(--blue)", borderRadius: 3 }} />
      </div>
      <span className="num" style={{ fontSize: 10, color: "var(--text-muted)", minWidth: 32, textAlign: "right" }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

// ── Company Bandit ─────────────────────────────────────────────────────────

function CompanyBanditCard({ data }: { data: LearningStatus["company_bandit"] }) {
  const arms = data.top_arms ?? [];
  const totalArms = data.total_arms ?? 0;
  const topMean = arms.length > 0 ? arms[0].mean : 0;

  return (
    <LoopCard
      title="Company Bandit"
      icon={Building2}
      accent="var(--green)"
      description="Learns which company sizes + event types produce best leads"
      healthLabel={totalArms > 0 ? `${totalArms} arms` : undefined}
      healthColor={totalArms > 20 ? "green" : "amber"}
    >
      {/* KPI */}
      <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
        <KpiMini label="Segment arms" value={totalArms} color="var(--green)" />
        <KpiMini label="Top arm score" value={`${(topMean * 100).toFixed(0)}%`} color="var(--green)" />
      </div>

      {arms.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.05em", marginBottom: 2 }}>BEST (SIZE, EVENT) COMBINATIONS</div>
          {arms.slice(0, 8).map((a, i) => {
            const barWidth = topMean > 0 ? (a.mean / topMean) * 100 : 0;
            // Parse arm name: "mid_regulation" → "Mid · Regulation"
            const parts = a.arm.split("_");
            const displayArm = parts.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(" · ");
            return (
              <div key={a.arm} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="num" style={{ fontSize: 10, color: "var(--text-xmuted)", minWidth: 14, textAlign: "right" }}>{i + 1}.</span>
                <span style={{ fontSize: 11, color: i < 3 ? "var(--text)" : "var(--text-secondary)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: i < 3 ? 500 : 400 }}>
                  {displayArm}
                </span>
                <div style={{ width: 80, height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${barWidth}%`, background: i === 0 ? "var(--green)" : "var(--green)", opacity: i === 0 ? 1 : 0.6, borderRadius: 3 }} />
                </div>
                <span className="num" style={{ fontSize: 10, color: i === 0 ? "var(--green)" : "var(--text-muted)", minWidth: 32, textAlign: "right" }}>
                  {(a.mean * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <EmptyState>No arms explored yet</EmptyState>
      )}
    </LoopCard>
  );
}

// ── Adaptive Thresholds ─────────────────────────────────────────────────────

const THRESHOLD_INFO: Record<string, { label: string; unit: string; ideal: string }> = {
  coherence_min: { label: "Coherence minimum", unit: "", ideal: "0.40 - 0.60" },
  merge_threshold: { label: "Cluster merge distance", unit: "", ideal: "0.70 - 0.85" },
  min_synthesis_confidence: { label: "Synthesis confidence floor", unit: "", ideal: "0.35 - 0.50" },
  min_signal_score: { label: "Signal score cutoff", unit: "", ideal: "0.30 - 0.50" },
};

function AdaptiveThresholdsCard({ data }: { data: LearningStatus["adaptive_thresholds"] }) {
  const thresholds = data.thresholds ?? {};
  const entries = Object.entries(thresholds);
  const updateCount = data.update_count ?? 0;

  return (
    <LoopCard
      title="Adaptive Thresholds"
      icon={Shield}
      accent="var(--amber)"
      description="Auto-tunes quality gates based on pipeline outcomes"
      healthLabel={updateCount > 0 ? `${updateCount} adjustments` : undefined}
      healthColor={updateCount > 5 ? "green" : "amber"}
    >
      {entries.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {entries.map(([key, val]) => {
            const info = THRESHOLD_INFO[key] || { label: key.replace(/_/g, " "), unit: "", ideal: "—" };
            const numVal = typeof val === "number" ? val : 0;
            // Visual: position on a 0-1 scale
            const pct = Math.min(numVal * 100, 100);
            return (
              <div key={key}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
                  <span style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "capitalize" }}>{info.label}</span>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
                    <span className="num" style={{ fontSize: 14, color: "var(--amber)", lineHeight: 1 }}>{numVal.toFixed(3)}</span>
                    <span style={{ fontSize: 9, color: "var(--text-xmuted)" }}>ideal: {info.ideal}</span>
                  </div>
                </div>
                <div style={{ width: "100%", height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden", position: "relative" }}>
                  <div style={{ position: "absolute", height: "100%", width: `${pct}%`, background: "var(--amber)", borderRadius: 3, opacity: 0.7, transition: "width 300ms" }} />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <EmptyState>No threshold data yet</EmptyState>
      )}
    </LoopCard>
  );
}

// ── Trend Memory ─────────────────────────────────────────────────────────

function TrendMemoryCard({ data }: { data: LearningStatus["trend_memory"] }) {
  const count = data.trend_count ?? 0;
  const isAvailable = data.status !== "unavailable";

  return (
    <LoopCard
      title="Trend Memory"
      icon={Brain}
      accent="var(--accent)"
      description="ChromaDB deduplication — prevents reporting same trends twice"
      healthLabel={isAvailable ? (count > 0 ? "Active" : "Empty") : "Offline"}
      healthColor={isAvailable && count > 0 ? "green" : "amber"}
    >
      {isAvailable ? (
        <div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 8 }}>
            <span className="num" style={{ fontSize: 36, color: "var(--text)", lineHeight: 1 }}>{count}</span>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>trends in memory</span>
          </div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, padding: "8px 10px", background: "var(--surface-raised)", borderRadius: 7 }}>
            <Zap size={10} style={{ color: "var(--accent)", display: "inline", verticalAlign: "middle", marginRight: 4 }} />
            Each new pipeline run checks incoming trends against this memory. Duplicate or near-duplicate trends are filtered out,
            ensuring only <strong>genuinely new</strong> market signals surface.
          </div>
        </div>
      ) : (
        <div>
          <EmptyState>ChromaDB not available</EmptyState>
          <div style={{ fontSize: 11, color: "var(--text-xmuted)", marginTop: 4 }}>
            Trend memory requires ChromaDB. The system will still work but may surface repeated trends.
          </div>
        </div>
      )}
    </LoopCard>
  );
}

// ── Feedback Loop ─────────────────────────────────────────────────────────

function FeedbackCard({ data }: { data: LearningStatus["feedback"] }) {
  const recent = data.recent_100;
  const total = data.total_records ?? 0;

  return (
    <LoopCard
      title="Feedback Loop"
      icon={MessageSquare}
      accent="var(--green)"
      description="Human + auto ratings that guide all other learning loops"
      healthLabel={total > 0 ? `${total} records` : undefined}
      healthColor={total > 100 ? "green" : total > 0 ? "amber" : undefined}
    >
      {recent ? (
        <div>
          {/* Stats row */}
          <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
            <KpiMini label="Total" value={total} color="var(--text)" />
            <KpiMini label="Human" value={recent.human} color="var(--green)" />
            <KpiMini label="Auto" value={recent.auto} color="var(--text-secondary)" />
            <KpiMini
              label="Auto %"
              value={`${total > 0 ? Math.round((recent.auto / (recent.auto + recent.human || 1)) * 100) : 0}%`}
              color="var(--text-muted)"
            />
          </div>

          {/* Rating breakdown */}
          {Object.entries(recent.by_rating).length > 0 && (
            <div>
              <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.05em", marginBottom: 6 }}>RATING DISTRIBUTION (last 100)</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {Object.entries(recent.by_rating)
                  .sort(([, a], [, b]) => b - a)
                  .map(([rating, count]) => {
                    const pct = Math.round((count / 100) * 100);
                    const ratingColor = rating === "good" || rating === "useful" ? "var(--green)" : rating === "bad" || rating === "irrelevant" ? "var(--red)" : "var(--text-muted)";
                    return (
                      <div key={rating} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 11, color: "var(--text-secondary)", minWidth: 60, textTransform: "capitalize" }}>
                          {rating}
                        </span>
                        <div style={{ flex: 1, height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${pct}%`, background: ratingColor, borderRadius: 3, opacity: 0.7 }} />
                        </div>
                        <span className="num" style={{ fontSize: 10, color: "var(--text-muted)", minWidth: 24, textAlign: "right" }}>
                          {count}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      ) : (
        <EmptyState>No feedback recorded yet</EmptyState>
      )}
    </LoopCard>
  );
}

// ── Shared Components ─────────────────────────────────────────────────────

function KpiMini({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div style={{ textAlign: "center", minWidth: 50 }}>
      <div className="num" style={{ fontSize: 18, color, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 9, color: "var(--text-xmuted)", marginTop: 2 }}>{label}</div>
    </div>
  );
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ padding: "12px 0", fontSize: 12, color: "var(--text-xmuted)", fontStyle: "italic" }}>
      {children}
    </div>
  );
}
