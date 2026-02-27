// Types mirroring backend Pydantic schemas (app/api/schemas.py)

export type Severity = "high" | "medium" | "low" | "negligible";
export type LeadType = "pain" | "opportunity" | "risk" | "intelligence";
export type PipelineStatus = "idle" | "running" | "complete" | "error";

// ── Trend (matches backend TrendResponse) ────────

export interface TrendData {
  id: string;
  title: string;
  summary: string;
  severity: string;
  trend_type: string;
  industries: string[];
  keywords: string[];
  trend_score: number;
  actionability_score: number;
  oss_score: number;
  article_count: number;
  event_5w1h: Record<string, string>;
  causal_chain: string[];
  buying_intent: Record<string, string>;
  affected_companies: string[];
  actionable_insight: string;
  // Source articles (top evidence snippets from clustered articles)
  article_snippets: string[];
  source_links: string[];
  // Impact analysis (joined from ImpactAnalysis by trend_title)
  direct_impact: string[];
  indirect_impact: string[];
  midsize_pain_points: string[];
  target_roles: string[];
  pitch_angle: string;
  evidence_citations: string[];
  who_needs_help: string;
  council_confidence: number;
}

// ── Lead / Call Sheet (flat — matches backend LeadResponse) ──

export interface LeadRecord {
  id?: number;
  run_id?: string;
  company_name: string;
  company_cin: string;
  company_state: string;
  company_city: string;
  company_size_band: string;
  company_website: string;
  company_domain: string;
  reason_relevant: string;
  hop: number;
  lead_type: string;
  trend_title: string;
  event_type: string;
  // Contact enrichment (from Apollo/Hunter via lead_gen)
  contact_name: string;
  contact_role: string;
  contact_email: string;
  contact_linkedin: string;
  email_confidence: number;
  // Personalized outreach (from email_agent)
  email_subject: string;
  email_body: string;
  // Sales content
  trigger_event: string;
  pain_point: string;
  service_pitch: string;
  opening_line: string;
  urgency_weeks: number;
  confidence: number;
  oss_score: number;
  data_sources: string[];
  company_news: { title: string; url: string; date?: string }[];
}

export interface LeadListResponse {
  total: number;
  leads: LeadRecord[];
}

// ── Pipeline ─────────────────────────────────────

export interface PipelineRunResult {
  run_id: string;
  status: string;
  trends_detected: number;
  companies_found: number;
  leads_generated: number;
  run_time_seconds: number;
  errors: string[];
  trends: TrendData[];
  leads: LeadRecord[];
}

export interface PipelineRunSummary {
  run_id: string;
  status: string;
  current_step: string;
  progress_pct: number;
  trends_detected: number;
  companies_found: number;
  leads_generated: number;
  errors: string[];
  started_at: string;
  elapsed_seconds: number;
}

export interface PipelineStreamEvent {
  event: "progress" | "log" | "complete" | "error" | "heartbeat";
  step?: string;
  message?: string;
  level?: string;
  progress_pct?: number;
  trends?: number;
  companies?: number;
  leads?: number;
  summary?: {
    trends: number;
    companies: number;
    leads: number;
    runtime: number;
    replay?: boolean;
    original_run?: string;
  };
}

export interface RunPipelineResponse {
  run_id: string;
  status: string;
  message: string;
}

// ── Health ────────────────────────────────────────

export interface ProviderHealth {
  provider_name: string;
  failure_count: number;
  status: string; // "available" | "degraded" | "broken"
  last_failure_time: string | null;
  backoff_until: string | null;
  last_successful_call: string | null;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  providers: Record<string, ProviderHealth>;
  config: {
    country: string;
    max_trends: number;
    mock_mode: boolean;
    [key: string]: unknown;
  };
}

// ── Learning ─────────────────────────────────────

export interface LearningStatus {
  source_bandit: {
    total_arms?: number;
    top_sources?: { source: string; mean: number; pulls: number }[];
  };
  weight_learner: {
    weights?: Record<string, number>;
    data_count?: number;
  };
  company_bandit: {
    total_arms?: number;
    top_arms?: { arm: string; mean: number }[];
  };
  adaptive_thresholds: {
    thresholds?: Record<string, number>;
    update_count?: number;
  };
  trend_memory: {
    trend_count?: number;
    status?: string;
  };
  feedback: {
    total_records?: number;
    recent_100?: {
      auto: number;
      human: number;
      by_rating: Record<string, number>;
    };
  };
}

// ── Dashboard State ──────────────────────────────

export interface DashboardStats {
  leadsGenerated: number;
  trendsDetected: number;
  companiesFound: number;
  pipelineRuns: number;
}
