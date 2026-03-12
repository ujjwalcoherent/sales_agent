// Types mirroring backend Pydantic schemas (app/api/schemas.py)

export type Severity = "high" | "medium" | "low" | "negligible";
export type TrendLevel = "major" | "sub" | "minor";
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
  trend_level: TrendLevel;
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

// ── Company news article (matches backend CompanyNewsArticle schema) ──

export interface CompanyNewsArticle {
  title: string;
  summary: string;
  url: string;
  source_name: string;
  published_at: string;
  sentiment_score?: number;
  source_type?: string;  // "cached" | "live"
}

// ── Person at a company (from people extraction pipeline) ──

export interface PersonRecord {
  person_name: string;
  role: string;
  seniority_tier: string; // "decision_maker" | "influencer" | "gatekeeper"
  linkedin_url: string;
  email: string;
  email_confidence: number;
  verified: boolean;
  reach_score: number;
  outreach_tone: string; // "executive" | "consultative" | "professional"
  outreach_subject: string;
  outreach_body: string;
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
  company_news?: CompanyNewsArticle[];
  // People extraction (tiered contacts with per-person outreach)
  people?: PersonRecord[];
}

// ── Email sending ────────────────────────────

export interface EmailSettings {
  sending_enabled: boolean;
  test_mode: boolean;
  test_recipient: string;
  brevo_configured: boolean;
  sender_email: string;
  sender_name: string;
}

export interface SendEmailResponse {
  success: boolean;
  message_id: string;
  recipient: string;
  subject: string;
  error: string;
  test_mode: boolean;
  sent_at: string;
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
    top_sources?: { source: string; mean: number; pulls: number; alpha?: number; beta?: number }[];
    arms?: Record<string, { mean?: number; alpha?: number; beta?: number; pulls?: number }>;
  };
  company_bandit: {
    total_arms?: number;
    top_arms?: { arm: string; mean: number; alpha?: number; beta?: number }[];
  };
  adaptive_thresholds: {
    thresholds?: Record<string, number>;
    update_count?: number;
  };
  trend_memory: {
    trend_count?: number;
    status?: string;
    collection?: string;
  };
  feedback: {
    total_records?: number;
    recent_100?: {
      auto: number;
      human: number;
      by_rating: Record<string, number>;
    };
  };
  signal_bus?: {
    system_confidence?: number;
    exploration_budget?: number;
    phase?: string;
    signal_count?: number;
    last_updated?: string;
  };
}

// ── Company Search ──────────────────────────────

export interface CompanySearchResult {
  id: string;
  company_name: string;
  domain: string;
  website: string;
  industry: string;
  description: string;
  headquarters: string;
  employee_count: string;
  founded_year: number | null;
  stock_ticker: string;
  ceo: string;
  funding_stage: string;
  reason_relevant: string;
  article_count: number;
  recent_articles: { title: string; summary: string; source_name: string; published_at: string; url: string }[];
  live_news: { title: string; url: string; content: string }[];
  // Enrichment fields from web intelligence
  sub_industries?: string[];
  key_people?: { name: string; role?: string; title?: string; linkedin_url?: string }[];
  products_services?: string[];
  competitors?: string[];
  revenue?: string;
  total_funding?: string;
  investors?: string[];
  tech_stack?: string[];
  validation_source?: string;
}

export interface CompanySearchResponse {
  companies: CompanySearchResult[];
  search_type: string;
  query: string;
  cached_articles_used: number;
  search_duration_ms: number;
}

export interface GenerateLeadsResponse {
  company_name: string;
  contacts: PersonRecord[];
  outreach_count: number;
  reasoning: string;
  duration_ms: number;
}

// ── Saved Company (DB-persisted search result with contacts) ──

export interface SavedCompany {
  id: string;
  company_name: string;
  domain: string;
  website: string;
  industry: string;
  description: string;
  headquarters: string;
  employee_count: string;
  founded_year: number | null;
  stock_ticker: string;
  ceo: string;
  funding_stage: string;
  wikidata_id: string;
  reason_relevant: string;
  article_count: number;
  recent_articles: { title: string; summary: string; source_name: string; published_at: string; url: string }[];
  live_news: { title: string; url: string; content: string }[];
  // Enrichment fields from web intelligence
  sub_industries?: string[];
  key_people?: { name: string; role?: string; title?: string; linkedin_url?: string }[];
  products_services?: string[];
  competitors?: string[];
  revenue?: string;
  total_funding?: string;
  investors?: string[];
  tech_stack?: string[];
  validation_source?: string;
  search_query: string;
  search_type: string;
  contacts: PersonRecord[];
  contacts_reasoning: string;
  contacts_generated_at: string | null;
  last_searched_at: string | null;
  created_at: string | null;
}

// ── Company News (on-demand from ChromaDB) ──────

export interface CompanyNewsResponse {
  articles: CompanyNewsArticle[];
  total: number;
  page: number;
  per_page: number;
  company_name: string;
}

export interface SavedCompanyListResponse {
  companies: SavedCompany[];
  total: number;
}

// ── Campaigns ────────────────────────────────────

export type CampaignType = "company_first" | "industry_first" | "report_driven";
export type CampaignStatus = "draft" | "running" | "completed" | "failed";

export interface CampaignCompanyInput {
  company_name: string;
  domain?: string;
  industry?: string;
  context?: string;
}

export interface CampaignConfig {
  max_companies: number;
  max_contacts_per_company: number;
  generate_outreach: boolean;
  target_roles: string[];
  country: string;
  background_deep: boolean;
  seniority_filter: "decision_maker" | "influencer" | "both";
  company_size_filter: "smb" | "mid_market" | "enterprise" | "all";
  trigger_signals: string[];
  product_context: string;
  narrow_keyword: string;
}

export interface CampaignContact {
  full_name: string;
  role: string;
  email: string;
  linkedin_url: string;
  seniority: string;
  email_confidence: number;
}

export interface CampaignEmail {
  recipient_name: string;
  recipient_role: string;
  subject: string;
  body: string;
}

export interface CampaignCompanyStatus {
  company_name: string;
  status: string;  // pending | enriching | contacts | outreach | done | failed
  domain: string;
  industry: string;
  description: string;
  contacts_found: number;
  outreach_generated: number;
  contacts: CampaignContact[];
  emails: CampaignEmail[];
  error: string;
}

export interface Campaign {
  id: string;
  name: string;
  campaign_type: CampaignType;
  status: CampaignStatus;
  companies: CampaignCompanyStatus[];
  total_companies: number;
  completed_companies: number;
  total_contacts: number;
  total_outreach: number;
  created_at: string;
  completed_at: string;
  error: string;
}

export interface CampaignListResponse {
  campaigns: Campaign[];
  total: number;
}

export interface CreateCampaignRequest {
  name?: string;
  campaign_type: CampaignType;
  companies?: CampaignCompanyInput[];
  industry?: string;
  report_text?: string;
  config?: Partial<CampaignConfig>;
}

export interface CampaignStreamEvent {
  event: "campaign_start" | "company_start" | "company_enriched" | "company_contacts" | "company_outreach" | "company_done" | "company_error" | "campaign_done" | "campaign_error" | "heartbeat";
  campaign_id?: string;
  company?: string;
  index?: number;
  total?: number;
  domain?: string;
  industry?: string;
  contacts_found?: number;
  outreach_generated?: number;
  total_contacts?: number;
  total_outreach?: number;
  error?: string;
}

// ─── User Profiles ──────────────────────────────────────────────────────────

export interface ProductEntry {
  name: string
  value_prop: string
  case_studies: string[]
  target_roles: string[]
  relevant_event_types: string[]
}

export interface IndustryTarget {
  industry_id: string
  display_name: string
  order: "1st" | "2nd" | "both"
  first_order_description: string
  second_order_description: string
  use_builtin: boolean
}

export interface ContactHierarchyEntry {
  event_type: string
  company_size: string
  role_priority: string[]
}

export interface EmailConfig {
  from_name: string
  from_email: string
}

export interface UserProfile {
  profile_id: string
  user_name: string
  own_company: string
  region: string
  own_products: ProductEntry[]
  target_industries: IndustryTarget[]
  account_list: string[]
  report_title: string
  report_summary: string
  contact_hierarchy: ContactHierarchyEntry[]
  min_lead_score: number
  email_config: EmailConfig
  path_preference: "auto" | "industry_first" | "company_first" | "report_driven"
}

export interface ProfileListResponse {
  profiles: UserProfile[]
  total: number
}

export type CreateProfileRequest = Omit<UserProfile, "profile_id">


