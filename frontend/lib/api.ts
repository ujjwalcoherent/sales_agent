import type {
  LeadListResponse,
  RunPipelineResponse,
  PipelineRunResult,
  PipelineRunSummary,
  PipelineStreamEvent,
  HealthResponse,
  LearningStatus,
  EmailSettings,
  SendEmailResponse,
  CompanySearchResponse,
  CompanyNewsResponse,
  GenerateLeadsResponse,
  SavedCompany,
  SavedCompanyListResponse,
  Campaign,
  CampaignListResponse,
  CreateCampaignRequest,
  CampaignStreamEvent,
  PersonRecord,
  UserProfile,
  ProfileListResponse,
  CreateProfileRequest,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Core fetch wrapper ─────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Endpoints ──────────────────────────────────────

export const api = {
  /** GET /health — provider status + config */
  health(): Promise<HealthResponse> {
    return apiFetch("/health");
  },

  /** POST /api/v1/pipeline/run — start pipeline, returns run_id */
  runPipeline(
    mockMode = false,
    replayRunId?: string,
    disabledProviders?: string[],
    overrides?: { country?: string; max_trends?: number; mode?: string; industry?: string; companies?: string[] },
  ): Promise<RunPipelineResponse> {
    return apiFetch("/api/v1/pipeline/run", {
      method: "POST",
      body: JSON.stringify({
        mock_mode: mockMode,
        ...(replayRunId ? { replay_run_id: replayRunId } : {}),
        ...(disabledProviders?.length ? { disabled_providers: disabledProviders } : {}),
        ...(overrides?.country ? { country: overrides.country } : {}),
        ...(overrides?.max_trends ? { max_trends: overrides.max_trends } : {}),
        ...(overrides?.mode ? { mode: overrides.mode } : {}),
        ...(overrides?.industry ? { industry: overrides.industry } : {}),
        ...(overrides?.companies?.length ? { companies: overrides.companies } : {}),
      }),
    });
  },

  /** GET /api/v1/pipeline/result/{runId} — full results after completion */
  getPipelineResult(runId: string): Promise<PipelineRunResult> {
    return apiFetch(`/api/v1/pipeline/result/${runId}`);
  },

  /** GET /api/v1/pipeline/status/{runId} — polling fallback */
  getPipelineStatus(runId: string): Promise<PipelineRunSummary> {
    return apiFetch(`/api/v1/pipeline/status/${runId}`);
  },

  /** GET /api/v1/pipeline/runs — recent pipeline run history */
  getPipelineRuns(limit = 20): Promise<PipelineRunSummary[]> {
    return apiFetch(`/api/v1/pipeline/runs?limit=${limit}`);
  },

  /** GET /api/v1/leads — filtered lead list with pagination */
  getLeads(params?: {
    run_id?: string;
    hop?: number;
    lead_type?: string;
    min_confidence?: number;
    limit?: number;
    offset?: number;
  }): Promise<LeadListResponse> {
    const qs = new URLSearchParams();
    if (params?.run_id) qs.set("run_id", params.run_id);
    if (params?.hop !== undefined) qs.set("hop", String(params.hop));
    if (params?.lead_type) qs.set("lead_type", params.lead_type);
    if (params?.min_confidence !== undefined)
      qs.set("min_confidence", String(params.min_confidence));
    if (params?.limit !== undefined) qs.set("limit", String(params.limit));
    if (params?.offset !== undefined) qs.set("offset", String(params.offset));
    const query = qs.toString();
    return apiFetch(`/api/v1/leads${query ? `?${query}` : ""}`);
  },

  /** GET /api/v1/leads/latest — leads from most recent completed run */
  getLatestLeads(limit = 50): Promise<LeadListResponse> {
    return apiFetch(`/api/v1/leads/latest?limit=${limit}`);
  },

  /** GET /api/v1/leads/email/settings — email sending config (safe, no secrets) */
  getEmailSettings(): Promise<EmailSettings> {
    return apiFetch("/api/v1/leads/email/settings");
  },

  /** POST /api/v1/leads/{leadId}/enrich — find contacts + generate outreach */
  enrichLead(leadId: number): Promise<{
    success: boolean;
    contacts_found: number;
    outreach_generated: number;
    people: PersonRecord[];
    error: string;
  }> {
    return apiFetch(`/api/v1/leads/${leadId}/enrich`, { method: "POST" });
  },

  /** POST /api/v1/leads/{leadId}/send-email — send outreach email via Brevo */
  sendEmail(leadId: number, personIndex = 0): Promise<SendEmailResponse> {
    return apiFetch(`/api/v1/leads/${leadId}/send-email`, {
      method: "POST",
      body: JSON.stringify({ person_index: personIndex }),
    });
  },

  /** POST /settings — update runtime settings (all configurable fields) */
  updateSettings(settings: Record<string, unknown>): Promise<{ updated: Record<string, unknown>; current: Record<string, unknown> }> {
    return apiFetch("/settings", {
      method: "POST",
      body: JSON.stringify(settings),
    });
  },

  /** GET /api/v1/learning/status — all 6 self-learning loop states */
  getLearningStatus(): Promise<LearningStatus> {
    return apiFetch("/api/v1/learning/status");
  },

  /** POST /api/v1/feedback — submit rating */
  submitFeedback(payload: {
    feedback_type: "trend" | "lead";
    item_id: string;
    rating: string;
    metadata?: Record<string, unknown>;
  }): Promise<{ saved: boolean }> {
    return apiFetch("/api/v1/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  /** POST /api/v1/pipeline/cancel — force-cancel all active runs */
  cancelPipeline(): Promise<{ cancelled: number; message: string }> {
    return apiFetch("/api/v1/pipeline/cancel", { method: "POST" });
  },

  /** POST /api/v1/companies/search — company or industry search */
  searchCompanies(query: string): Promise<CompanySearchResponse> {
    return apiFetch("/api/v1/companies/search", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
  },

  /** POST /api/v1/companies/{companyId}/generate-leads — on-demand lead gen (long-running, 2-3 min) */
  async generateLeads(companyId: string, companyName: string, domain: string): Promise<GenerateLeadsResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 180_000); // 3 min
    try {
      return await apiFetch(`/api/v1/companies/${companyId}/generate-leads`, {
        method: "POST",
        body: JSON.stringify({ company_name: companyName, domain }),
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new Error("Lead generation timed out (3 min). The backend may still be processing — try refreshing in a minute.");
      }
      throw err;
    } finally {
      clearTimeout(timeout);
    }
  },

  /** GET /api/v1/companies/saved — all previously searched companies */
  getSavedCompanies(limit = 50): Promise<SavedCompanyListResponse> {
    return apiFetch(`/api/v1/companies/saved?limit=${limit}`);
  },

  /** GET /api/v1/companies/saved/{id} — single saved company with contacts */
  getSavedCompany(companyId: string): Promise<SavedCompany> {
    return apiFetch(`/api/v1/companies/saved/${companyId}`);
  },

  /** GET /api/v1/companies/{id}/news — paginated company news from ChromaDB */
  getCompanyNews(companyId: string, page = 1, perPage = 20): Promise<CompanyNewsResponse> {
    return apiFetch(`/api/v1/companies/${companyId}/news?page=${page}&per_page=${perPage}`);
  },

  /**
   * GET /api/v1/pipeline/stream/{runId}
   * Opens an SSE connection with auto-reconnect. Returns a cleanup function.
   *
   * Backend emits: progress | log | complete | error | heartbeat
   * Same SSE format for real pipeline runs and mock replay.
   * On disconnect, retries up to 3 times with 2s/4s/6s backoff.
   */
  streamPipeline(
    runId: string,
    onEvent: (event: PipelineStreamEvent) => void,
    onDone: () => void,
    onError: (err: string) => void,
  ): () => void {
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAYS = [2000, 4000, 6000];
    let currentEs: EventSource | null = null;
    let closed = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (closed) return;
      const es = new EventSource(`${API_BASE}/api/v1/pipeline/stream/${runId}`);
      currentEs = es;

      es.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data) as PipelineStreamEvent;
          retryCount = 0; // Reset on successful message
          onEvent(event);

          if (event.event === "complete") {
            es.close();
            onDone();
          } else if (event.event === "error") {
            es.close();
            onError(event.message ?? "Pipeline failed");
          }
        } catch {
          // ignore malformed frames
        }
      };

      es.onerror = () => {
        es.close();
        if (closed) return;
        if (retryCount < MAX_RETRIES) {
          const delay = RETRY_DELAYS[retryCount] ?? 6000;
          retryCount++;
          retryTimer = setTimeout(connect, delay);
        } else {
          onError("Stream disconnected after retries");
        }
      };
    }

    connect();

    return () => {
      closed = true;
      if (retryTimer) clearTimeout(retryTimer);
      currentEs?.close();
    };
  },

  // ── Campaigns ──────────────────────────────────

  /** POST /api/v1/campaigns — create a new campaign */
  createCampaign(req: CreateCampaignRequest): Promise<Campaign> {
    return apiFetch("/api/v1/campaigns/", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },

  /** GET /api/v1/campaigns — list all campaigns */
  listCampaigns(limit = 50): Promise<CampaignListResponse> {
    return apiFetch(`/api/v1/campaigns/?limit=${limit}`);
  },

  /** GET /api/v1/campaigns/{id} — get campaign details */
  getCampaign(campaignId: string): Promise<Campaign> {
    return apiFetch(`/api/v1/campaigns/${campaignId}`);
  },

  /** POST /api/v1/campaigns/{id}/run — start campaign execution */
  runCampaign(campaignId: string): Promise<{ status: string; campaign_id: string }> {
    return apiFetch(`/api/v1/campaigns/${campaignId}/run`, { method: "POST" });
  },

  /** PATCH /api/v1/campaigns/{id} — update a draft/failed campaign */
  updateCampaign(campaignId: string, updates: {
    name?: string;
    companies?: { company_name: string; domain?: string; industry?: string; context?: string }[];
    industry?: string;
    report_text?: string;
    config?: Partial<{ max_companies: number; max_contacts_per_company: number; generate_outreach: boolean }>;
  }): Promise<Campaign> {
    return apiFetch(`/api/v1/campaigns/${campaignId}`, {
      method: "PATCH",
      body: JSON.stringify(updates),
    });
  },

  /** DELETE /api/v1/campaigns/{id} — delete a campaign */
  deleteCampaign(campaignId: string): Promise<{ deleted: boolean }> {
    return apiFetch(`/api/v1/campaigns/${campaignId}`, { method: "DELETE" });
  },

  /** POST /api/v1/campaigns/{id}/send-email — send outreach email from campaign */
  sendCampaignEmail(campaignId: string, payload: {
    company_name: string;
    recipient_name: string;
    recipient_email: string;
    subject: string;
    body: string;
  }): Promise<{ success: boolean; message_id: string; recipient: string; error: string; test_mode: boolean }> {
    return apiFetch(`/api/v1/campaigns/${campaignId}/send-email`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  /** GET /api/v1/campaigns/{id}/export/csv — download URL for CSV export */
  getCampaignExportUrl(campaignId: string): string {
    return `${API_BASE}/api/v1/campaigns/${campaignId}/export/csv`;
  },

  /**
   * GET /api/v1/campaigns/{id}/stream — SSE for campaign progress.
   * Returns cleanup function. Same pattern as streamPipeline.
   */
  streamCampaign(
    campaignId: string,
    onEvent: (event: CampaignStreamEvent) => void,
    onDone: () => void,
    onError: (err: string) => void,
  ): () => void {
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAYS = [2000, 4000, 6000];
    let currentEs: EventSource | null = null;
    let closed = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (closed) return;
      const es = new EventSource(`${API_BASE}/api/v1/campaigns/${campaignId}/stream`);
      currentEs = es;

      es.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data) as CampaignStreamEvent;
          retryCount = 0;
          onEvent(event);

          if (event.event === "campaign_done") {
            es.close();
            onDone();
          } else if (event.event === "campaign_error") {
            es.close();
            onError(event.error ?? "Campaign failed");
          }
        } catch {
          // ignore malformed frames
        }
      };

      es.onerror = () => {
        es.close();
        if (closed) return;
        if (retryCount < MAX_RETRIES) {
          const delay = RETRY_DELAYS[retryCount] ?? 6000;
          retryCount++;
          retryTimer = setTimeout(connect, delay);
        } else {
          onError("Campaign stream disconnected after retries");
        }
      };
    }

    connect();

    return () => {
      closed = true;
      if (retryTimer) clearTimeout(retryTimer);
      currentEs?.close();
    };
  },

  /** GET /api/v1/settings/mock-mode — get global mock mode state */
  getMockMode(): Promise<MockModeState> {
    return apiFetch("/api/v1/settings/mock-mode");
  },

  /** POST /api/v1/settings/mock-mode — toggle global mock mode */
  setMockMode(enabled: boolean, recording?: string): Promise<MockModeState> {
    return apiFetch("/api/v1/settings/mock-mode", {
      method: "POST",
      body: JSON.stringify({ enabled, ...(recording ? { recording } : {}) }),
    });
  },
};

// ─── Profiles ────────────────────────────────────────────────────────────────

export async function listProfiles(): Promise<ProfileListResponse> {
  const data = await apiFetch<UserProfile[] | ProfileListResponse>("/api/v1/profiles")
  // Backend returns a plain array; normalise to { profiles, total }
  if (Array.isArray(data)) return { profiles: data, total: data.length }
  return data as ProfileListResponse
}

export async function createProfile(data: CreateProfileRequest): Promise<UserProfile> {
  return apiFetch("/api/v1/profiles", {
    method: "POST",
    body: JSON.stringify(data),
  })
}

export async function getProfile(profileId: string): Promise<UserProfile> {
  return apiFetch(`/api/v1/profiles/${profileId}`)
}

export async function updateProfile(profileId: string, data: Partial<CreateProfileRequest>): Promise<UserProfile> {
  return apiFetch(`/api/v1/profiles/${profileId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  })
}

export async function deleteProfile(profileId: string): Promise<void> {
  return apiFetch(`/api/v1/profiles/${profileId}`, { method: "DELETE" })
}

// ─── Global Mock Mode ─────────────────────────────────────────────────────────

export interface MockModeState {
  enabled: boolean
  selected_recording: string
  available_recordings: { run_id: string; files: number; has_leads: boolean; recommended: boolean }[]
  recommendation: string
}

export async function getMockMode(): Promise<MockModeState> {
  return apiFetch("/api/v1/settings/mock-mode")
}

export async function setMockMode(enabled: boolean, recording?: string): Promise<MockModeState> {
  return apiFetch("/api/v1/settings/mock-mode", {
    method: "POST",
    body: JSON.stringify({ enabled, ...(recording ? { recording } : {}) }),
  })
}
