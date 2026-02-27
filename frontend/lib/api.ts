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
  runPipeline(mockMode = false, replayRunId?: string, disabledProviders?: string[]): Promise<RunPipelineResponse> {
    return apiFetch("/api/v1/pipeline/run", {
      method: "POST",
      body: JSON.stringify({
        mock_mode: mockMode,
        ...(replayRunId ? { replay_run_id: replayRunId } : {}),
        ...(disabledProviders?.length ? { disabled_providers: disabledProviders } : {}),
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

  /** POST /api/v1/leads/{leadId}/send-email — send outreach email via Brevo */
  sendEmail(leadId: number, personIndex = 0, dryRun = false): Promise<SendEmailResponse> {
    return apiFetch(`/api/v1/leads/${leadId}/send-email`, {
      method: "POST",
      body: JSON.stringify({ person_index: personIndex, dry_run: dryRun }),
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

  /**
   * GET /api/v1/pipeline/stream/{runId}
   * Opens an SSE connection. Returns a cleanup function.
   *
   * Backend emits: progress | log | complete | error | heartbeat
   * Same SSE format for real pipeline runs and mock replay.
   */
  streamPipeline(
    runId: string,
    onEvent: (event: PipelineStreamEvent) => void,
    onDone: () => void,
    onError: (err: string) => void,
  ): () => void {
    const es = new EventSource(`${API_BASE}/api/v1/pipeline/stream/${runId}`);

    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as PipelineStreamEvent;
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
      onError("Stream disconnected");
    };

    return () => es.close();
  },
};
