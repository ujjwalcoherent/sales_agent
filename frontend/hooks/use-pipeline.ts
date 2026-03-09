"use client";

import { useState, useCallback, useRef } from "react";
import { api } from "@/lib/api";
import type {
  PipelineStreamEvent,
  PipelineRunResult,
  PipelineStatus,
} from "@/lib/types";

export interface LogEntry {
  text: string;
  level: string; // "info" | "warning" | "error" | "debug"
}

export interface PipelineState {
  status: PipelineStatus;
  runId: string | null;
  progress: number;
  currentStep: string;
  messages: LogEntry[];
  result: PipelineRunResult | null;
  error: string | null;
  trendCount: number;
  companyCount: number;
  leadCount: number;
}

const INITIAL_STATE: PipelineState = {
  status: "idle",
  runId: null,
  progress: 0,
  currentStep: "",
  messages: [],
  result: null,
  error: null,
  trendCount: 0,
  companyCount: 0,
  leadCount: 0,
};

/** Read disabled providers from localStorage */
function getDisabledProviders(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem("harbinger_disabled_providers");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/** Read pipeline settings from localStorage */
function getPipelineOverrides(): { country?: string; max_trends?: number } {
  if (typeof window === "undefined") return {};
  const overrides: { country?: string; max_trends?: number } = {};
  const country = localStorage.getItem("harbinger_country");
  if (country) overrides.country = country;
  const maxTrends = localStorage.getItem("harbinger_max_trends");
  if (maxTrends) overrides.max_trends = parseInt(maxTrends, 10);
  return overrides;
}

/**
 * Core pipeline hook — handles real runs and mock replay via the same SSE path.
 *
 * mockMode=true triggers backend replay of a recorded pipeline run (~45s).
 * mockMode=false triggers a real pipeline run. The SSE format is identical
 * for both — the frontend code path is the same.
 */
export function usePipeline(onComplete?: (result: PipelineRunResult) => void) {
  const [state, setState] = useState<PipelineState>(INITIAL_STATE);
  const cleanupRef = useRef<(() => void) | null>(null);

  const run = useCallback(
    async (mockMode = false) => {
      cleanupRef.current?.();

      setState({
        ...INITIAL_STATE,
        status: "running",
        messages: [
          {
            text: mockMode ? "Starting mock replay..." : "Starting pipeline...",
            level: "info",
          },
        ],
      });

      try {
        const disabledProviders = getDisabledProviders();
        const overrides = getPipelineOverrides();
        const { run_id } = await api.runPipeline(mockMode, undefined, disabledProviders, overrides);
        setState((s) => ({ ...s, runId: run_id }));

        const cleanup = api.streamPipeline(
          run_id,
          (event: PipelineStreamEvent) => {
            if (event.event === "heartbeat") return;

            setState((s) => {
              const updates: Partial<PipelineState> = {};

              if (event.event === "progress") {
                updates.progress = event.progress_pct ?? s.progress;
                updates.currentStep = event.step ?? s.currentStep;
                updates.trendCount = event.trends ?? s.trendCount;
                updates.companyCount = event.companies ?? s.companyCount;
                updates.leadCount = event.leads ?? s.leadCount;
              }

              if (event.event === "log" && event.message) {
                const entry: LogEntry = { text: event.message, level: event.level ?? "info" };
                updates.messages = [...s.messages, entry].slice(-200);
              }

              return { ...s, ...updates };
            });
          },
          async () => {
            try {
              const result = await api.getPipelineResult(run_id);
              setState((s) => ({
                ...s,
                status: "complete",
                progress: 100,
                result,
              }));
              onComplete?.(result);
            } catch {
              setState((s) => ({ ...s, status: "complete", progress: 100 }));
            }
          },
          (err) =>
            setState((s) => ({ ...s, status: "error", error: err })),
        );

        cleanupRef.current = cleanup;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Pipeline failed";
        const isAlreadyRunning = msg.includes("409") || msg.toLowerCase().includes("already running");
        setState((s) => ({
          ...s,
          status: "error",
          error: isAlreadyRunning
            ? "Pipeline already running. Use Force Reset to cancel the stuck run."
            : msg,
        }));
      }
    },
    [onComplete],
  );

  const reset = useCallback(() => {
    cleanupRef.current?.();
    setState(INITIAL_STATE);
  }, []);

  const forceCancel = useCallback(async () => {
    cleanupRef.current?.();
    try {
      await api.cancelPipeline();
    } catch {
      // Even if cancel fails, reset local state
    }
    setState(INITIAL_STATE);
  }, []);

  return { ...state, run, reset, forceCancel };
}
