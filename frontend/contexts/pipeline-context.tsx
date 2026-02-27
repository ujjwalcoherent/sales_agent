"use client";

import { createContext, useContext, useState, useCallback, useEffect } from "react";
import { usePipeline, type LogEntry } from "@/hooks/use-pipeline";
import { api } from "@/lib/api";
import type { LeadRecord, TrendData, PipelineRunResult } from "@/lib/types";

export interface PipelineContextValue {
  status: "idle" | "running" | "complete" | "error";
  runId: string | null;
  progress: number;
  currentStep: string;
  messages: LogEntry[];
  result: PipelineRunResult | null;
  error: string | null;
  trendCount: number;
  companyCount: number;
  leadCount: number;
  leads: LeadRecord[];
  trends: TrendData[];
  lastRunTime: string | null;
  initialLoading: boolean;
  run: (mockMode?: boolean) => void;
  reset: () => void;
}

const PipelineContext = createContext<PipelineContextValue | null>(null);

export function PipelineProvider({ children }: { children: React.ReactNode }) {
  const [leads, setLeads] = useState<LeadRecord[]>([]);
  const [trends, setTrends] = useState<TrendData[]>([]);
  const [lastRunTime, setLastRunTime] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);

  const onComplete = useCallback((result: PipelineRunResult) => {
    setLastRunTime(new Date().toLocaleTimeString());
    if (result.leads.length > 0) setLeads(result.leads);
    if (result.trends.length > 0) setTrends(result.trends);
  }, []);

  const pipeline = usePipeline(onComplete);

  // Load latest data from API on mount
  useEffect(() => {
    (async () => {
      try {
        const { leads: freshLeads } = await api.getLatestLeads();
        if (freshLeads.length > 0) setLeads(freshLeads);
      } catch {
        /* no data yet */
      }
      try {
        const runs = await api.getPipelineRuns(1);
        if (runs.length > 0 && runs[0].status === "completed") {
          const result = await api.getPipelineResult(runs[0].run_id);
          if (result.trends.length > 0) setTrends(result.trends);
        }
      } catch {
        /* no runs yet */
      }
      setInitialLoading(false);
    })();
  }, []);

  const value: PipelineContextValue = {
    ...pipeline,
    leads,
    trends,
    lastRunTime,
    initialLoading,
  };

  return (
    <PipelineContext.Provider value={value}>
      {children}
    </PipelineContext.Provider>
  );
}

export function usePipelineContext(): PipelineContextValue {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error("usePipelineContext must be used within <PipelineProvider>");
  return ctx;
}
