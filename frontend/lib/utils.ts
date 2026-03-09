import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// ── Shared lead/company UI helpers ────────────────────

/** Map confidence 0-1 to text color + background color */
export function confidenceColor(c: number): { text: string; bg: string } {
  if (c >= 0.75) return { text: "var(--green)",     bg: "var(--green-light)"   };
  if (c >= 0.50) return { text: "var(--accent)",    bg: "var(--amber-light)"   };
  return               { text: "var(--text-muted)", bg: "var(--surface-raised)" };
}

/** Lead type to badge CSS class mapping */
export const TYPE_CLASSES: Record<string, string> = {
  pain: "badge-red",
  opportunity: "badge-green",
  risk: "badge-amber",
  intelligence: "badge-blue",
};

/** Lead type to accent color + badge class mapping */
export const TYPE_COLORS: Record<string, { badge: string; accent: string }> = {
  pain:         { badge: "badge-red",   accent: "var(--red)"   },
  opportunity:  { badge: "badge-green", accent: "var(--green)" },
  risk:         { badge: "badge-amber", accent: "var(--amber)" },
  intelligence: { badge: "badge-blue",  accent: "var(--blue)"  },
};

// ── Text cleanup helpers ─────────────────────────

import type { LeadRecord } from "./types";

/** Clean up trigger_event: remove if it's just the trend_title repeated/truncated */
export function cleanTriggerEvent(lead: LeadRecord): string | null {
  if (!lead.trigger_event) return null;
  const trigger = lead.trigger_event.trim();
  const title = lead.trend_title.trim();
  if (trigger === title) return null;
  if (trigger.startsWith(title + " — " + title.substring(0, 20))) return null;
  if (trigger.startsWith(title + " —")) return null;
  return trigger;
}

/** Clean up opening_line: fix common template issues */
export function cleanOpeningLine(line: string): string {
  if (!line) return "";
  let cleaned = line;
  cleaned = cleaned.replace(/ for Your /g, " for your ");
  cleaned = cleaned.replace(/\.\./g, ".");
  cleaned = cleaned.replace(/^"?The recent (.+?) creates/, (match, title) => {
    const capitalized = title.charAt(0).toUpperCase() + title.slice(1);
    return match.replace(title, capitalized);
  });
  return cleaned;
}

// ── URL / display helpers ────────────────────────

/** Extract hostname from URL for display, e.g. "livemint.com" */
export function extractDomain(url: string): string {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch { return url.substring(0, 40); }
}

/** Parse a snippet "Title: body..." into parts */
export function parseSnippet(snippet: string): { title: string; body: string } {
  const colonIdx = snippet.indexOf(":");
  if (colonIdx > 0 && colonIdx < 120)
    return { title: snippet.substring(0, colonIdx).trim(), body: snippet.substring(colonIdx + 1).trim() };
  return { title: snippet.substring(0, 100), body: "" };
}

// ── Date formatting ──────────────────────────────

type DateStyle = "short" | "medium" | "long";

const DATE_FORMATS: Record<DateStyle, Intl.DateTimeFormatOptions> = {
  short:  { month: "short", day: "numeric" },
  medium: { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" },
  long:   { month: "short", day: "numeric", year: "numeric" },
};

/** Locale-aware date formatting — uses browser locale instead of hardcoded "en-IN" */
export function formatDate(dateStr: string | null | undefined, style: DateStyle = "short"): string {
  if (!dateStr) return "";
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return "";
    return d.toLocaleDateString(undefined, DATE_FORMATS[style]);
  } catch {
    return "";
  }
}
