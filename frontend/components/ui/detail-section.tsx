import type { LucideIcon } from "lucide-react";

// ── Variant A: icon-based section (leads-panel, lead-detail-panel) ──────
// Uses an Icon component prop, padding + bottom border layout.

interface DetailSectionProps {
  label: string;
  icon?: LucideIcon;
  children: React.ReactNode;
}

export function DetailSection({ label, icon: Icon, children }: DetailSectionProps) {
  return (
    <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 8 }}>
        {Icon && <Icon size={11} style={{ color: "var(--text-xmuted)" }} />}
        <span style={{ fontSize: 10, fontWeight: 600, color: "var(--text-xmuted)", letterSpacing: "0.06em" }}>{label}</span>
      </div>
      {children}
    </div>
  );
}

// ── Variant B: trend-style section (trends/page.tsx) ────────────────────
// Uses ReactNode icon (allows inline JSX like <Target size={11} />), margin-based layout.

interface TrendSectionProps {
  label: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}

export function TrendSection({ label, icon, children }: TrendSectionProps) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}>
        {icon}
        <span style={{ fontSize: 10, color: "var(--text-xmuted)", letterSpacing: "0.07em", fontWeight: 600 }}>{label}</span>
      </div>
      {children}
    </div>
  );
}

// ── Variant C: history panel section (history/[id]/page.tsx) ────────────
// Centered label with decorative horizontal lines.

interface PanelSectionProps {
  title: string;
  children: React.ReactNode;
}

export function PanelSection({ title, children }: PanelSectionProps) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.09em", color: "var(--text-xmuted)", textTransform: "uppercase", marginBottom: 9, display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
        {title}
        <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
      </div>
      {children}
    </div>
  );
}
