// ── Variant A: badge-based tag list (trends/page.tsx) ────────────────────
// Uses .badge CSS classes for pill styling.

interface BadgeTagListProps {
  items: string[];
  className?: string;
}

export function BadgeTagList({ items, className = "badge-amber" }: BadgeTagListProps) {
  const unique = [...new Set(items)];
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {unique.map((item, idx) => (
        <span key={`${item}-${idx}`} className={`badge ${className}`}>{item}</span>
      ))}
    </div>
  );
}

// ── Variant B: inline tag list (company-detail-panel.tsx) ────────────────
// Uses inline styles with border for enrichment data.

interface InlineTagListProps {
  items: string[];
  color?: string;
}

export function InlineTagList({ items, color }: InlineTagListProps) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {items.map((item) => (
        <span
          key={item}
          style={{
            fontSize: 10, padding: "2px 7px", borderRadius: 4,
            background: "var(--surface-raised)", color: color ?? "var(--text-secondary)",
            border: "1px solid var(--border)",
          }}
        >
          {item}
        </span>
      ))}
    </div>
  );
}
