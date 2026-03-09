"use client";

import { useState } from "react";
import { Building2 } from "lucide-react";

/**
 * Company logo with fallback chain: Google Favicon → logo.dev → Building2 icon.
 * Shared component used in pipeline cards, search results, detail pages.
 */
export function CompanyLogo({ domain, size = 36 }: { domain?: string; size?: number }) {
  const [level, setLevel] = useState<0 | 1 | 2>(0);

  if (!domain || level >= 2) {
    return (
      <div
        style={{
          width: size, height: size, borderRadius: 8,
          background: "var(--surface-raised)",
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0, border: "1px solid var(--border)",
        }}
      >
        <Building2 size={size * 0.44} style={{ color: "var(--text-secondary)" }} />
      </div>
    );
  }

  const srcs = [
    `https://www.google.com/s2/favicons?domain=${domain}&sz=128`,
    `https://img.logo.dev/${domain}?token=pk_anonymous&size=128&format=png`,
  ];

  return (
    <div
      style={{
        width: size, height: size, borderRadius: 8,
        background: "#fff",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0, border: "1px solid var(--border)", overflow: "hidden",
      }}
    >
      <img
        src={srcs[level]}
        alt=""
        width={size - 8}
        height={size - 8}
        onError={() => setLevel((l) => Math.min(l + 1, 2) as 0 | 1 | 2)}
        style={{ objectFit: "contain" }}
      />
    </div>
  );
}
