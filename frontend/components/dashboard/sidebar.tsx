"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Zap, ChevronRight, Sun, Moon, Settings } from "lucide-react";
import { NAV_ITEMS, APP_NAME, APP_TAGLINE } from "@/lib/config";

export function Sidebar() {
  const pathname = usePathname();
  const [dark, setDark] = useState(false);

  // Restore dark mode from localStorage (inline script already applied the class)
  useEffect(() => {
    const stored = localStorage.getItem("harbinger_dark");
    setDark(stored === "true" || (!stored && document.documentElement.classList.contains("dark")));
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("harbinger_dark", String(dark));
  }, [dark]);

  const isSettings = pathname === "/settings" || pathname.startsWith("/settings/");

  return (
    <aside
      style={{
        width: 220,
        minWidth: 220,
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        padding: "22px 12px",
        gap: 4,
        height: "100vh",
        position: "sticky",
        top: 0,
        overflowY: "auto",
      }}
    >
      {/* Logo */}
      <div style={{ padding: "0 6px 20px" }}>
        <Link
          href="/"
          style={{ display: "flex", alignItems: "center", gap: 8, textDecoration: "none" }}
        >
          <div
            style={{
              width: 28,
              height: 28,
              background: "var(--text)",
              borderRadius: 6,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Zap size={14} color="#F8F7F2" strokeWidth={2.5} />
          </div>
          <div>
            <div className="font-display" style={{ fontSize: 15, color: "var(--text)", lineHeight: 1.2 }}>
              {APP_NAME}
            </div>
            <div style={{ fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.05em" }}>
              {APP_TAGLINE}
            </div>
          </div>
        </Link>
      </div>

      {/* Nav */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
        <div
          style={{
            fontSize: 10,
            color: "var(--text-xmuted)",
            letterSpacing: "0.08em",
            fontWeight: 600,
            padding: "0 10px",
            marginBottom: 4,
          }}
        >
          WORKSPACE
        </div>

        {NAV_ITEMS.map(({ id, label, href, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={id}
              href={href}
              className={`nav-item ${isActive ? "active" : ""}`}
            >
              <Icon
                size={15}
                style={{ color: isActive ? "var(--text)" : "var(--text-muted)", flexShrink: 0 }}
              />
              <span>{label}</span>
              {isActive && (
                <ChevronRight size={12} style={{ marginLeft: "auto", color: "var(--text-muted)" }} />
              )}
            </Link>
          );
        })}
      </div>

      {/* Bottom section: dark mode + settings */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "0 2px" }}>
        <button
          onClick={() => setDark((v) => !v)}
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            gap: 7,
            padding: "6px 8px",
            borderRadius: 7,
            border: "1px solid var(--border)",
            background: "var(--surface-raised)",
            cursor: "pointer",
            fontSize: 12,
            color: "var(--text-muted)",
            transition: "background 150ms, color 150ms",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-hover)"; (e.currentTarget as HTMLElement).style.color = "var(--text)"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "var(--surface-raised)"; (e.currentTarget as HTMLElement).style.color = "var(--text-muted)"; }}
        >
          {dark ? <Sun size={13} /> : <Moon size={13} />}
          {dark ? "Light mode" : "Dark mode"}
        </button>

        <Link
          href="/settings"
          className={`nav-item ${isSettings ? "active" : ""}`}
          style={{ borderRadius: 7 }}
        >
          <Settings
            size={15}
            style={{ color: isSettings ? "var(--text)" : "var(--text-muted)", flexShrink: 0 }}
          />
          <span>Settings</span>
          {isSettings && (
            <ChevronRight size={12} style={{ marginLeft: "auto", color: "var(--text-muted)" }} />
          )}
        </Link>
      </div>
    </aside>
  );
}
