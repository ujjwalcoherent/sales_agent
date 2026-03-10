"use client";

import { useState, useEffect } from "react";
import { Plus, Edit2, Trash2, User } from "lucide-react";
import { listProfiles, deleteProfile } from "@/lib/api";
import type { UserProfile } from "@/lib/types";
import ProfileWizard from "./_components/ProfileWizard";

// ── Path badge config ──────────────────────────────────────────────────────

const PATH_BADGE: Record<
  UserProfile["path_preference"],
  { label: string; bg: string; color: string }
> = {
  auto:           { label: "Auto",           bg: "var(--accent-light)",  color: "var(--accent)" },
  industry_first: { label: "Industry-First", bg: "var(--green-light)",   color: "var(--green)"  },
  company_first:  { label: "Company-First",  bg: "#EBF2F9",              color: "var(--blue)"   },
  report_driven:  { label: "Report-Driven",  bg: "#EDE8F5",              color: "#6B3FA0"        },
};

// ── Loading skeleton ──────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="card" style={{ padding: "18px 20px" }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 14 }}>
        <div className="skeleton" style={{ width: 44, height: 44, borderRadius: "50%" }} />
        <div style={{ flex: 1 }}>
          <div className="skeleton" style={{ height: 13, width: "55%", marginBottom: 6 }} />
          <div className="skeleton" style={{ height: 10, width: "40%" }} />
        </div>
      </div>
      <div className="skeleton" style={{ height: 10, width: "70%", marginBottom: 8 }} />
      <div className="skeleton" style={{ height: 10, width: "50%", marginBottom: 16 }} />
      <div style={{ display: "flex", gap: 8 }}>
        <div className="skeleton" style={{ flex: 1, height: 30, borderRadius: 7 }} />
        <div className="skeleton" style={{ width: 80, height: 30, borderRadius: 7 }} />
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function ProfilesPage() {
  const [profiles, setProfiles] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // wizard: undefined = closed, null = new, UserProfile = edit
  const [wizardProfile, setWizardProfile] = useState<UserProfile | null | undefined>(undefined);

  // delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => { load(); }, []);

  async function load() {
    setLoading(true);
    try {
      const res = await listProfiles();
      setProfiles(res.profiles);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load profiles");
    } finally {
      setLoading(false);
    }
  }

  function handleSaved(saved: UserProfile) {
    setProfiles((prev) => {
      const idx = prev.findIndex((p) => p.profile_id === saved.profile_id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = saved;
        return next;
      }
      return [saved, ...prev];
    });
    setWizardProfile(undefined);
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    const target = deleteTarget;
    setDeleting(target.id);
    setDeleteTarget(null);
    try {
      await deleteProfile(target.id);
      setProfiles((prev) => prev.filter((p) => p.profile_id !== target.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  }

  return (
    <>
      {/* ── Wizard overlay ── */}
      {wizardProfile !== undefined && (
        <ProfileWizard
          profile={wizardProfile}
          onClose={() => setWizardProfile(undefined)}
          onSaved={handleSaved}
        />
      )}

      {/* ── Delete confirm modal ── */}
      {deleteTarget && (
        <div
          onClick={() => setDeleteTarget(null)}
          style={{
            position: "fixed", inset: 0, zIndex: 2000,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: "var(--surface)", borderRadius: 14,
              padding: "24px 28px", width: "min(400px, 90vw)",
              border: "1px solid var(--border)",
              boxShadow: "0 20px 60px rgba(0,0,0,0.25)",
            }}
          >
            <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>
              Delete Profile
            </div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 22 }}>
              Delete <strong>{deleteTarget.name}</strong>? This cannot be undone.
            </div>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button
                onClick={() => setDeleteTarget(null)}
                style={{
                  padding: "8px 16px", fontSize: 13, borderRadius: 8,
                  border: "1px solid var(--border)", background: "var(--surface-raised)",
                  color: "var(--text-secondary)", cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  padding: "8px 18px", fontSize: 13, fontWeight: 600,
                  borderRadius: 8, border: "none", background: "var(--red)",
                  color: "#fff", cursor: "pointer",
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Header ── */}
      <div
        style={{
          padding: "16px 24px 14px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <h1
              className="font-display"
              style={{ fontSize: 20, color: "var(--text)", letterSpacing: "-0.02em" }}
            >
              Profiles
            </h1>
            {!loading && (
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                {profiles.length} {profiles.length === 1 ? "profile" : "profiles"}
              </span>
            )}
          </div>
          <button
            onClick={() => setWizardProfile(null)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "var(--accent)", color: "#fff",
              border: "none", borderRadius: 8, padding: "8px 16px",
              fontSize: 13, fontWeight: 500, cursor: "pointer",
              transition: "opacity 150ms",
            }}
          >
            <Plus size={14} />
            New Profile
          </button>
        </div>
      </div>

      {/* ── Content ── */}
      <div style={{ flex: 1, overflow: "auto", padding: "18px 24px" }}>

        {/* Error banner */}
        {error && (
          <div
            style={{
              padding: "10px 14px",
              background: "#FEF1F0",
              color: "var(--red)",
              borderRadius: 8,
              fontSize: 12,
              marginBottom: 16,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              border: "1px solid rgba(168,50,38,0.15)",
            }}
          >
            {error}
            <button
              onClick={() => setError(null)}
              style={{ background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 15, fontWeight: 700 }}
            >
              &times;
            </button>
          </div>
        )}

        {/* Loading */}
        {loading ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 14 }}>
            {[0, 1, 2].map((i) => <SkeletonCard key={i} />)}
          </div>

        ) : profiles.length === 0 ? (
          /* Empty state */
          <div
            style={{
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              padding: "72px 24px",
              border: "2px dashed var(--border)",
              borderRadius: 14, textAlign: "center",
            }}
          >
            <div
              style={{
                width: 56, height: 56, borderRadius: "50%",
                background: "var(--accent-light)",
                display: "flex", alignItems: "center", justifyContent: "center",
                marginBottom: 16,
              }}
            >
              <User size={24} style={{ color: "var(--accent)" }} />
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>
              No profiles yet
            </div>
            <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.7, maxWidth: 400, marginBottom: 24 }}>
              Profiles store your targeting preferences — which industries to track, which companies to watch,
              and which contacts to reach. Create one to personalise the pipeline.
            </div>
            <button
              onClick={() => setWizardProfile(null)}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                background: "var(--accent)", color: "#fff",
                border: "none", borderRadius: 8, padding: "10px 22px",
                fontSize: 13, fontWeight: 600, cursor: "pointer",
              }}
            >
              <Plus size={14} />
              Create your first profile
            </button>
          </div>

        ) : (
          /* Profile grid */
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 14 }}>
            {profiles.map((profile) => {
              const badge   = PATH_BADGE[profile.path_preference] ?? PATH_BADGE.auto;
              const initial = profile.user_name.charAt(0).toUpperCase();
              const visibleIndustries = profile.target_industries.slice(0, 3);
              const extraIndustries   = profile.target_industries.length - 3;

              return (
                <div
                  key={profile.profile_id}
                  className="card card-hover"
                  onClick={() => setWizardProfile(profile)}
                  style={{ padding: "18px 20px", cursor: "pointer", position: "relative" }}
                >
                  {/* Path badge — top right */}
                  <div
                    style={{
                      position: "absolute", top: 14, right: 14,
                      padding: "2px 9px", borderRadius: 999,
                      fontSize: 10, fontWeight: 600,
                      background: badge.bg, color: badge.color,
                      letterSpacing: "0.02em",
                    }}
                  >
                    {badge.label}
                  </div>

                  {/* Avatar + name */}
                  <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, paddingRight: 90 }}>
                    <div
                      style={{
                        width: 44, height: 44, borderRadius: "50%",
                        background: "var(--accent)", color: "#fff",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 18, fontWeight: 700, flexShrink: 0,
                        fontFamily: "var(--font-display)",
                        boxShadow: "0 2px 8px rgba(176,112,48,0.3)",
                      }}
                    >
                      {initial}
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: 14, fontWeight: 700, color: "var(--text)",
                          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                        }}
                      >
                        {profile.user_name}
                      </div>
                      {profile.own_company && (
                        <div
                          style={{
                            fontSize: 11, color: "var(--text-muted)",
                            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                          }}
                        >
                          {profile.own_company}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Stats row */}
                  <div style={{ display: "flex", gap: 14, fontSize: 11, color: "var(--text-muted)", marginBottom: 10 }}>
                    <span>
                      <strong style={{ color: "var(--text-secondary)" }}>
                        {profile.target_industries.length}
                      </strong>{" "}
                      {profile.target_industries.length === 1 ? "industry" : "industries"}
                    </span>
                    <span>
                      <strong style={{ color: "var(--text-secondary)" }}>
                        {profile.own_products.length}
                      </strong>{" "}
                      {profile.own_products.length === 1 ? "product" : "products"}
                    </span>
                    {profile.account_list.length > 0 && (
                      <span>
                        <strong style={{ color: "var(--text-secondary)" }}>
                          {profile.account_list.length}
                        </strong>{" "}
                        accounts
                      </span>
                    )}
                    <span style={{ marginLeft: "auto", color: "var(--text-muted)" }}>{profile.region}</span>
                  </div>

                  {/* Industry tags */}
                  {visibleIndustries.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 12 }}>
                      {visibleIndustries.map((ind) => (
                        <span
                          key={ind.industry_id}
                          style={{
                            padding: "2px 8px", borderRadius: 999,
                            background: "var(--surface-raised)", border: "1px solid var(--border)",
                            fontSize: 10, fontWeight: 500, color: "var(--text-secondary)",
                          }}
                        >
                          {ind.display_name}
                        </span>
                      ))}
                      {extraIndustries > 0 && (
                        <span
                          style={{
                            padding: "2px 8px", borderRadius: 999,
                            background: "var(--surface-raised)", border: "1px solid var(--border)",
                            fontSize: 10, color: "var(--text-muted)",
                          }}
                        >
                          +{extraIndustries} more
                        </span>
                      )}
                    </div>
                  )}

                  {/* Pipeline stats */}
                  <div
                    style={{
                      display: "flex", gap: 16,
                      paddingTop: 10, borderTop: "1px solid var(--border)",
                      fontSize: 11, color: "var(--text-muted)",
                      marginBottom: 12,
                    }}
                  >
                    <span>
                      <strong className="num" style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                        {profile.total_runs}
                      </strong>{" "}runs
                    </span>
                    <span>
                      <strong className="num" style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                        {profile.total_emails_sent}
                      </strong>{" "}emails
                    </span>
                    <span>
                      <strong className="num" style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                        {profile.total_replies}
                      </strong>{" "}replies
                    </span>
                  </div>

                  {/* Action buttons */}
                  <div
                    style={{ display: "flex", gap: 6 }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      onClick={() => setWizardProfile(profile)}
                      style={{
                        flex: 1,
                        display: "flex", alignItems: "center", justifyContent: "center", gap: 5,
                        padding: "7px 12px", fontSize: 12, fontWeight: 500,
                        borderRadius: 7, border: "1px solid var(--border)",
                        background: "var(--surface-raised)", color: "var(--text-secondary)",
                        cursor: "pointer",
                      }}
                    >
                      <Edit2 size={12} />
                      Edit
                    </button>
                    <button
                      onClick={() => setDeleteTarget({ id: profile.profile_id, name: profile.user_name })}
                      disabled={deleting === profile.profile_id}
                      style={{
                        display: "flex", alignItems: "center", justifyContent: "center", gap: 5,
                        padding: "7px 12px", fontSize: 12, fontWeight: 500,
                        borderRadius: 7, border: "1px solid rgba(168,50,38,0.2)",
                        background: "#FEF1F0", color: "var(--red)",
                        cursor: deleting === profile.profile_id ? "not-allowed" : "pointer",
                        opacity: deleting === profile.profile_id ? 0.5 : 1,
                      }}
                    >
                      <Trash2 size={12} />
                      {deleting === profile.profile_id ? "…" : "Delete"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
