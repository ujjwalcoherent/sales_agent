"use client";

import { createContext, useContext, useState, useEffect, useCallback } from "react";
import { listProfiles } from "@/lib/api";
import type { UserProfile } from "@/lib/types";

const STORAGE_KEY = "harbinger_active_profile_id";

interface ProfileContextValue {
  activeProfile: UserProfile | null;
  profiles: UserProfile[];
  loading: boolean;
  setActiveProfileId: (id: string) => void;
  refreshProfiles: () => Promise<void>;
}

const ProfileContext = createContext<ProfileContextValue>({
  activeProfile: null,
  profiles: [],
  loading: true,
  setActiveProfileId: () => {},
  refreshProfiles: async () => {},
});

export function ProfileProvider({ children }: { children: React.ReactNode }) {
  const [profiles, setProfiles] = useState<UserProfile[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshProfiles = useCallback(async () => {
    try {
      const res = await listProfiles();
      setProfiles(res.profiles);
      // Auto-select: restore stored id, or pick first if only one
      const stored = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
      const ids = res.profiles.map((p) => p.profile_id);
      if (stored && ids.includes(stored)) {
        setActiveId(stored);
      } else if (res.profiles.length === 1) {
        setActiveId(res.profiles[0].profile_id);
      }
    } catch { /* noop */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { refreshProfiles(); }, [refreshProfiles]);

  const setActiveProfileId = useCallback((id: string) => {
    setActiveId(id);
    if (typeof window !== "undefined") localStorage.setItem(STORAGE_KEY, id);
  }, []);

  const activeProfile = profiles.find((p) => p.profile_id === activeId) ?? null;

  return (
    <ProfileContext.Provider value={{ activeProfile, profiles, loading, setActiveProfileId, refreshProfiles }}>
      {children}
    </ProfileContext.Provider>
  );
}

export function useProfile() {
  return useContext(ProfileContext);
}
