import {
  LayoutDashboard,
  TrendingUp,
  Users,
  Building2,
  Crosshair,
  Brain,
  History,
} from "lucide-react";

export const APP_NAME = "Harbinger";
export const APP_TAGLINE = "Signal Intelligence";

export const NAV_ITEMS = [
  { id: "dashboard",  label: "Dashboard",  href: "/dashboard",  icon: LayoutDashboard },
  { id: "trends",     label: "Trends",     href: "/trends",     icon: TrendingUp },
  { id: "leads",      label: "Leads",      href: "/leads",      icon: Users },
  { id: "companies",  label: "Companies",  href: "/companies",  icon: Building2 },
  { id: "campaigns",  label: "Campaigns",  href: "/campaigns",  icon: Crosshair },
  { id: "learning",   label: "Learning",   href: "/learning",   icon: Brain },
  { id: "history",    label: "History",     href: "/history",    icon: History },
] as const;

export type NavId = (typeof NAV_ITEMS)[number]["id"] | "settings";
