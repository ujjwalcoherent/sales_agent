"use client";

import React from "react";
import Link from "next/link";
import { ArrowRight, ArrowUpRight, Zap, CheckCircle2, Building2, Mail } from "lucide-react";
import { APP_NAME } from "@/lib/config";

const TICKER_ITEMS = [
  "Regulatory Shifts", "Policy Changes", "Macro Signals", "Technology Disruption",
  "Industry Events", "Talent Movements", "Capital Flows", "M&A Activity",
  "ESG Mandates", "Market Expansion",
];

const HOW_IT_WORKS = [
  {
    num: "01",
    title: "Scan",
    subtitle: "24+ sources, real-time",
    description:
      "Intelligent crawlers sweep financial news, regulatory portals and industry feeds for market signals — deduplicated, ranked by urgency, and ready for analysis.",
    visual: <ScanVisual />,
    tag: "Source Intel",
  },
  {
    num: "02",
    title: "Analyse",
    subtitle: "AI Council · 4 specialists",
    description:
      "Four specialist agents debate the impact for mid-size companies. Causal chains, pain points and buying intent are synthesised — with a moderator resolving disagreements.",
    visual: <AnalyseVisual />,
    tag: "Causal Council",
  },
  {
    num: "03",
    title: "Deliver",
    subtitle: "Verified leads + email",
    description:
      "Companies are matched to the signal, decision-makers identified and verified, and a personalised outreach email generated for each contact — pitch-ready.",
    visual: <DeliverVisual />,
    tag: "Lead Crystallizer",
  },
];

const PROOF_POINTS = [
  "Reads 187+ articles per run on average",
  "Proprietary entity resolution on every match",
  "Adaptive intelligence selects highest-signal feeds",
  "6 self-optimising intelligence layers improve every run",
  "Multi-channel contact verification ensures deliverability",
  "Pipeline completes in under 60 seconds",
];


export default function LandingPage() {
  return (
    <div style={{ background: "var(--bg)", minHeight: "100vh", fontFamily: "var(--font-sans)" }}>

      {/* ── Nav ──────────────────────────────────────────── */}
      <nav style={{
        position: "sticky", top: 0, zIndex: 50,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 64px",
        background: "var(--nav-bg)", backdropFilter: "blur(14px)",
        borderBottom: "1px solid var(--border)",
      }}>
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 8, textDecoration: "none" }}>
          <div style={{ width: 26, height: 26, background: "var(--text)", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Zap size={13} color="var(--bg)" strokeWidth={2.5} />
          </div>
          <span className="font-display" style={{ fontSize: 15, color: "var(--text)" }}>{APP_NAME}</span>
        </Link>
        <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
          {[{ label: "How it works", href: "#how" }, { label: "Features", href: "#features" }].map(({ label, href }) => (
            <a key={label} href={href} className="nav-link-landing" style={{ fontSize: 13, color: "var(--text-secondary)", textDecoration: "none" }}>{label}</a>
          ))}
          <Link href="/onboarding" className="landing-btn-primary"
            style={{ display: "flex", alignItems: "center", gap: 5, padding: "7px 16px", borderRadius: 8, background: "var(--text)", color: "var(--bg)", fontSize: 12, fontWeight: 600, textDecoration: "none", letterSpacing: "0.02em" }}>
            Open App <ArrowRight size={12} />
          </Link>
        </div>
      </nav>

      {/* ── Hero ─────────────────────────────────────────── */}
      <section style={{
        background: "var(--bg)", position: "relative", overflow: "hidden",
        paddingTop: 96,
      }}>
        {/* Warm glow */}
        <div style={{ position: "absolute", top: -100, left: "40%", width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle, rgba(176,112,48,0.07) 0%, transparent 65%)", pointerEvents: "none" }} />
        <div style={{ position: "absolute", top: 60, right: "5%", width: 400, height: 400, borderRadius: "50%", background: "radial-gradient(circle, rgba(45,106,79,0.05) 0%, transparent 65%)", pointerEvents: "none" }} />

        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 96px", display: "flex", alignItems: "center", gap: 48 }}>

          {/* Left: headline */}
          <div style={{ flex: "0 0 480px", maxWidth: 480, paddingBottom: 112 }}>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 999, border: "1px solid rgba(176,112,48,0.25)", background: "var(--accent-light)", marginBottom: 28 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", display: "inline-block", animation: "pulse-dot 1.5s ease-in-out infinite" }} />
              <span style={{ fontSize: 11, color: "var(--accent)", fontWeight: 600, letterSpacing: "0.08em" }}>SIGNAL INTELLIGENCE</span>
            </div>

            <h1 className="font-display" style={{ fontSize: "clamp(40px, 4vw, 62px)", color: "var(--text)", lineHeight: 1.06, letterSpacing: "-0.03em", marginBottom: 22 }}>
              Find buyers
              <br />
              <span style={{ color: "var(--accent)" }}>before they know</span>
              <br />
              they&apos;re buying.
            </h1>

            <p style={{ fontSize: 15, color: "var(--text-secondary)", lineHeight: 1.75, marginBottom: 36, maxWidth: 400 }}>
              We read hundreds of market signals daily — regulatory shifts, capital flows, M&A activity — and surface the exact companies that need your solution right now, with verified contacts and personalised outreach ready to send.
            </p>

            <div style={{ display: "flex", gap: 10 }}>
              <Link href="/onboarding" className="landing-btn-primary"
                style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 24px", borderRadius: 10, background: "var(--accent)", color: "var(--surface)", fontSize: 13, fontWeight: 700, textDecoration: "none" }}>
                Open Dashboard <ArrowRight size={14} />
              </Link>
              <a href="#how" className="landing-btn-light"
                style={{ display: "flex", alignItems: "center", gap: 6, padding: "12px 20px", borderRadius: 10, border: "1px solid var(--border)", color: "var(--text-secondary)", fontSize: 13, textDecoration: "none", background: "var(--surface)" }}>
                See how it works
              </a>
            </div>
          </div>

          {/* Right: result card mockup */}
          <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "flex-end", paddingBottom: 0 }}>
            <LeadResultsMockup />
          </div>
        </div>

        <SectionWave fill="var(--surface-raised)" variant="a" />
      </section>

      {/* ── Ticker ───────────────────────────────────────── */}
      <div style={{ background: "var(--surface-raised)", borderTop: "1px solid var(--border)", padding: "12px 0 52px", overflow: "hidden", position: "relative" }}>
        <div style={{ display: "flex", gap: 0, animation: "ticker 28s linear infinite", width: "max-content" }}>
          {[...TICKER_ITEMS, ...TICKER_ITEMS].map((item, i) => (
            <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 12, padding: "0 24px", fontSize: 11, color: "var(--text-muted)", fontWeight: 500, letterSpacing: "0.06em", whiteSpace: "nowrap" }}>
              <span style={{ width: 4, height: 4, borderRadius: "50%", background: "var(--accent)", display: "inline-block", flexShrink: 0 }} />
              {item.toUpperCase()}
            </span>
          ))}
        </div>
        <style>{`@keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }`}</style>
        <SectionWave fill="var(--bg)" variant="b" />
      </div>

      {/* ── HOW IT WORKS ─────────────────────────────────── */}
      <section id="how" style={{ padding: "100px 0 140px", background: "var(--bg)", position: "relative" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 96px" }}>
          <div style={{ textAlign: "center", marginBottom: 64 }}>
            <div style={{ fontSize: 10, color: "var(--text-xmuted)", letterSpacing: "0.12em", fontWeight: 700, marginBottom: 12 }}>HOW IT WORKS</div>
            <h2 className="font-display" style={{ fontSize: "clamp(28px, 3vw, 42px)", color: "var(--text)", letterSpacing: "-0.025em", lineHeight: 1.15 }}>
              Three steps from raw news<br />to pipeline-ready leads
            </h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 56px 1fr 56px 1fr", alignItems: "center", gap: 0 }}>
            {HOW_IT_WORKS.map((step, i) => (
              <React.Fragment key={step.num}>
                <StepCard {...step} />
                {i < HOW_IT_WORKS.length - 1 && (
                  <div key={`arrow-${i}`} style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <div style={{ position: "relative", width: 36, height: 2, background: "linear-gradient(90deg, var(--border), var(--accent-mid))", borderRadius: 2 }}>
                      <div className="travel-dot" />
                      <div style={{ position: "absolute", right: -5, top: "50%", transform: "translateY(-50%)", width: 0, height: 0, borderTop: "4px solid transparent", borderBottom: "4px solid transparent", borderLeft: "6px solid var(--accent-mid)" }} />
                    </div>
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
        <SectionWave fill="var(--surface)" variant="c" />
      </section>

      {/* ── Features strip ───────────────────────────────── */}
      <section id="features" style={{ padding: "80px 0 140px", background: "var(--surface)", position: "relative" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 96px" }}>
          <div style={{ textAlign: "center", marginBottom: 48 }}>
            <div style={{ fontSize: 10, color: "var(--text-xmuted)", letterSpacing: "0.12em", fontWeight: 700, marginBottom: 12 }}>UNDER THE HOOD</div>
            <h2 className="font-display" style={{ fontSize: 34, color: "var(--text)", letterSpacing: "-0.025em" }}>
              Built to be accurate,<br />not just fast.
            </h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 1, background: "var(--border)", borderRadius: 14, overflow: "hidden", border: "1px solid var(--border)" }}>
            {PROOF_POINTS.map((point) => (
              <div key={point} style={{ padding: "22px 24px", background: "var(--surface)", display: "flex", alignItems: "flex-start", gap: 10 }}>
                <CheckCircle2 size={14} style={{ color: "var(--green)", marginTop: 2, flexShrink: 0 }} />
                <span style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>{point}</span>
              </div>
            ))}
          </div>
        </div>
        <SectionWave fill="var(--bg-dark)" variant="d" />
      </section>

      {/* ── CTA ──────────────────────────────────────────── */}
      <section style={{ background: "var(--bg-dark)", padding: "88px 0 140px", position: "relative" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 96px", display: "grid", gridTemplateColumns: "1fr auto", alignItems: "center", gap: 80 }}>

          {/* Left: copy */}
          <div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 999, border: "1px solid rgba(176,112,48,0.2)", background: "rgba(176,112,48,0.08)", marginBottom: 24 }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--accent)", display: "inline-block", animation: "pulse-dot 1.5s ease-in-out infinite" }} />
              <span style={{ fontSize: 10, color: "var(--accent)", fontWeight: 600, letterSpacing: "0.1em" }}>LIVE DEMO — NO SETUP</span>
            </div>

            <h2 className="font-display" style={{ fontSize: "clamp(36px, 4vw, 56px)", color: "var(--term-text)", letterSpacing: "-0.025em", lineHeight: 1.08, marginBottom: 20 }}>
              Watch it work.<br />
              <span style={{ color: "var(--accent)" }}>15 steps.</span> 45 seconds.
            </h2>

            <p style={{ fontSize: 14, color: "var(--term-text-muted)", lineHeight: 1.75, maxWidth: 420, marginBottom: 36 }}>
              Hit <span style={{ color: "var(--accent)", fontWeight: 600 }}>Demo</span> in the Pipeline bar — the system scans real sources, debates market impact, matches companies, verifies contacts and generates personalised outreach. All in under a minute.
            </p>

            <div style={{ display: "flex", gap: 10 }}>
              <Link href="/onboarding" className="landing-btn-primary"
                style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 24px", borderRadius: 10, background: "var(--accent)", color: "var(--term-bg)", fontSize: 13, fontWeight: 700, textDecoration: "none" }}>
                <Zap size={13} /> Open Dashboard
              </Link>
              <Link href="/leads"
                style={{ display: "flex", alignItems: "center", gap: 6, padding: "12px 20px", borderRadius: 10, border: "1px solid var(--term-border)", color: "var(--term-text-muted)", fontSize: 13, textDecoration: "none", background: "transparent" }}>
                Browse sample leads <ArrowUpRight size={12} />
              </Link>
            </div>
          </div>

          {/* Right: step preview */}
          <div style={{ flexShrink: 0 }}>
            <DemoStepPreview />
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────── */}
      <LandingFooter />
    </div>
  );
}

// ── Section Wave ──────────────────────────────────────────────────────

const WAVE_PATHS = {
  // gentle S-curve (left-biased)
  a: "M0,0 Q360,56 720,40 Q1080,24 1440,48 L1440,56 L0,56 Z",
  // inverse S (right-biased)
  b: "M0,44 Q400,8 720,36 Q1040,60 1440,12 L1440,56 L0,56 Z",
  // deep single arc
  c: "M0,24 Q360,64 720,36 Q1080,8 1440,44 L1440,56 L0,56 Z",
  // broad shallow roll
  d: "M0,40 Q480,4 720,28 Q960,52 1440,16 L1440,56 L0,56 Z",
} as const;

function SectionWave({ fill, variant = "a", height = 56 }: {
  fill: string; variant?: keyof typeof WAVE_PATHS; height?: number;
}) {
  return (
    <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, lineHeight: 0, pointerEvents: "none", zIndex: 2 }}>
      <svg viewBox={`0 0 1440 ${height}`} xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="none" style={{ display: "block", width: "100%", height }}>
        <path d={WAVE_PATHS[variant]} fill={fill} />
      </svg>
    </div>
  );
}

// ── Lead Results Mockup ────────────────────────────────────────────────

function LeadResultsMockup() {
  const leads = [
    { co: "Aroha Finserv",     role: "Chief Compliance Officer", domain: "arohafinserv.com",   score: 88, col: { bg: "#EBF5F0", text: "#2D6A4F" } },
    { co: "Kalpataru Systems", role: "Chief Technology Officer", domain: "kalpataru.tech",       score: 84, col: { bg: "#F5EDD8", text: "#B07030" } },
    { co: "Capita Finance",    role: "VP Operations",            domain: "capitafinance.in",    score: 79, col: { bg: "#E6EEF7", text: "#2A5A8A" } },
  ];

  return (
    <div style={{
      width: 420,
      background: "#FFFFFF",
      borderRadius: 16,
      boxShadow: "0 -20px 60px rgba(24,23,15,0.10), 0 -4px 20px rgba(24,23,15,0.06), 0 0 0 1px rgba(24,23,15,0.06)",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{ padding: "11px 16px", borderBottom: "1px solid #F0EFEA", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#FAFAF7" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ display: "flex", gap: 4 }}>
            {["#FF5F56","#FFBD2E","#27C93F"].map((c) => <div key={c} style={{ width: 8, height: 8, borderRadius: "50%", background: c, opacity: 0.9 }} />)}
          </div>
          <span style={{ fontSize: 11, color: "#8A8878", marginLeft: 6 }}>Pipeline Results</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#2D6A4F" }} />
          <span style={{ fontSize: 10, color: "#2D6A4F", fontWeight: 600 }}>Complete · 8 leads · 47s</span>
        </div>
      </div>

      {/* Lead rows */}
      {leads.map((l) => (
        <div key={l.co} style={{ padding: "12px 16px", borderBottom: "1px solid #F5F4EF", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 9, background: "#F5F4EF", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <Building2 size={14} style={{ color: "#8A8878" }} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#18170F" }}>{l.co}</div>
            <div style={{ fontSize: 10, color: "#8A8878", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{l.role} · {l.domain}</div>
          </div>
          <div style={{ background: l.col.bg, borderRadius: 7, padding: "4px 10px", textAlign: "center", flexShrink: 0 }}>
            <div style={{ fontSize: 16, fontFamily: "DM Serif Display, serif", color: l.col.text, lineHeight: 1 }}>{l.score}</div>
          </div>
        </div>
      ))}

      {/* Email preview */}
      <div style={{ padding: "12px 16px", background: "#F8F7F2" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
          <Mail size={11} style={{ color: "#8A8878" }} />
          <span style={{ fontSize: 9, color: "#8A8878", fontWeight: 700, letterSpacing: "0.07em" }}>PERSONALISED OUTREACH READY</span>
        </div>
        <div style={{ fontSize: 11, color: "#4A4940", lineHeight: 1.6, fontStyle: "italic", background: "#FFF", border: "1px solid #E4E2D8", borderRadius: 8, padding: "9px 11px" }}>
          "Hi Priya, the new RBI digital lending circular puts a 90-day implementation clock on NBFCs like Aroha..."
        </div>
        <div style={{ marginTop: 8, display: "flex", gap: 5 }}>
          <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 999, background: "#EBF5F0", color: "#2D6A4F", fontWeight: 600 }}>✓ Verified email</span>
          <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 999, background: "#F5EDD8", color: "#B07030", fontWeight: 600 }}>82% confidence</span>
        </div>
      </div>
    </div>
  );
}

// ── Demo Step Preview ──────────────────────────────────────────────────

function DemoStepPreview() {
  const steps = [
    { label: "Scanning 24+ sources",          done: true  },
    { label: "Classifying events by severity", done: true  },
    { label: "Causal council debating impact", done: true  },
    { label: "Matching companies to signal",   done: true  },
    { label: "Generating outreach emails",     done: false, active: true },
  ];
  return (
    <div style={{ width: 248, background: "var(--surface-raised)", border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden" }}>
      <div style={{ padding: "9px 12px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", display: "inline-block", animation: "pulse-dot 1.5s ease-in-out infinite" }} />
        <span style={{ fontSize: 10, color: "var(--text-secondary)", fontWeight: 600 }}>Demo Run</span>
        <div style={{ marginLeft: "auto", fontSize: 10, color: "var(--accent)", fontWeight: 600 }}>73%</div>
      </div>
      <div style={{ height: 2, background: "var(--border)" }}>
        <div style={{ height: "100%", width: "73%", background: "linear-gradient(90deg, var(--accent), var(--accent-mid))", borderRadius: 1 }} />
      </div>
      <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
        {steps.map(({ label, done, active }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 14, height: 14, borderRadius: "50%", background: done ? "var(--green-light)" : active ? "var(--accent-light)" : "var(--surface)", border: `1px solid ${done ? "var(--green)" : active ? "var(--accent)" : "var(--border)"}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
              {done   && <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--green)" }} />}
              {active && <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", animation: "pulse-dot 1.5s ease-in-out infinite" }} />}
            </div>
            <span style={{ fontSize: 11, color: done ? "var(--green)" : active ? "var(--text)" : "var(--text-xmuted)" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Step Card ─────────────────────────────────────────────────────────

function StepCard({ num, title, subtitle, description, visual, tag }: {
  num: string; title: string; subtitle: string; description: string;
  visual: React.ReactNode; tag: string;
}) {
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 16, overflow: "hidden", position: "relative", boxShadow: "var(--shadow-sm)" }}>
      <div className="font-display" style={{ position: "absolute", top: -8, right: 16, fontSize: 96, color: "var(--surface-raised)", lineHeight: 1, userSelect: "none", pointerEvents: "none", letterSpacing: "-0.04em" }}>
        {num}
      </div>
      <div style={{ height: 168, background: "var(--surface-raised)", display: "flex", alignItems: "center", justifyContent: "center", borderBottom: "1px solid var(--border)", position: "relative", zIndex: 1, overflow: "hidden" }}>
        {visual}
      </div>
      <div style={{ padding: "20px 22px 24px", position: "relative", zIndex: 1 }}>
        <div style={{ display: "inline-block", fontSize: 10, color: "var(--accent)", background: "var(--accent-light)", padding: "2px 8px", borderRadius: 999, fontWeight: 600, letterSpacing: "0.04em", marginBottom: 10 }}>{tag}</div>
        <h3 className="font-display" style={{ fontSize: 22, color: "var(--text)", letterSpacing: "-0.02em", marginBottom: 4 }}>{title}</h3>
        <div style={{ fontSize: 12, color: "var(--accent)", fontWeight: 500, marginBottom: 10 }}>{subtitle}</div>
        <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.65 }}>{description}</p>
      </div>
    </div>
  );
}

// ── Step Visuals (clean, static with minimal motion) ──────────────────

function ScanVisual() {
  const sources = [
    { name: "rbi.org.in",          w: 86, active: true  },
    { name: "economictimes.com",   w: 72, active: false },
    { name: "moneycontrol.com",    w: 60, active: false },
    { name: "businessline.com",    w: 48, active: false },
    { name: "ndtv.com/business",   w: 36, active: false },
  ];
  return (
    <div style={{ padding: "14px 20px 10px", width: "100%", display: "flex", flexDirection: "column", gap: 8 }}>
      {sources.map((src) => (
        <div key={src.name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", flexShrink: 0, background: src.active ? "var(--green)" : "var(--accent)", ...(src.active ? { animation: "pulse-dot 1.5s ease-in-out infinite" } : { opacity: 0.5 }) }} />
          <div style={{ flex: 1, height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${src.w}%`, background: src.active ? "var(--green-light)" : "var(--accent-light)", borderRadius: 3, borderRight: `2px solid ${src.active ? "var(--green)" : "var(--accent)"}` }} />
          </div>
          <span style={{ fontSize: 9, color: "var(--text-xmuted)", whiteSpace: "nowrap", minWidth: 128, textAlign: "right" }}>{src.name}</span>
        </div>
      ))}
      <div style={{ marginTop: 2, display: "flex", alignItems: "center", gap: 5 }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--accent)", display: "inline-block", animation: "pulse-dot 1s ease-in-out infinite" }} />
        <span style={{ fontSize: 9, color: "var(--text-muted)" }}>Scanning 187 articles · 24 sources</span>
      </div>
    </div>
  );
}

function AnalyseVisual() {
  const agents = [
    { name: "Risk Analyst",  color: "var(--red)",    bg: "var(--red-light)",    w: 78 },
    { name: "Market Expert", color: "var(--blue)",   bg: "var(--blue-light)",   w: 65 },
    { name: "Sales Advisor", color: "var(--green)",  bg: "var(--green-light)",  w: 82 },
    { name: "Moderator",     color: "var(--accent)", bg: "var(--accent-light)", w: 58 },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7, padding: "14px 20px", width: "100%" }}>
      {agents.map(({ name, color, bg, w }) => (
        <div key={name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ padding: "3px 8px", borderRadius: 6, background: bg, flexShrink: 0, minWidth: 96 }}>
            <div style={{ fontSize: 9, fontWeight: 700, color, letterSpacing: "0.02em" }}>{name}</div>
          </div>
          <div style={{ flex: 1, height: 4, background: `${color}18`, borderRadius: 2, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${w}%`, background: color, borderRadius: 2, opacity: 0.75 }} />
          </div>
        </div>
      ))}
      <div style={{ fontSize: 9, color: "var(--text-xmuted)", marginTop: 2 }}>debating → consensus</div>
    </div>
  );
}

function DeliverVisual() {
  return (
    <div style={{ padding: "12px 16px", width: "100%" }}>
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "11px 13px", boxShadow: "var(--shadow-xs)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text)" }}>Aroha Finserv</div>
            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>NBFC · CCO · priya@arohafinserv.com</div>
          </div>
          <div style={{ background: "var(--green-light)", borderRadius: 6, padding: "3px 8px" }}>
            <span className="num" style={{ fontSize: 16, color: "var(--green)" }}>88</span>
          </div>
        </div>
        <div style={{ fontSize: 10, color: "var(--text-secondary)", padding: "7px 8px", background: "var(--surface-raised)", borderRadius: 6, fontStyle: "italic", lineHeight: 1.55 }}>
          "Hi Priya, the new RBI digital lending circular puts a 90-day deadline on NBFCs like Aroha..."
        </div>
        <div style={{ marginTop: 7, display: "flex", gap: 4 }}>
          <span className="badge badge-green" style={{ fontSize: 9 }}>✉ 82%</span>
          <span className="badge badge-amber" style={{ fontSize: 9 }}>Verified</span>
        </div>
      </div>
    </div>
  );
}

// ── Landing Footer ────────────────────────────────────────────────────

function LandingFooter() {
  return (
    <footer>
      {/* Zone 1: Utility strip — dark, seamless from CTA */}
      <div style={{
        background: "var(--bg-dark)",
        borderTop: "1px solid rgba(255,255,255,0.06)",
        padding: "0 64px",
        height: 48,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <div style={{ width: 20, height: 20, background: "var(--surface-hover)", border: "1px solid var(--term-border)", borderRadius: 5, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Zap size={10} color="var(--term-accent)" strokeWidth={2.5} />
          </div>
          <span className="font-display" style={{ fontSize: 13, color: "var(--term-text-dim)" }}>{APP_NAME}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {[
            { label: "Dashboard", href: "/dashboard" },
            { label: "Trends",    href: "/trends"    },
            { label: "Leads",     href: "/leads"     },
          ].map(({ label, href }, i, arr) => (
            <span key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Link href={href} style={{ fontSize: 12, color: "var(--term-text-dim)", textDecoration: "none" }}>
                {label}
              </Link>
              {i < arr.length - 1 && (
                <span style={{ fontSize: 12, color: "var(--term-text-xdim)" }}>·</span>
              )}
            </span>
          ))}
        </div>

        <span style={{ fontSize: 11, color: "var(--term-text-xdim)" }}>© 2026 Harbinger</span>
      </div>

      {/* Zone 2: Wordmark anchor */}
      <WordmarkAnchor />
    </footer>
  );
}


function WordmarkAnchor() {
  return (
    <div
      style={{
        background: "var(--bg-dark)",
        overflow: "hidden",
        height: 140,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        paddingTop: 28,
      }}
    >
      <p style={{
        fontSize: 11,
        color: "var(--term-text-dim)",
        letterSpacing: "0.06em",
        marginBottom: 6,
        fontWeight: 500,
      }}>
        Turning market noise into pipeline since 2026
      </p>
      <div
        className="font-display"
        style={{
          fontSize: "clamp(80px, 12vw, 140px)",
          color: "var(--accent)",
          lineHeight: 1,
          letterSpacing: "-0.03em",
          userSelect: "none",
        }}
      >
        {APP_NAME}
      </div>
    </div>
  );
}
