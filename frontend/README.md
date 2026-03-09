# frontend/ — Next.js Dashboard

Sales intelligence dashboard. Connects to the FastAPI backend at `http://localhost:8000`.

## Dev Setup

```bash
cd frontend
npm install
npm run dev     # http://localhost:3000
```

Backend must be running separately:
```bash
uvicorn app.main:app --reload --port 8000
```

## Stack

- **Next.js 16.1.6** (App Router, `(dashboard)` route group)
- **React 19.2.3**
- **shadcn/ui** (Radix primitives — components live in `components/ui/`, NOT node_modules)
- **Tailwind CSS 4**
- **Lucide React** icons
- **TypeScript** throughout

## Pages

All dashboard pages live under `app/(dashboard)/` which applies a shared sidebar + nav layout.

| Route | What it does |
|-------|-------------|
| `/` | Landing page / redirect |
| `/dashboard` | Pipeline trigger, run status, SSE log stream, summary metrics |
| `/trends` | Detected clusters: coherence scores, article counts, entity lists |
| `/leads` | Lead list with filters (hop type, confidence, industry); inline lead panel |
| `/leads/[id]` | Full lead sheet: company profile, contact, email draft, score breakdown |
| `/companies` | Company search (web-enriched), saved companies, company news feed |
| `/companies/[id]` | Company detail: news timeline, generate-leads button |
| `/campaigns` | Campaign CRUD: create, list, run, delete, export CSV |
| `/learning` | Learning loop status: bandit posteriors, weight history, hypothesis version |
| `/history` | Past pipeline run recordings with full result replay |
| `/settings` | Provider config, API keys status, pipeline parameters, cooldown state |

## API Layer (`lib/api.ts`)

Single module for all backend calls. Uses typed `apiFetch<T>()` wrapper.

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
```

**Key API methods:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `api.runPipeline()` | `POST /api/v1/pipeline/run` | Start pipeline, get `run_id` |
| `api.streamProgress(runId)` | `GET /api/v1/pipeline/stream/{id}` | SSE real-time events |
| `api.getResult(runId)` | `GET /api/v1/pipeline/result/{id}` | Full results after completion |
| `api.getLeads(filters)` | `GET /api/v1/leads` | Filtered lead list |
| `api.submitFeedback(...)` | `POST /api/v1/feedback` | Rating → SetFit training |
| `api.searchCompanies(q)` | `POST /api/v1/companies/search` | Web-enriched company search |
| `api.getCompanyNews(id)` | `GET /api/v1/companies/{id}/news` | Paginated ChromaDB news |
| `api.createCampaign(...)` | `POST /api/v1/campaigns/` | Create outreach campaign |
| `api.runCampaign(id)` | `POST /api/v1/campaigns/{id}/run` | Execute campaign |
| `api.getLearningStatus()` | `GET /api/v1/learning/status` | All 7 loop statuses |

SSE stream uses `EventSource` (not `fetch`). Each event is a `PipelineStreamEvent` with
`type` (`stage_start`, `stage_complete`, `log`, `error`) and `data` payload.

## State Management

`contexts/pipeline-context.tsx` provides `PipelineProvider` wrapping all dashboard pages:
- Pipeline status: `idle` / `running` / `complete` / `error`
- SSE log stream (ring buffer, last 200 lines)
- Result data: trends, leads, companies
- Run/reset/cancel actions

`hooks/use-pipeline.ts` manages `EventSource` lifecycle: connect on run start, auto-reconnect on drop, cleanup on unmount.

Use `usePipelineContext()` in any page/component to access state.

## File Map

```
frontend/
├── app/
│   ├── globals.css                   # Tailwind + CSS custom properties (dark theme tokens)
│   ├── layout.tsx                    # Root layout (font, metadata)
│   ├── page.tsx                      # Landing page
│   └── (dashboard)/
│       ├── layout.tsx                # Sidebar + nav shell (shared across all pages)
│       ├── dashboard/page.tsx
│       ├── trends/page.tsx
│       ├── leads/
│       │   ├── page.tsx
│       │   └── [id]/page.tsx
│       ├── companies/
│       │   ├── page.tsx
│       │   └── [id]/page.tsx
│       ├── campaigns/page.tsx
│       ├── learning/page.tsx
│       ├── history/[id]/page.tsx
│       └── settings/page.tsx
├── components/
│   ├── ui/                           # shadcn components (editable, not node_modules)
│   └── dashboard/                    # Feature components (lead panels, terminal, etc.)
├── contexts/pipeline-context.tsx     # Global pipeline state
├── hooks/use-pipeline.ts             # SSE + polling logic
└── lib/
    ├── api.ts                        # All backend API calls
    ├── types.ts                      # TypeScript types (mirrors Pydantic schemas)
    ├── config.ts                     # Feature flags + constants
    └── utils.ts                      # cn() helper + formatting
```

## Types (`lib/types.ts`)

Mirrors FastAPI Pydantic schemas exactly:

```typescript
interface TrendData {
    id: string;
    trend_title: string;           // internal field name (NOT "title")
    industries_affected: string[]; // internal field name (NOT "industries")
    coherence_score: number;
    article_count: number;
    evidence_chain: EvidenceChain | null;
}

interface LeadSheet {
    id: string;
    company_name: string;
    contact_name: string | null;
    contact_email: string | null;
    email_draft: string | null;
    opportunity_signal_strength: number;  // 0.0–1.0
    fit_score: number;
}
```

## Gotchas

**`NEXT_PUBLIC_API_URL` is build-time only.** Next.js inlines `NEXT_PUBLIC_*` variables during
`next build`. Setting it as a runtime env var does nothing for client-side code.
In dev mode, the `?? "http://localhost:8000"` fallback handles local development.

**SSE connection.** The pipeline uses Server-Sent Events, not WebSockets. `EventSource` does not
support custom headers — auth (if added) must go in query params. If the API is unreachable,
SSE silently fails with no error in console — check browser Network tab.

**shadcn components.** Added via `npx shadcn@latest add <component>`. They live in
`components/ui/` and are fully editable source files. Customization goes in `globals.css`
CSS custom properties.

**TrendData field names.** The API response uses `trend_title` and `industries_affected`
(not `title`/`industries`). The backend maps these at the serialization layer. TypeScript
types in `lib/types.ts` must match the backend field names exactly.

## Build

```bash
npm run build    # produces .next/ static output
npm start        # serves production build on port 3000
```
