# frontend/ -- Next.js Dashboard

Sales intelligence dashboard. Connects to the FastAPI backend for pipeline execution, trend browsing, and lead management.

## Dev Setup

```bash
cd frontend
npm install
npm run dev     # http://localhost:3000
```

Or from root:
```bash
npm run frontend          # same as above
npm run dev               # starts backend + frontend together
```

The frontend defaults to `http://localhost:8000` for the API (see `lib/api.ts`). No env var needed for local dev.

## Stack

- Next.js 16 (App Router, `(dashboard)` route group)
- React 19
- shadcn/ui (Radix primitives via `@radix-ui`)
- Tailwind CSS 4
- Lucide React icons

## Pages

All dashboard pages live under `app/(dashboard)/` which wraps them in a shared sidebar + nav layout.

| Route | What it does |
|-------|-------------|
| `/` | Landing / redirect to dashboard |
| `/dashboard` | Pipeline status, recent runs, key metrics |
| `/trends` | Detected trends with scores, signal breakdowns, article lists |
| `/leads` | Lead list with filtering (hop, type, confidence) |
| `/leads/[id]` | Full lead sheet: company, contact, pitch, scores |
| `/companies` | Discovered companies with bandit relevance scores |
| `/learning` | 6 feedback loop status, weight history, bandit posteriors |
| `/settings` | Provider config, pipeline parameters, cooldown state |
| `/history` | Past pipeline run results and recordings |

## API Layer

`lib/api.ts` -- single module for all backend calls. Uses a typed `apiFetch<T>()` wrapper.

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
```

Key patterns:
- `api.runPipeline(mockMode)` -- POST to start, returns `run_id`
- `api.streamProgress(runId)` -- SSE via `EventSource`, returns async iterator of `PipelineStreamEvent`
- `api.getResult(runId)` -- GET full results after completion
- `api.getLeads(filters)` -- GET filtered leads list
- `api.submitFeedback(...)` -- POST rating for learning loops

Types are in `lib/types.ts` -- mirrors the FastAPI Pydantic schemas.

## State Management

`contexts/pipeline-context.tsx` provides `PipelineProvider` wrapping all dashboard pages:
- Pipeline status (`idle` / `running` / `complete` / `error`)
- SSE log stream
- Result data (trends, leads, companies)
- Run/reset actions

Use `usePipelineContext()` in any dashboard page to access pipeline state.

## Key Files

```
frontend/
├── app/
│   ├── globals.css                   # Tailwind + CSS custom properties
│   ├── layout.tsx                    # Root layout (font, metadata)
│   ├── page.tsx                      # Landing page
│   └── (dashboard)/                  # Route group -- shared layout
│       ├── layout.tsx                # Sidebar + nav shell
│       └── {dashboard,trends,...}/   # Page directories
├── components/                       # Reusable UI (shadcn + custom)
├── contexts/pipeline-context.tsx     # Global pipeline state
├── hooks/use-pipeline.ts             # SSE + polling logic
├── lib/
│   ├── api.ts                        # Backend API client
│   ├── types.ts                      # TypeScript types (mirrors Pydantic)
│   └── utils.ts                      # cn() helper
└── package.json
```

## Gotchas

**NEXT_PUBLIC_API_URL is build-time only.** In Docker, this is passed as a build arg (see `docker/Dockerfile.frontend`). Setting it as a runtime env var does nothing for client-side code -- Next.js inlines `NEXT_PUBLIC_*` during `next build`. In dev mode, the `?? "http://localhost:8000"` fallback handles it.

**SSE connection.** The pipeline stream endpoint uses Server-Sent Events. The `use-pipeline` hook manages EventSource lifecycle, reconnection, and cleanup. If the API is unreachable, SSE silently fails -- check browser Network tab.

**shadcn components.** Added via `npx shadcn@latest add <component>`. They live in `components/ui/` and are fully editable (not a node_module). Customization goes in `globals.css` CSS custom properties.

## Build

```bash
npm run build    # produces .next/ output
npm start        # serves production build on port 3000
```

For Docker: the `docker/Dockerfile.frontend` uses a multi-stage build (builder + runner).
