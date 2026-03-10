# app/agents/ — LangGraph Multi-Agent Pipeline

8-node `StateGraph` compiled with `InMemorySaver` checkpointing. `stream_mode="updates"` is mandatory.

```
agents/
├── orchestrator.py       # StateGraph definition, all node functions, _compute_oss()
├── deps.py               # AgentDeps (lazy-init shared state), HopSignal, LearningSignal
├── source_intel.py       # Node 1: RSS + Tavily fetch, embed, classify
├── analysis.py           # Node 2: intelligence/pipeline.execute() → TrendData[]
├── quality.py            # Node 4: deterministic quality gate (no LLM)
└── workers/
    ├── schemas.py         # Pydantic output models for all workers
    ├── company_agent.py   # Company enrichment (called from lead_gen_node)
    ├── contact_agent.py   # ContactFinder: TREND_ROLE_MAPPING + Apollo + Hunter
    ├── email_agent.py     # EmailGenerator: structured LLM email drafting
    ├── impact_agent.py    # ImpactAnalyzer: per-trend impact analysis
    ├── impact_council.py  # Single structured LLM call → ImpactCouncilResult
    └── lead_validator.py  # LLM quality gate for company-trend pairing
```

---

## Pipeline Flow

```
START
  └─> source_intel_node
        ├─ (0 articles) ──────────────────────────────────> learning_update_node
        └─ (articles found) ──> analysis_node
                                    └─> impact_node
                                          └─> quality_validation_node
                                                ├─ (retry, max 2×) ──> analysis_node
                                                ├─ (no viable trends) ──> learning_update_node
                                                └─ (viable) ──> causal_council_node
                                                                    └─> lead_crystallize_node
                                                                          └─> lead_gen_node
                                                                                └─> learning_update_node
                                                                                      └─> END
```

`learning_update_node` always runs at end (even on skip) — learning fires on every execution.

---

## GraphState (TypedDict)

Defined in `orchestrator.py`:

```python
class GraphState(TypedDict):
    deps: Any                                          # AgentDeps instance
    run_id: str                                        # datetime-based unique ID
    trends: List[TrendData]
    impacts: List[ImpactAnalysis]
    companies: List[CompanyData]
    contacts: List[ContactData]
    outreach_emails: List[OutreachEmail]
    errors: Annotated[List[str], operator.add]         # merged across branches
    current_step: str
    retry_counts: Dict[str, int]
    agent_reasoning: Dict[str, str]
    stage_advisories: Annotated[List[Dict], operator.add]  # inter-stage communication
```

`Annotated[list, operator.add]` — LangGraph merges list fields by concatenation across parallel branches.

**Critical:** `stream_mode="updates"` — NEVER `"values"`. `AgentDeps` contains asyncio locks that cannot be serialized; `"values"` crashes on the full state dict.

---

## AgentDeps (`deps.py`)

Lazy-initialized shared state container. All properties initialize on first access, never on import.

```python
@dataclass
class AgentDeps:
    mock_mode: bool
    scope: Optional[DiscoveryScope]

    # Lazy @property tools:
    llm_service      # LLMService(full chain: OpenAI → GeminiDirect → VertexLlama → NVIDIA → Groq → OpenRouter → Ollama)
    llm_lite_service # LLMService(lite=True: OpenAINano → GeminiDirectLite → Groq → standard chain)
    tavily_tool      # TavilyTool — primary web search
    rss_tool         # RSSTool — 103 active RSS feeds
    apollo_tool      # ApolloTool — contact search (semaphore in tool: 3)
    hunter_tool      # HunterTool — email finding (semaphore in tool: 2)
    embedding_tool   # EmbeddingTool — NVIDIA/OpenAI embeddings
    article_cache    # ArticleCache — ChromaDB article store
    source_bandit    # SourceBandit — Thompson Sampling source ranking
    company_bandit   # CompanyRelevanceBandit — company-trend arm scoring
    meta_reasoner    # MetaReasoner — GUTTED, always returns empty stubs
    search_manager   # SearchManager — BM25 + DDG fallback
    recorder         # RunRecorder (None in mock mode)

    # Mutable working data set by agents:
    _articles, _embeddings, _event_distribution
    _trend_tree, _pipeline, _trend_data
    _impacts, _viable_impacts
    _companies, _contacts, _outreach, _person_profiles
    _causal_results, _lead_sheets, _signals
```

Semaphores are defined in the tool files (`apollo_tool.py: _APOLLO_SEM = Semaphore(3)`, `hunter_tool.py: _HUNTER_SEM = Semaphore(2)`), not in `AgentDeps`.

---

## Node 1: source_intel_node

`source_intel.py` — collects articles via bandit-prioritized sources.

Sources: Google News RSS + configured feeds (bandit-ordered), then Tavily web search if article count is insufficient. Full-content scrape + NVIDIA/OpenAI embeddings. Event type classification.

Conditional edge: 0 articles → skip directly to `learning_update_node`.

---

## Node 2: analysis_node

`analysis.py` calls `intelligence.pipeline.execute()`. The intelligence pipeline runs all math gates (dedup → NLI filter → entity extraction → similarity → clustering → validation) before any LLM call.

Results bridged via `_intelligence_clusters_to_trends()`:

```
strategic_score >= 0.50  →  severity = HIGH
strategic_score 0.25–0.50 →  severity = MEDIUM
strategic_score < 0.25   →  severity = LOW
(fallback to coherence if strategic_score == 0):
  coherence >= 0.70  →  HIGH
  coherence 0.50–0.70 → MEDIUM
  coherence < 0.50   →  LOW
```

Retries up to 2× with adjusted `dedup_title_threshold` / `coherence_min` if quality check fails.

---

## Node 3: impact_node

`workers/impact_council.py` — single structured LLM call with 4 analytical perspectives embedded in the system prompt (replaces multi-agent debate pattern — research: Smit et al. ICML 2024 shows MAD does not reliably outperform single-call baselines).

**`ImpactCouncilResult`** (defined in `workers/schemas.py`):
```python
class ImpactCouncilResult(BaseModel):
    perspectives: List[CouncilPerspective]      # one per analyst perspective
    consensus_reasoning: str                     # moderator synthesis
    debate_summary: str                          # key disagreements + resolution
    detailed_reasoning: str
    pitch_angle: str
    service_recommendations: List[ServiceRecommendation]
    evidence_citations: List[str]
    overall_confidence: float                    # 0.0-1.0
    affected_sectors: List[str]
    affected_company_types: List[str]
    pain_points: List[str]
    business_opportunities: List[str]
    target_roles: List[str]
```

4 perspectives in system prompt: `industry_analyst`, `strategy_consultant`, `risk_analyst`, `market_researcher`.

---

## Node 4: quality_validation_node

`quality.py` — **deterministic quality gate, no LLM call** (replaced LLM-based agent March 2026).

**Quality formula:**
```python
quality = 0.40 * mean_coherence + 0.30 * (1 - noise_rate) + 0.30 * mean_oss
```

**Trend check thresholds:**
- `mean_coherence < 0.40` → WARN / trigger retry if `>= 0.35`
- `n_clusters < 3` → WARN
- `min_coherence < 0.25` → WARN
- `mean_coherence < 0.35` → FAIL (no retry)

**Impact check:** filter by `council_confidence >= settings.min_trend_confidence_for_agents`. Fail-open: if ALL impacts below threshold → keep top 3 by confidence score.

**`QualityVerdict` fields:** `stage`, `passed`, `should_retry`, `items_passed`, `items_filtered`, `quality_score`, `issues`, `reasoning`

Conditional edge: `"analysis"` (retry), `"end"` → `learning_update_node` (no viable), `"lead_gen"` → `causal_council_node` (viable).

---

## OSS Score (`_compute_oss()` in `orchestrator.py`)

Operational Specificity Score — deterministic, no LLM. Measures how actionable a cluster is for B2B outreach:

```python
oss = 0.35 * has_entity    # named entity (company/person) present
    + 0.25 * has_action    # action verb: launch, acquir, rais, expand, partner, invest, hire, ...
    + 0.25 * has_numbers   # quantitative data: $X, Y%, YYYY year
    + 0.15 * has_industry  # industry classification set
```

---

## Node 5: causal_council_node

Multi-hop business impact tracer using pydantic-ai tool calling + BM25 KB search.

```
hop1: Companies directly NAMED in the article (evidence required)
hop2: Buyers, suppliers, or partners of hop1 companies
hop3: Downstream chain from hop2

For each hop:
  segment       = affected industry segment (e.g. "Steel importers")
  mechanism     = specific pain/opportunity explanation
  companies_found = real company names from BM25 KB search
  confidence    = 0.0–1.0
  lead_type     = "pain" | "opportunity" | "risk" | "intelligence"
```

Hops below confidence 0.35 are dropped in `lead_crystallize_node`.

---

## Node 6: lead_crystallize_node

In `orchestrator.py` — converts `CausalChainResult[]` → `LeadSheet[]`:

```
For each hop with confidence >= 0.35:
  1. Resolve segment → real company names (KB lookup + geo resolution)
  2. Fetch company-specific recent news (ChromaDB article_cache)
  3. Assign contact_role via _CONTACT_ROLES[event_type]
  4. Assign service_pitch via _SERVICES[(lead_type, event_type)]
  5. Generate opening_line (ready-to-use first sentence)

Output: LeadSheet[] sorted by (confidence DESC, urgency_weeks ASC)
```

---

## Node 7: lead_gen_node

`run_lead_gen()` in `leads.py` — 4 enrichment phases, each concurrent internally.

### Phase 1: Company Enrichment
- DB cache lookup (7-day TTL)
- Domain resolution via `extract_clean_domain()`
- Batch enrich via `company_enricher.enrich()` — Semaphore(5), 8s timeout

### Phase 2: Contact Finding (`contact_agent.py`)
```
1. TREND_ROLE_MAP lookup → target roles for this event type
2. Apollo: search_people_at_company(domain, roles, limit) — _APOLLO_SEM = Semaphore(3)
3. Hunter: find_email(domain, full_name) — _HUNTER_SEM = Semaphore(2) (fallback)
4. Filter: EMAIL_CONFIDENCE_THRESHOLD = 70
   → ContactResult[] sorted by confidence DESC
```

Contact matching: `match_roles_to_trend()` uses `trend_type` as primary key into `TREND_ROLE_MAPPING`, then scans `trend_title`/`pain_point`/`who_needs_help` for secondary keyword signals. Returns up to 8 deduplicated roles.

### TREND_ROLE_MAPPING (defined in `app/config.py`)

| Category key | Target roles (ordered, most senior first) |
|--------------|------------------------------------------|
| `regulation` / `policy` | CEO, Chief Strategy Officer, VP Strategy, Director Biz Dev |
| `trade` | VP Supply Chain, Procurement Director, CPO, Director Sourcing |
| `market_shift` / `competition` | CMO, VP Marketing, Chief Strategy Officer |
| `technology` / `digital_transformation` | CTO, VP Engineering, Chief Digital Officer |
| `expansion` / `market_expansion` | CEO, VP Business Development, CSO |
| `supply_chain` | COO, VP Operations, CPO, Director Supply Chain |
| `funding` | CEO, CFO, Chief Strategy Officer, VP Corporate Development |
| `cybersecurity` | CISO, VP Security, Head of Information Security |
| `compliance` / `data_privacy` | CCO, VP Legal, DPO, General Counsel |
| `ai_adoption` | CTO, Chief AI Officer, VP Engineering, Head of Data Science |
| `cloud_migration` | CTO, VP Infrastructure, Head of Cloud, IT Director |
| `cost_reduction` | CFO, COO, VP Operations, Head of Procurement |
| `sustainability` | Chief Sustainability Officer, VP Sustainability, Head of ESG |
| `talent` | CHRO, VP People, Head of Talent, HR Director |
| `default` | CEO, CTO, CFO, VP Operations, Head of Strategy |

### Phase 3: Email Generation (`email_agent.py`)

Structured LLM call (GPT-4.1-mini via `run_structured()`). Company news snippet injected into prompt so email references the triggering event.

### Phase 4: Person Profiles

`_build_person_profiles()` → `PersonProfile[]`
- `seniority_tier`: `"decision_maker"` | `"influencer"` | `"gatekeeper"`
- `reach_score`: composite of email confidence + verified + linkedin + tier + role_relevance
- Sorted: decision_maker first, then by reach_score DESC

---

## Node 8: learning_update_node

Always the final node. Publishes `HopSignal` data to `LearningSignalBus`, then runs all learning loop updates in sequence:

1. Source Bandit update (`source_bandit.update_from_run()`)
2. Company Bandit update (reward from lead sheet outcomes)
3. Contact Bandit update (reward = `confidence × 0.5` + 0.3 if company CIN found)
4. Auto-feedback quality scoring and `bus.publish_auto_feedback()`
5. Adaptive thresholds publish (`bus.publish_adaptive_thresholds()`)
6. Backward cascade signals (`bus.publish_backward_signals()`)
7. `bus.compute_derived_signals()` → system_confidence, exploration_budget
8. `bus.save()` → persist to `data/signal_bus.json`

---

## 3 Discovery Modes (`DiscoveryScope.mode`)

| Mode | Scope | Sources |
|------|-------|---------|
| `COMPANY_FIRST` | `scope.companies` set | Company-specific Tavily search + filtered RSS |
| `INDUSTRY_FIRST` | `scope.industry` set | Industry-keyword RSS + Tavily |
| `REPORT_DRIVEN` | `scope.report_text` set | NLI against user report text as hypothesis |

---

## Worker Concurrency

Semaphores are module-level constants in each tool file:

```python
# app/tools/crm/apollo_tool.py
_APOLLO_SEM = asyncio.Semaphore(3)

# app/tools/crm/hunter_tool.py
_HUNTER_SEM = asyncio.Semaphore(2)
```

Workers are standard async functions, not LangGraph nodes.

---

## Key Rules

- `stream_mode="updates"` — NEVER `"values"` (AgentDeps serialization crash)
- `AgentDeps` uses lazy `@property` — NEVER import heavy deps at module level
- All CRM imports: `from app.tools.crm.*` — never at flat path
- Provider calls: always through `deps.llm_service` or `deps.llm_lite_service`
- Learning signals → `signal_bus.py` only — no direct cross-loop imports
- `from __future__ import annotations` is BANNED with local classes + `get_type_hints()`
- `MetaReasoner` is present for type compat only — never make LLM calls through it
