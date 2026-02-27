# app/agents/ -- LangGraph Pipeline Agents

The agent layer implements an 8-step sales intelligence pipeline using LangGraph `StateGraph` with typed state passing and pydantic-ai for LLM interaction.

## Orchestrator

`orchestrator.py` defines the LangGraph `StateGraph` that wires all pipeline steps:

```
START -> source_intel -> analysis -> impact -> quality_validation
                                                      |
                                               [retry or pass]
                                                      |
                                               causal_council -> lead_crystallize -> lead_gen -> learning_update -> END
```

### GraphState (TypedDict)

```python
class GraphState(TypedDict):
    deps: Any                              # AgentDeps instance
    run_id: str                            # Unique run identifier
    trends: List[TrendData]
    impacts: List[ImpactAnalysis]
    companies: List[CompanyData]
    contacts: List[ContactData]
    outreach_emails: List[OutreachEmail]
    errors: Annotated[List[str], operator.add]
    current_step: str
    retry_counts: Dict[str, int]
    agent_reasoning: Dict[str, str]
```

The `errors` field uses `operator.add` for append-only accumulation across nodes.

## Agent Types

### Top-Level Agents

| File | Role |
|------|------|
| `orchestrator.py` | StateGraph definition, node wiring, `run_pipeline()` entry point |
| `source_intel.py` | RSS ingestion, event classification, dedup |
| `analysis.py` | Embed + Leiden cluster + coherence + synthesis |
| `market_impact.py` | Per-trend sector impact analysis |
| `quality.py` | Confidence gate, retry logic |
| `lead_gen.py` | Company discovery + contact finding + email generation |
| `lead_crystallizer.py` | Convert causal hops to `LeadSheet` call sheets |
| `agent_validator.py` | Validation utilities |
| `company_relevance_bandit.py` | Thompson Sampling for company targeting |

### Worker Agents (`workers/`)

Specialized single-task agents called by orchestrator nodes:

| File | Task |
|------|------|
| `impact_agent.py` | LLM-based trend impact analysis |
| `company_agent.py` | Company discovery via SearXNG + NER verification |
| `contact_agent.py` | Decision-maker finding via Apollo |
| `email_agent.py` | Email finding (Apollo + Hunter) + pitch generation |
| `validator_agent.py` | Quality validation checks |

### Council Agents (`workers/council/`)

Multi-agent validation layer for quality control:

| File | Role |
|------|------|
| `article_triage.py` | Pre-filters articles before clustering |
| `causal_council.py` | 4-agent causal reasoning funnel (PreFilter -> Mechanism -> Cascade -> Evidence) |
| `trend_validator.py` | Classifies trends as MAJOR/SUB/MICRO/NOISE |
| `lead_validator.py` | Validates lead quality before output |
| `schemas.py` | Pydantic models for council results |

## AgentDeps Dependency Injection

All agents share a single `AgentDeps` instance (defined in `deps.py`) passed through `GraphState["deps"]`. It provides lazy-initialized access to all tools:

```python
@dataclass
class AgentDeps:
    mock_mode: bool = False
    log_callback: Optional[object] = None

    # Lazy-initialized tools (created on first access)
    @property
    def llm_service(self) -> LLMService: ...
    @property
    def embedding_tool(self) -> EmbeddingTool: ...
    @property
    def apollo_tool(self) -> ApolloTool: ...
    @property
    def source_bandit(self) -> SourceBandit: ...
    @property
    def search_manager(self) -> SearchManager: ...

    # Mutable working data (set by agents during execution)
    _articles: List[Any]
    _trend_tree: Optional[TrendTree]
    _lead_sheets: List[LeadSheet]
    _recorder: Optional[RunRecorder]  # captures snapshots for mock replay
```

Key design: tools are imported and instantiated only when first accessed, keeping startup fast and avoiding circular imports.

## Adding a New Worker Agent

1. Create `app/agents/workers/my_agent.py`:
   ```python
   async def run_my_agent(deps: AgentDeps, input_data: ...) -> ...:
       llm = deps.llm_service
       result = await llm.generate_structured(MyOutput, prompt, system_prompt)
       return result
   ```

2. Add a node function in `orchestrator.py`:
   ```python
   async def my_step(state: GraphState) -> dict:
       deps = state["deps"]
       result = await run_my_agent(deps, state["trends"])
       return {"my_output": result, "current_step": "my_step"}
   ```

3. Wire into the graph:
   ```python
   graph.add_node("my_step", my_step)
   graph.add_edge("previous_step", "my_step")
   ```

4. Add output field to `GraphState` TypedDict.

## Important Constraints

- **stream_mode="updates"**: The orchestrator must use `"updates"` not `"values"` to avoid msgpack serialization of `AgentDeps` (contains ChromaDB, LLM models).
- **Provider reset**: Each API run must call `provider_health.reset_for_new_run()`, `ProviderManager.reset_cooldowns()`, and `LLMService.clear_cache()` before starting.
- **Recording**: Real runs capture step snapshots to `data/recordings/{run_id}/` for mock replay.
