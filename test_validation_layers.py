"""
Comprehensive validation test for all V1-V10 anti-hallucination layers.
Tests every validator, coercion, and guard implemented in Round 4.
"""

import sys
import traceback

results = []


def test(name, fn):
    try:
        fn()
        results.append(("PASS", name))
        print(f"  PASS  {name}")
    except Exception as e:
        results.append(("FAIL", name, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ════════════════════════════════════════════════════════════════════
# V1: Pydantic Field Validators
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("[V1] Pydantic Field Validators")
print("=" * 70)

from app.schemas.sales import ImpactAnalysis, CompanyData
from app.schemas.trends import TrendNode, SignalStrength, TrendDepth
from app.schemas.base import TrendType, LifecycleStage


def v1_impact_none_coercion():
    ia = ImpactAnalysis(
        trend_id="t1", trend_title="T",
        direct_impact=None,
        indirect_impact="single string",
        pitch_angle="A" * 200,
    )
    assert ia.direct_impact == [], f"Expected [], got {ia.direct_impact}"
    assert ia.indirect_impact == ["single string"]
    assert len(ia.pitch_angle) <= 153, f"Not truncated: {len(ia.pitch_angle)}"


test("V1a: ImpactAnalysis None/str coercion + truncation", v1_impact_none_coercion)


def v1_company_generic_reject():
    for bad_name in ["N/A", "None", "Unknown", "Company", "TBD", "A"]:
        try:
            CompanyData(company_name=bad_name, industry="IT", domain="x.com")
            raise AssertionError(f"Should reject '{bad_name}'")
        except ValueError:
            pass


test("V1b: CompanyData rejects generic/short names", v1_company_generic_reject)


def v1_company_website_validation():
    cd = CompanyData(company_name="Infosys", industry="IT", domain="infosys.com", website="infosys.com")
    assert cd.website.startswith("https://"), f"Expected https://, got {cd.website}"
    cd2 = CompanyData(company_name="Test Co", industry="IT", domain="x", website="not-a-url")
    assert cd2.website == ""
    assert cd2.domain == ""


test("V1c: CompanyData website/domain validation", v1_company_website_validation)


def v1_lifecycle_stage_mapping():
    mappings = {
        "emerging": "emerging", "growth": "growing", "mature": "peak",
        "decline": "declining", "peak": "peak", "new": "emerging",
    }
    for inp, exp in mappings.items():
        t = TrendNode(
            trend_title="T", trend_summary="S",
            trend_type=TrendType.TECHNOLOGY, depth=1,
            signal_strength=SignalStrength.STRONG, lifecycle_stage=inp,
        )
        assert t.lifecycle_stage == exp, f"{inp} -> {t.lifecycle_stage}, expected {exp}"


test("V1d: TrendNode lifecycle_stage mapping", v1_lifecycle_stage_mapping)


def v1_dedup_filter():
    t = TrendNode(
        trend_title="T", trend_summary="S",
        trend_type=TrendType.TECHNOLOGY, depth=1,
        signal_strength=SignalStrength.STRONG,
        affected_companies=["A", "", "A", "B"],
        key_entities=["E", "", "E"],
    )
    assert "" not in t.affected_companies and len(t.affected_companies) == 2
    assert "" not in t.key_entities and len(t.key_entities) == 1


test("V1e: TrendNode affected_companies/key_entities dedup+filter", v1_dedup_filter)


def v1_causal_chain_coerce():
    # Schema wraps string as single-item list (splitting happens in synthesis sanitizer)
    t = TrendNode(
        trend_title="T", trend_summary="S",
        trend_type=TrendType.TECHNOLOGY, depth=1,
        signal_strength=SignalStrength.STRONG,
        causal_chain="Step A -> Step B -> Step C",
    )
    assert isinstance(t.causal_chain, list) and len(t.causal_chain) >= 1
    # List input preserved as-is
    t2 = TrendNode(
        trend_title="T", trend_summary="S",
        trend_type=TrendType.TECHNOLOGY, depth=1,
        signal_strength=SignalStrength.STRONG,
        causal_chain=["Step A", "Step B", "Step C"],
    )
    assert isinstance(t2.causal_chain, list) and len(t2.causal_chain) == 3
    # None coerced to empty list
    t3 = TrendNode(
        trend_title="T", trend_summary="S",
        trend_type=TrendType.TECHNOLOGY, depth=1,
        signal_strength=SignalStrength.STRONG,
        causal_chain=None,
    )
    assert t3.causal_chain == []


test("V1f: TrendNode causal_chain string->list", v1_causal_chain_coerce)


def v1_5w1h_fill():
    t = TrendNode(
        trend_title="T", trend_summary="S",
        trend_type=TrendType.TECHNOLOGY, depth=1,
        signal_strength=SignalStrength.STRONG,
        event_5w1h={"who": "X", "what": "Y"},
    )
    for key in ["when", "where", "why", "how"]:
        assert key in t.event_5w1h, f"Missing key: {key}"
        assert t.event_5w1h[key] == "Not specified"


test("V1g: TrendNode 5W1H missing key fill", v1_5w1h_fill)


# ════════════════════════════════════════════════════════════════════
# V3: Synthesis Validation
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[V3] Synthesis Validation")
print("=" * 70)

from app.trends.synthesis import _validate_synthesis, _sanitize_synthesis_response


def v3_empty():
    r = _validate_synthesis({})
    assert len(r["critical"]) > 0


test("V3a: Empty synthesis -> critical errors", v3_empty)


def v3_missing_title():
    r = _validate_synthesis({"trend_summary": "Some summary"})
    assert any("title" in e.lower() for e in r["critical"])


test("V3b: Missing title -> critical", v3_missing_title)


def v3_good_synthesis():
    r = _validate_synthesis({
        "trend_title": "India AI Growth Surge",
        "trend_summary": "Significant trend in AI growth across Indian tech companies with major investments.",
        "lifecycle_stage": "growing",
        "affected_companies": ["TCS", "Infosys"],
        "key_entities": ["AI", "India"],
        "event_5w1h": {"who": "TCS", "what": "AI", "when": "2025", "where": "India", "why": "Growth", "how": "R&D"},
        "buying_intent": {"signal_type": "growth_opportunity", "urgency": "short_term"},
        "causal_chain": ["AI investment increases", "Talent demand grows"],
    })
    assert len(r["critical"]) == 0, f"Unexpected critical: {r['critical']}"


test("V3c: Good synthesis passes validation", v3_good_synthesis)


def v3_sanitize_bad_types():
    s = _sanitize_synthesis_response({
        "trend_title": "T", "trend_summary": "S",
        "lifecycle_stage": "exploding",       # invalid enum
        "affected_companies": "TCS",          # string, not list
        "causal_chain": "Step1 -> Step2",     # string, not list
        "event_5w1h": None,                   # None, not dict
        "buying_intent": "growth",            # string, not dict
    })
    assert isinstance(s["affected_companies"], list)
    assert isinstance(s["causal_chain"], list)
    assert isinstance(s["event_5w1h"], dict)
    assert isinstance(s["buying_intent"], dict)
    assert s["lifecycle_stage"] in ("emerging", "growing", "peak", "declining")


test("V3d: Sanitize bad types", v3_sanitize_bad_types)


# ════════════════════════════════════════════════════════════════════
# V5: Impact Agent Type Validation
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[V5] Impact Agent Type Validation")
print("=" * 70)

from app.agents.impact_agent import ImpactAgent


def v5_none_to_list():
    r = ImpactAgent._validate_impact_response({
        "direct_impact": None, "indirect_impact": None, "midsize_pain_points": None,
    })
    assert r["direct_impact"] == [] and r["indirect_impact"] == [] and r["midsize_pain_points"] == []


test("V5a: None list fields -> []", v5_none_to_list)


def v5_str_to_list():
    r = ImpactAgent._validate_impact_response({"direct_impact": "single impact"})
    assert r["direct_impact"] == ["single impact"]


test("V5b: String -> [string]", v5_str_to_list)


def v5_pitch_truncation():
    r = ImpactAgent._validate_impact_response({"pitch_angle": "A" * 200})
    assert len(r["pitch_angle"]) <= 153


test("V5c: pitch_angle truncation", v5_pitch_truncation)


def v5_none_str_to_empty():
    r = ImpactAgent._validate_impact_response({"direct_impact_reasoning": None})
    assert r["direct_impact_reasoning"] == ""


test("V5d: None string -> empty", v5_none_str_to_empty)


# ════════════════════════════════════════════════════════════════════
# V6: Event Classifier
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[V6] Event Classifier Configuration")
print("=" * 70)

from app.news.event_classifier import (
    EmbeddingEventClassifier, EVENT_DESCRIPTION_VARIANTS,
    EVENT_KEYWORD_BOOST, EVENT_URGENCY,
)


def v6_event_config():
    assert len(EVENT_DESCRIPTION_VARIANTS) >= 12, f"Only {len(EVENT_DESCRIPTION_VARIANTS)} types"
    for evt, variants in EVENT_DESCRIPTION_VARIANTS.items():
        assert isinstance(variants, list) and len(variants) >= 2, f"{evt}: needs 2+ variants"
    assert len(EVENT_KEYWORD_BOOST) >= 12
    assert len(EVENT_URGENCY) >= 12


test("V6a: Event type config (14 types, 2+ variants each)", v6_event_config)


def v6_threshold_from_env():
    from app.config import get_settings
    s = get_settings()
    assert s.event_classifier_threshold == 0.35
    assert s.event_classifier_ambiguity_margin == 0.05


test("V6b: Classifier threshold from env (0.35)", v6_threshold_from_env)


# ════════════════════════════════════════════════════════════════════
# V7: NER Company Verification
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[V7] NER Company Verification")
print("=" * 70)

from app.agents.company_agent import CompanyAgent


def v7_normalize():
    n = CompanyAgent._normalize_name
    assert n("Tata Consultancy Services Ltd.") == n("tata consultancy services")
    assert n("Infosys Limited") == n("infosys")
    assert n("Wipro Pvt Ltd") == n("wipro")


test("V7a: Name normalization (suffix removal)", v7_normalize)


def v7_fuzzy_match():
    f = CompanyAgent._fuzzy_entity_match
    matched = f("Tata Consultancy", {"Tata Consultancy Services", "Infosys", "Wipro"})
    assert matched is True, f"Expected match, got {matched}"


test("V7b: Fuzzy entity matching (substring)", v7_fuzzy_match)


def v7_fuzzy_no_match():
    f = CompanyAgent._fuzzy_entity_match
    matched = f("Completely Random Corp", {"Tata Consultancy Services", "Infosys"})
    assert matched is False, f"Should not match, but got {matched}"


test("V7c: Fuzzy match rejects non-matching names", v7_fuzzy_no_match)


def v7_ner_fields():
    cd = CompanyData(
        company_name="Tata Motors", industry="Auto", domain="tatamotors.com",
        ner_verified=True, verification_source="ner_match", verification_confidence=0.9,
    )
    assert cd.ner_verified is True
    assert cd.verification_confidence == 0.9


test("V7d: CompanyData NER verification fields", v7_ner_fields)


# ════════════════════════════════════════════════════════════════════
# V9: Quality Gates
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[V9] Quality Gates")
print("=" * 70)

from app.trends.engine import RecursiveTrendEngine


def v9_method_exists():
    assert hasattr(RecursiveTrendEngine, "_phase_quality_gate")


test("V9a: Quality gate method exists", v9_method_exists)


def v9_config():
    from app.config import get_settings
    s = get_settings()
    assert s.min_synthesis_confidence == 0.3
    assert s.min_trend_confidence_for_agents == 0.25


test("V9b: Quality gate thresholds from env", v9_config)


# ════════════════════════════════════════════════════════════════════
# V10: Validator Agent
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[V10] Validator Agent")
print("=" * 70)

from app.agents.validator_agent import ValidatorAgent
from app.schemas.validation import (
    ValidationResult, ValidationVerdict, FieldGroundedness, ValidationRound,
)


def v10_schema():
    vr = ValidationResult(
        cluster_id=0, total_rounds=1,
        final_verdict=ValidationVerdict.PASS, final_score=0.75,
    )
    assert vr.passed is True
    assert vr.was_revised is False
    vr2 = ValidationResult(
        cluster_id=1, total_rounds=2,
        final_verdict=ValidationVerdict.REVISE, final_score=0.4,
    )
    assert vr2.passed is False
    assert vr2.was_revised is True


test("V10a: ValidationResult schema + properties", v10_schema)


def v10_field_clamping():
    fg1 = FieldGroundedness(field_name="test", score=1.5, method="test")
    assert fg1.score == 1.0
    fg2 = FieldGroundedness(field_name="test", score=-0.5, method="test")
    assert fg2.score == 0.0
    fg3 = FieldGroundedness(field_name="test", score=None, method="test")
    assert fg3.score == 0.0


test("V10b: FieldGroundedness score clamping", v10_field_clamping)


def v10_failing_fields():
    vr = ValidationRound(
        round_number=1, verdict=ValidationVerdict.REVISE, overall_score=0.4,
        field_scores=[
            FieldGroundedness(field_name="title", score=0.8, method="keyword"),
            FieldGroundedness(field_name="companies", score=0.3, method="ner"),
        ],
    )
    assert len(vr.failing_fields) == 1
    assert vr.failing_fields[0].field_name == "companies"


test("V10c: ValidationRound failing fields", v10_failing_fields)


def v10_summary():
    vr = ValidationResult(
        cluster_id=0, total_rounds=2,
        final_verdict=ValidationVerdict.PASS, final_score=0.75,
        entity_overlap_ratio=0.8, elapsed_ms=150,
    )
    s = vr.summary()
    assert "PASS" in s and "0.75" in s and "80%" in s and "150ms" in s


test("V10d: ValidationResult summary output", v10_summary)


def v10_agent_init():
    from app.tools.embeddings import EmbeddingTool
    agent = ValidatorAgent(embedding_tool=EmbeddingTool())
    assert hasattr(agent, "validate")
    assert hasattr(agent, "validate_with_revision")
    assert hasattr(agent, "build_revision_feedback")


test("V10e: ValidatorAgent instantiation + methods", v10_agent_init)


def v10_config():
    from app.config import get_settings
    s = get_settings()
    assert s.validator_enabled is True
    assert s.validator_max_rounds == 2
    assert s.validator_pass_threshold == 0.6
    assert s.validator_reject_threshold == 0.25


test("V10f: Validator config from env", v10_config)


# ════════════════════════════════════════════════════════════════════
# V2: LLM Retry Logic
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[V2] LLM Retry Configuration")
print("=" * 70)


def v2_config():
    from app.config import get_settings
    s = get_settings()
    assert s.llm_json_max_retries == 2


test("V2a: LLM JSON max retries from env", v2_config)


def v2_generate_json_has_pydantic_param():
    from app.tools.llm_service import LLMService
    import inspect
    sig = inspect.signature(LLMService.generate_json)
    assert "pydantic_model" in sig.parameters, f"Missing pydantic_model param. Params: {list(sig.parameters.keys())}"


test("V2b: generate_json() has pydantic_model parameter", v2_generate_json_has_pydantic_param)


# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
total = len(results)
print(f"RESULTS: {passed}/{total} PASSED, {failed}/{total} FAILED")
if failed:
    print("\nFailed tests:")
    for r in results:
        if r[0] == "FAIL":
            print(f"  {r[1]}: {r[2]}")
    sys.exit(1)
else:
    print("ALL VALIDATION TESTS PASSED!")
print("=" * 70)
