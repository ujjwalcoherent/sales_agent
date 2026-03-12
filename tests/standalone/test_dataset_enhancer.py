"""
Standalone tests for app/learning/dataset_enhancer.py

Tests _add_example quality controls (dedup, class balance, size cap, truncation,
hash persistence). Bootstrap and SetFit methods were removed (March 2026 cleanup).

Run:
    venv/Scripts/python.exe tests/standalone/test_dataset_enhancer.py
"""
import hashlib
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

# Fix Windows console encoding only when run directly (not under pytest)
if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ── test helpers ──────────────────────────────────────────────────────────────

_PASS = "[PASS]"
_FAIL = "[FAIL]"

_results: List[Tuple[str, bool, str]] = []


def assert_test(name: str, condition: bool, detail: str = "") -> None:
    status = _PASS if condition else _FAIL
    _results.append((name, condition, detail))
    tag = f"{status} {name}"
    print(tag + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ── helper: isolated enhancer instance ────────────────────────────────────────

def _make_enhancer(tmp_dir: Path):
    """Return a DatasetEnhancer that writes to tmp_dir, not data/."""
    from app.learning.dataset_enhancer import DatasetEnhancer
    enhancer = DatasetEnhancer.__new__(DatasetEnhancer)
    enhancer._dataset_path = tmp_dir / "dynamic_dataset.jsonl"
    enhancer._stats_path = tmp_dir / "dataset_stats.json"
    enhancer._seen_hashes = set()
    enhancer._cached_positives = 0
    enhancer._cached_negatives = 0
    return enhancer


# ══════════════════════════════════════════════════════════════════════════════
# _add_example quality controls
# ══════════════════════════════════════════════════════════════════════════════

def test_add_example_deduplication() -> None:
    section("Part 1: RFT — Deduplication via MD5 hash")
    tmp = Path(tempfile.mkdtemp())
    try:
        enhancer = _make_enhancer(tmp)
        text = "TechCorp raises $500M in Series E to expand cloud operations globally."

        first = enhancer._add_example(text, label=1, source="test", confidence=0.9)
        second = enhancer._add_example(text, label=1, source="test", confidence=0.9)
        third = enhancer._add_example(text, label=1, source="test", confidence=0.9)

        assert_test("dedup: first add returns True", first)
        assert_test("dedup: second add returns False (duplicate)", not second)
        assert_test("dedup: third add returns False (duplicate)", not third)

        records = []
        with open(enhancer._dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        assert_test("dedup: only 1 record written to file", len(records) == 1, f"count={len(records)}")

        expected_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert_test(
            "dedup: hash stored in _seen_hashes",
            expected_hash in enhancer._seen_hashes,
        )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_add_example_class_balance() -> None:
    section("Part 2: RFT — MAX_CLASS_RATIO enforcement (2:1)")
    from app.learning.dataset_enhancer import MAX_CLASS_RATIO

    tmp = Path(tempfile.mkdtemp())
    try:
        enhancer = _make_enhancer(tmp)

        enhancer._add_example("Cricket team wins championship in dramatic final.", label=0,
                               source="test", confidence=0.9)
        r1 = enhancer._add_example("FinTech startup closes $20M seed round.", label=1,
                                   source="test", confidence=0.9)
        r2 = enhancer._add_example("SaaS company secures $40M Series B funding.", label=1,
                                   source="test", confidence=0.9)
        r3 = enhancer._add_example("CloudCorp announces new enterprise product.", label=1,
                                   source="test", confidence=0.9)

        stats = enhancer.get_stats()
        assert_test("class_balance: first positive accepted", r1,
                    f"pos={stats['positives']} neg={stats['negatives']}")
        assert_test("class_balance: second positive accepted (ratio at boundary)", r2)
        assert_test("class_balance: third positive rejected (would exceed 2:1 ratio)", not r3,
                    f"pos={stats['positives']} neg={stats['negatives']} MAX_CLASS_RATIO={MAX_CLASS_RATIO}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_add_example_size_cap() -> None:
    section("Part 3: RFT — MAX_DATASET_SIZE cap")
    from app.learning.dataset_enhancer import MAX_DATASET_SIZE

    tmp = Path(tempfile.mkdtemp())
    try:
        enhancer = _make_enhancer(tmp)

        import app.learning.dataset_enhancer as de_mod
        original_max = de_mod.MAX_DATASET_SIZE
        de_mod.MAX_DATASET_SIZE = 5

        try:
            added = 0
            for i in range(10):
                label = i % 2
                ok = enhancer._add_example(
                    f"Unique text entry number {i} for testing the size cap enforcement.",
                    label=label, source="test", confidence=0.9,
                )
                if ok:
                    added += 1

            assert_test("size_cap: total added capped at MAX_DATASET_SIZE=5", added <= 5,
                        f"added={added}")
            stats = enhancer.get_stats()
            assert_test("size_cap: dataset total <= 5", stats["total"] <= 5,
                        f"total={stats['total']}")

        finally:
            de_mod.MAX_DATASET_SIZE = original_max

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_add_example_text_truncation() -> None:
    section("Part 4: RFT — Text truncated to 512 chars in file")
    tmp = Path(tempfile.mkdtemp())
    try:
        enhancer = _make_enhancer(tmp)
        long_text = "Company acquires rival for $1B deal. " * 50  # ~1850 chars

        enhancer._add_example(long_text, label=1, source="test", confidence=0.9)

        with open(enhancer._dataset_path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        stored_len = len(record.get("text", ""))
        assert_test("text_truncation: stored text <= 512 chars", stored_len <= 512,
                    f"stored={stored_len}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_seen_hashes_on_init() -> None:
    section("Part 5: RFT — _load_seen_hashes pre-loads dedup state from file")
    tmp = Path(tempfile.mkdtemp())
    try:
        e1 = _make_enhancer(tmp)
        for i in range(5):
            e1._add_example(f"Unique test article {i} about enterprise deals.", label=i % 2,
                            source="test", confidence=0.9)

        hashes_e1 = set(e1._seen_hashes)

        e2 = _make_enhancer(tmp)
        e2._load_seen_hashes()

        assert_test("_load_seen_hashes: new instance loads all prior hashes",
                    hashes_e1 == e2._seen_hashes,
                    f"e1={len(hashes_e1)} e2={len(e2._seen_hashes)}")

        re_added = 0
        for i in range(5):
            ok = e2._add_example(f"Unique test article {i} about enterprise deals.", label=i % 2,
                                  source="test", confidence=0.9)
            if ok:
                re_added += 1

        assert_test("_load_seen_hashes: second instance rejects all 5 already-stored texts",
                    re_added == 0, f"re_added={re_added}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all() -> None:
    print("\n" + "=" * 60)
    print("  DATASET ENHANCER — STANDALONE TEST SUITE")
    print("=" * 60)

    test_add_example_deduplication()
    test_add_example_class_balance()
    test_add_example_size_cap()
    test_add_example_text_truncation()
    test_load_seen_hashes_on_init()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(_results)} assertions")
    print("=" * 60)

    if failed:
        print("\nFailed assertions:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  {_FAIL} {name}" + (f" — {detail}" if detail else ""))
        sys.exit(1)
    else:
        print("\nAll assertions passed.")


if __name__ == "__main__":
    run_all()
