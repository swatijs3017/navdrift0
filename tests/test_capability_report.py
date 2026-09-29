"""
tests/test_capability_report.py — sanity tests for edge/capability_report.py

Verifies the report generator produces internally consistent, reproducible
output without hardcoded/fabricated claims, and never uses banned
subjective language.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.capability_report import generate_report, KNOWN_ONNX_MODELS, NEW_OFFLINE_MODULES


def test_report_has_all_required_top_level_fields():
    report = generate_report(run_tests=False)
    for key in ["report_type", "status_categories_used", "requirement_status_counts",
                "requirements", "requirement_evidence_consistency_problems",
                "known_onnx_models_present", "new_offline_modules_present"]:
        assert key in report


def test_status_counts_sum_matches_requirement_count():
    report = generate_report(run_tests=False)
    assert sum(report["requirement_status_counts"].values()) == len(report["requirements"])


def test_no_evidence_consistency_problems():
    """This report must never claim evidence for a file/test that doesn't
    actually exist in the repository."""
    report = generate_report(run_tests=False)
    assert report["requirement_evidence_consistency_problems"] == []


def test_new_offline_modules_all_reported_present():
    report = generate_report(run_tests=False)
    for path in NEW_OFFLINE_MODULES:
        assert report["new_offline_modules_present"][path] is True, f"{path} reported missing"


def test_known_onnx_models_status_is_a_real_boolean_per_file():
    report = generate_report(run_tests=False)
    for path in KNOWN_ONNX_MODELS:
        assert isinstance(report["known_onnx_models_present"][path], bool)


def test_no_subjective_language_anywhere_in_report():
    banned = ["excellent", "production-ready", "production ready", "state-of-the-art",
              "cutting-edge", "flawless", "perfect", "world-class"]
    report = generate_report(run_tests=False)
    text = str(report).lower()
    for word in banned:
        assert word not in text, f"capability report contains subjective language: '{word}'"


def test_report_skips_test_run_when_disabled():
    report = generate_report(run_tests=False)
    assert report["evidence_test_run"] is None


def test_report_actually_runs_tests_when_enabled():
    report = generate_report(run_tests=True)
    assert report["evidence_test_run"]["ran"] is True
    assert report["evidence_test_run"]["failed"] == 0
