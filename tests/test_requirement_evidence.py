"""
tests/test_requirement_evidence.py — sanity tests for edge/requirement_evidence.py

Ensures the requirement -> evidence mapping stays honest and in sync with
the actual repository: every referenced file/test must exist, every status
must be one of the defined categories, and no requirement can silently
claim VALIDATED/IMPLEMENTED status while its limitation text says it is
data-blocked (a consistency check against self-contradiction).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.requirement_evidence import (
    REQUIREMENT_EVIDENCE, VALID_STATUSES, validate_evidence_records, to_json,
)


def test_all_records_reference_real_files_and_tests():
    problems = validate_evidence_records()
    assert problems == [], f"evidence references nonexistent files/tests: {problems}"


def test_all_statuses_are_from_the_defined_set():
    for rec in REQUIREMENT_EVIDENCE:
        assert rec.validation_type in VALID_STATUSES


def test_data_blocked_requirements_do_not_claim_validated_or_implemented():
    """A requirement whose limitation text says data is missing must not
    ALSO be marked VALIDATED or IMPLEMENTED — status must reflect the
    honest limitation, not just that some code exists."""
    for rec in REQUIREMENT_EVIDENCE:
        limitation_lower = rec.limitation.lower()
        mentions_missing_data = ("no raw io-vnbd" in limitation_lower
                                  or "not present in this repository" in limitation_lower
                                  or "does not have" in limitation_lower)
        if mentions_missing_data:
            assert rec.validation_type == "DATA_BLOCKED", (
                f"{rec.requirement} limitation mentions missing data but status is "
                f"{rec.validation_type}, not DATA_BLOCKED")


def test_every_requirement_has_a_non_empty_limitation():
    for rec in REQUIREMENT_EVIDENCE:
        assert rec.limitation.strip() != ""


def test_no_subjective_language_in_implementation_or_limitation_text():
    banned = ["excellent", "production-ready", "production ready", "state-of-the-art",
              "cutting-edge", "flawless", "perfect", "world-class"]
    for rec in REQUIREMENT_EVIDENCE:
        text = (rec.implementation + " " + rec.limitation).lower()
        for word in banned:
            assert word not in text, f"{rec.requirement} uses subjective language: '{word}'"


def test_to_json_round_trips_and_matches_record_count():
    import json
    data = json.loads(to_json())
    assert len(data) == len(REQUIREMENT_EVIDENCE)
    assert data[0]["requirement"] == REQUIREMENT_EVIDENCE[0].requirement


def test_requirement_ids_are_unique():
    ids = [rec.requirement for rec in REQUIREMENT_EVIDENCE]
    assert len(ids) == len(set(ids))
