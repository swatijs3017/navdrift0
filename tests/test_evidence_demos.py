"""
tests/test_evidence_demos.py — sanity tests for the evidence-generation demo
scripts (edge/calibration_evidence_demo.py, edge/hmm_viterbi_evidence_demo.py).

These check the demo scripts run and produce internally consistent,
correctly-labeled output — not new algorithmic behavior (that's covered by
the underlying modules' own test suites).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.calibration_evidence_demo import run_all_fixtures
from edge.hmm_viterbi_evidence_demo import run_demo


def test_calibration_evidence_all_cases_present():
    cases = run_all_fixtures()
    case_names = {c["case"] for c in cases}
    assert "flat_phone_stationary" in case_names
    assert "known_roll_15deg" in case_names
    assert "gnss_yaw_reliable_moving" in case_names
    assert "gnss_yaw_unreliable_too_slow" in case_names
    assert "fallback_insufficient_low_motion" in case_names
    assert "fallback_accel_magnitude_far_from_g" in case_names


def test_calibration_evidence_fixtures_labeled_deterministic():
    cases = run_all_fixtures()
    for c in cases:
        assert c["kind"] == "deterministic_software_fixture"


def test_calibration_evidence_known_roll_recovered():
    cases = run_all_fixtures()
    case = next(c for c in cases if c["case"] == "known_roll_15deg")
    import math
    roll_deg = math.degrees(case["result"]["roll_rad"])
    assert abs(roll_deg - 15.0) < 1.0


def test_calibration_evidence_unreliable_gnss_leaves_yaw_unset():
    cases = run_all_fixtures()
    case = next(c for c in cases if c["case"] == "gnss_yaw_unreliable_too_slow")
    assert case["result"]["yaw_rad"] is None


def test_calibration_evidence_fallback_cases_are_invalid():
    cases = run_all_fixtures()
    for name in ["fallback_insufficient_low_motion", "fallback_accel_magnitude_far_from_g"]:
        case = next(c for c in cases if c["case"] == name)
        assert case["result"]["valid"] is False


def test_hmm_viterbi_evidence_demonstrates_correction():
    """The whole point of this evidence artifact: the Viterbi best path
    must differ from the local greedy pick at tick 0 (correcting it) and
    then agree with it for the rest of the run."""
    trace, final_trace = run_demo()
    assert len(final_trace) == 6
    assert final_trace[0]["local_greedy_pick"] == "decoy_rd"
    assert final_trace[0]["viterbi_best_path_pick"] == "true_rd"
    assert final_trace[0]["corrected"] is True
    for row in final_trace[1:]:
        assert row["corrected"] is False
        assert row["viterbi_best_path_pick"] == "true_rd"


def test_hmm_viterbi_evidence_trace_has_real_scores():
    trace, final_trace = run_demo()
    for row in trace:
        for cs in row["candidate_scores"]:
            assert isinstance(cs["emission_log_prob"], float)
