"""
tests/test_hmm_map_matcher.py — unit tests for edge/hmm_map_matcher.py

All road/candidate data here is SYNTHETIC and constructed by hand to have one
unambiguous correct answer (e.g. "given these candidates, the only path that
scores well is straight-line travel on segment A the whole time"). These
tests verify the ALGORITHM (candidate scoring, path accumulation, backpointer
reconstruction, gap handling) is correct, not real-world map-matching
accuracy — no real GNSS/road dataset is used or claimed here.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.hmm_map_matcher import (
    RoadCandidate, Observation, ViterbiMapMatcher, run_offline_match,
    emission_log_prob, transition_log_prob,
)


def _seg_point(seg_id, lat, lon, bearing_rad, dist_m):
    return RoadCandidate(seg_id=seg_id, lat=lat, lon=lon, bearing_rad=bearing_rad, dist_m=dist_m)


def test_emission_prefers_closer_and_aligned_candidate():
    obs = Observation(lat=10.0, lon=20.0, heading_rad=0.0, speed_mps=5.0, dt_s=1.0)
    near_aligned = _seg_point("A", 10.0001, 20.0, 0.0, dist_m=2.0)
    far_misaligned = _seg_point("B", 10.001, 20.001, math.pi / 2, dist_m=80.0)
    assert emission_log_prob(near_aligned, obs) > emission_log_prob(far_misaligned, obs)


def test_emission_ignores_heading_when_unknown():
    obs_no_heading = Observation(lat=10.0, lon=20.0, heading_rad=None, speed_mps=None, dt_s=1.0)
    c1 = _seg_point("A", 10.0, 20.0, 0.0, dist_m=5.0)
    c2 = _seg_point("A", 10.0, 20.0, math.pi, dist_m=5.0)  # same distance, opposite bearing
    # With heading unknown, distance-only score must be identical regardless of bearing.
    assert emission_log_prob(c1, obs_no_heading) == emission_log_prob(c2, obs_no_heading)


def test_transition_rewards_matching_speed_times_dt():
    prev = _seg_point("A", 10.0, 20.0, 0.0, dist_m=0.0)
    # ~11.1m north of prev at this latitude for a 0.0001 deg step (roughly).
    close_step = _seg_point("A", 10.0001, 20.0, 0.0, dist_m=0.0)
    far_step = _seg_point("A", 10.01, 20.0, 0.0, dist_m=0.0)
    obs = Observation(lat=10.0001, lon=20.0, heading_rad=0.0, speed_mps=11.0, dt_s=1.0)
    assert transition_log_prob(prev, close_step, obs) > transition_log_prob(prev, far_step, obs)


def test_transition_same_segment_bonus_without_speed():
    prev = _seg_point("A", 10.0, 20.0, 0.0, dist_m=0.0)
    same_seg = _seg_point("A", 10.0, 20.0, 0.0, dist_m=0.0)
    diff_seg = _seg_point("B", 10.0, 20.0, 0.0, dist_m=0.0)
    obs = Observation(lat=10.0, lon=20.0, heading_rad=None, speed_mps=None, dt_s=1.0)
    assert transition_log_prob(prev, same_seg, obs) > transition_log_prob(prev, diff_seg, obs)


def test_single_unambiguous_road_straight_line_path():
    """
    Synthetic fixture: a vehicle travels straight along segment 'main_st' for
    5 ticks. At each tick the only two candidates are the correct segment
    (near, aligned) and a decoy segment (far, misaligned). The best path must
    select 'main_st' at every single step.
    """
    def candidate_fn(obs: Observation):
        correct = _seg_point("main_st", obs.lat, obs.lon, 0.0, dist_m=1.0)
        decoy = _seg_point("side_st", obs.lat + 0.01, obs.lon + 0.01, math.pi / 2, dist_m=150.0)
        return [correct, decoy]

    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(5)
    ]
    path = run_offline_match(obs_stream, candidate_fn, history_window=50)
    assert len(path) == 5
    for state in path:
        assert state.had_candidates
        assert state.candidate.seg_id == "main_st"


def test_transition_continuity_breaks_ties_toward_consistent_segment():
    """
    Synthetic fixture: two candidates are EMISSION-tied (equal distance,
    equal heading agreement) at every tick, but only one of them is
    consistent with real sustained motion (matches speed*dt each step); the
    other silently teleports between unrelated segments. The Viterbi path
    (which accumulates transition score across the whole run) must prefer
    the physically consistent segment over the whole window, even though a
    single-tick greedy argmax over emission alone could not tell them apart.
    """
    def candidate_fn(obs: Observation):
        # Both candidates have identical emission (same dist_m, same bearing).
        consistent = _seg_point("consistent", obs.lat, obs.lon, 0.0, dist_m=5.0)
        # "erratic" jumps to a location uncorrelated with sustained forward motion.
        erratic_lat = 10.0 + 0.05 * math.sin(obs.lat * 1000.0)
        erratic = _seg_point("erratic", erratic_lat, obs.lon, 0.0, dist_m=5.0)
        return [consistent, erratic]

    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(8)
    ]
    path = run_offline_match(obs_stream, candidate_fn, history_window=50)
    consistent_count = sum(1 for s in path if s.had_candidates and s.candidate.seg_id == "consistent")
    assert consistent_count >= 6  # dominant majority of the window picks the physically consistent segment


def test_gap_with_no_candidates_does_not_crash_or_fabricate():
    def candidate_fn(obs: Observation):
        if 2 <= obs.lat < 3:  # sentinel marker unused; gap is driven by empty return below
            pass
        return []  # always no candidates — simulates total road-graph coverage gap

    obs_stream = [Observation(lat=10.0, lon=20.0, heading_rad=0.0, speed_mps=5.0, dt_s=1.0) for _ in range(3)]
    path = run_offline_match(obs_stream, candidate_fn, history_window=10)
    assert len(path) == 3
    for state in path:
        assert state.had_candidates is False
        assert state.candidate is None


def test_gap_recovers_after_candidates_return():
    calls = {"n": 0}

    def candidate_fn(obs: Observation):
        calls["n"] += 1
        if calls["n"] <= 2:
            return []  # gap for the first two steps
        return [_seg_point("recovered", obs.lat, obs.lon, 0.0, dist_m=1.0)]

    obs_stream = [Observation(lat=10.0, lon=20.0, heading_rad=0.0, speed_mps=5.0, dt_s=1.0) for _ in range(4)]
    path = run_offline_match(obs_stream, candidate_fn, history_window=10)
    assert path[0].had_candidates is False
    assert path[1].had_candidates is False
    assert path[2].had_candidates is True and path[2].candidate.seg_id == "recovered"
    assert path[3].had_candidates is True and path[3].candidate.seg_id == "recovered"


def test_bounded_history_window_is_respected():
    def candidate_fn(obs: Observation):
        return [_seg_point("only", obs.lat, obs.lon, 0.0, dist_m=1.0)]

    matcher = ViterbiMapMatcher(history_window=5)
    for i in range(20):
        obs = Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=5.0, dt_s=1.0)
        matcher.step(obs, candidate_fn(obs))
    assert len(matcher) == 5
    assert len(matcher.best_path()) == 5


def test_deterministic_reproducibility():
    def candidate_fn(obs: Observation):
        return [_seg_point("A", obs.lat, obs.lon, 0.0, dist_m=1.0),
                _seg_point("B", obs.lat + 0.0005, obs.lon, 0.3, dist_m=40.0)]

    obs_stream = [Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
                  for i in range(10)]
    path1 = run_offline_match(obs_stream, candidate_fn, history_window=50)
    path2 = run_offline_match(obs_stream, candidate_fn, history_window=50)
    assert [s.candidate.seg_id if s.had_candidates else None for s in path1] == \
           [s.candidate.seg_id if s.had_candidates else None for s in path2]
    assert [s.log_prob for s in path1] == [s.log_prob for s in path2]


def test_history_window_rejects_invalid_value():
    with pytest.raises(ValueError):
        ViterbiMapMatcher(history_window=0)


def test_intersection_ambiguous_then_resolved_by_continuity():
    """
    Synthetic fixture: a vehicle approaches an intersection where two roads
    (through-road and cross-road) are genuinely EMISSION-ambiguous for one
    tick (near-identical distance/heading agreement), then the vehicle's
    continued straight-line motion resolves which road it actually took.
    A correct Viterbi path must settle on the through-road once the
    accumulated evidence makes it unambiguous, even though a single-tick
    greedy decision at the ambiguous step alone could go either way.
    """
    def candidate_fn(obs: Observation):
        through = _seg_point("through_rd", obs.lat, obs.lon, 0.0, dist_m=3.0)
        # Cross-road candidate is also very close right at the intersection,
        # but only near the intersection point (obs index ~ tick 3).
        cross = _seg_point("cross_rd", obs.lat, obs.lon, math.pi / 2, dist_m=3.0)
        return [through, cross]

    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(8)
    ]
    path = run_offline_match(obs_stream, candidate_fn, history_window=50)
    # The accumulated path, scored over the WHOLE run of consistent forward
    # motion, must prefer the through-road at the overwhelming majority of
    # steps (transition continuity keeps rewarding staying on 'through_rd').
    through_count = sum(1 for s in path if s.had_candidates and s.candidate.seg_id == "through_rd")
    assert through_count >= 6


def test_parallel_roads_disambiguated_by_lateral_offset():
    """
    Synthetic fixture: two parallel roads running the same direction, one
    genuinely closer to the observed track throughout. Emission distance
    alone should consistently favor the closer one at every tick (this is a
    sanity check that the matcher does not spuriously flip between
    emission-favored candidates when there is no ambiguity at all).
    """
    def candidate_fn(obs: Observation):
        near_road = _seg_point("near_lane", obs.lat, obs.lon, 0.0, dist_m=2.0)
        far_road = _seg_point("far_lane", obs.lat, obs.lon, 0.0, dist_m=25.0)
        return [near_road, far_road]

    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(6)
    ]
    path = run_offline_match(obs_stream, candidate_fn, history_window=50)
    assert all(s.candidate.seg_id == "near_lane" for s in path if s.had_candidates)


def test_noisy_gnss_observations_still_track_correct_road():
    """
    Synthetic fixture: observations carry small, deterministic (seeded)
    positional noise around the true straight-line track, simulating real
    GNSS jitter. The correct road segment (which itself absorbs the same
    small offset each tick, as a real nearest-segment projection would) must
    still dominate the best path despite the added noise.
    """
    rng = __import__("random").Random(99)

    def candidate_fn(obs: Observation):
        jitter = rng.uniform(-0.5, 0.5)
        correct = _seg_point("noisy_main_rd", obs.lat, obs.lon, 0.0, dist_m=3.0 + abs(jitter))
        decoy = _seg_point("noisy_decoy_rd", obs.lat + 0.005, obs.lon, math.pi / 2, dist_m=120.0)
        return [correct, decoy]

    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(10)
    ]
    path = run_offline_match(obs_stream, candidate_fn, history_window=50)
    correct_count = sum(1 for s in path if s.had_candidates and s.candidate.seg_id == "noisy_main_rd")
    assert correct_count == 10  # decoy is far enough that jitter never changes the outcome


def test_temporary_gnss_gap_mid_run_then_recovers_on_correct_road():
    """
    Combines a mid-run candidate gap (as if GNSS/road-graph coverage
    dropped out) with recovery afterward on the correct road, verifying both
    behaviors together rather than in isolation (as the two separate
    gap tests above do individually).
    """
    calls = {"n": 0}

    def candidate_fn(obs: Observation):
        calls["n"] += 1
        if 4 <= calls["n"] <= 6:
            return []  # a 3-tick coverage gap in the middle of the run
        return [_seg_point("through_after_gap", obs.lat, obs.lon, 0.0, dist_m=2.0),
                _seg_point("decoy_after_gap", obs.lat + 0.01, obs.lon, math.pi / 2, dist_m=100.0)]

    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(10)
    ]
    path = run_offline_match(obs_stream, candidate_fn, history_window=50)
    assert len(path) == 10
    for i in range(3, 6):  # 0-indexed calls 4..6 -> path indices 3..5
        assert path[i].had_candidates is False
    for i in list(range(3)) + list(range(6, 10)):
        assert path[i].had_candidates is True
        assert path[i].candidate.seg_id == "through_after_gap"


def test_wrong_early_candidate_corrected_by_later_global_evidence():
    """
    The critical Viterbi-vs-greedy fixture: at the VERY FIRST tick, a
    decoy segment has a slightly BETTER emission score than the true
    segment (so a purely local/greedy per-tick argmax would lock onto the
    wrong segment immediately and never reconsider it). Every subsequent
    tick then gives overwhelming, unambiguous evidence for the true
    segment. Because this module accumulates path score across the whole
    window and reconstructs via backpointers, the globally best path must
    select the TRUE segment at the early tick too, correcting the initial
    locally-best-looking choice - not just from the point evidence arrives.
    """
    def candidate_fn(obs: Observation):
        idx = round((obs.lat - 10.0) / 0.0001)
        true_seg = _seg_point("true_rd", obs.lat, obs.lon, 0.0,
                               dist_m=(1.0 if idx == 0 else 1.0))
        if idx == 0:
            # At tick 0 only: decoy has a (very slightly) better emission score.
            decoy = _seg_point("decoy_rd", obs.lat, obs.lon, 0.0, dist_m=0.5)
        else:
            # From tick 1 onward: decoy becomes an obviously terrible, distant match.
            decoy = _seg_point("decoy_rd", obs.lat + 0.02, obs.lon, math.pi / 2, dist_m=300.0)
        return [true_seg, decoy]

    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(6)
    ]
    path = run_offline_match(obs_stream, candidate_fn, history_window=50)
    assert len(path) == 6
    # Global best path selects true_rd even at tick 0, where local emission
    # alone would have favored decoy_rd.
    assert path[0].candidate.seg_id == "true_rd"
    for s in path:
        assert s.candidate.seg_id == "true_rd"


def test_min_candidates_threshold_forces_gap():
    def candidate_fn(obs: Observation):
        return [_seg_point("lonely", obs.lat, obs.lon, 0.0, dist_m=3.0)]  # only 1 candidate

    matcher = ViterbiMapMatcher(history_window=10, min_candidates=2)  # require >=2 candidates
    obs = Observation(lat=10.0, lon=20.0, heading_rad=0.0, speed_mps=5.0, dt_s=1.0)
    matcher.step(obs, candidate_fn(obs))
    path = matcher.best_path()
    assert len(path) == 1
    assert path[0].had_candidates is False
