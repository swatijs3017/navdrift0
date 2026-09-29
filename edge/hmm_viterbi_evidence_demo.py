"""
edge/hmm_viterbi_evidence_demo.py — reproducible offline Viterbi evidence generator
ISRO SIH 2026 PS #26168

Runs edge/hmm_map_matcher.py against a deterministic, labeled SYNTHETIC
road/candidate fixture and prints/exports the full pipeline trace:

    GNSS observations -> road candidates -> emission scores -> transition
    scores -> accumulated sequence score -> Viterbi backtracking -> final
    matched path

The fixture used is the "wrong early candidate corrected by later global
evidence" case: at tick 0 a decoy segment has a slightly BETTER emission
score than the true segment, so a purely greedy per-tick matcher would lock
onto the wrong segment immediately. Every later tick then gives
unambiguous evidence for the true segment. Because this module accumulates
path score across the whole window and backtracks, the reconstructed path
picks the TRUE segment even at tick 0 — this is demonstrated explicitly
below, tick by tick, distinct from what a local/greedy decision would have
produced at that same tick.

This makes NO real-world map-matching accuracy claim. The road/candidate
data is hand-constructed synthetic data, not real GNSS or OSM data. This is
also explicitly NOT the live greedy matcher (LiveHMM) in
frontend/mobile.html, which is not called, modified, or exercised by this
script in any way.

Usage:
    python -m edge.hmm_viterbi_evidence_demo --out_json hmm_viterbi_evidence.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.hmm_map_matcher import (
    Observation, RoadCandidate, ViterbiMapMatcher,
    emission_log_prob, transition_log_prob,
)


def _seg(seg_id, lat, lon, bearing_rad, dist_m):
    return RoadCandidate(seg_id=seg_id, lat=lat, lon=lon, bearing_rad=bearing_rad, dist_m=dist_m)


def candidate_fn(obs: Observation, tick: int):
    """SYNTHETIC, hand-constructed candidate generator. At tick 0 the decoy
    has a better (smaller) emission distance than the true segment; from
    tick 1 onward the decoy is an obviously bad, distant match."""
    true_seg = _seg("true_rd", obs.lat, obs.lon, 0.0, dist_m=1.0)
    if tick == 0:
        decoy = _seg("decoy_rd", obs.lat, obs.lon, 0.0, dist_m=0.5)
    else:
        decoy = _seg("decoy_rd", obs.lat + 0.02, obs.lon, math.pi / 2, dist_m=300.0)
    return [true_seg, decoy]


def run_demo():
    obs_stream = [
        Observation(lat=10.0 + 0.0001 * i, lon=20.0, heading_rad=0.0, speed_mps=11.1, dt_s=1.0)
        for i in range(6)
    ]

    matcher = ViterbiMapMatcher(history_window=50)
    trace = []
    for tick, obs in enumerate(obs_stream):
        cands = candidate_fn(obs, tick)
        # Record per-candidate emission/transition scores BEFORE calling
        # step(), so the trace shows the same numbers the matcher itself
        # computes internally.
        prev_best = None
        if trace and trace[-1]["step_had_candidates"]:
            prev_best_id = trace[-1]["local_greedy_pick"]
            prev_best = next(c for c in trace[-1]["candidates_raw"] if c.seg_id == prev_best_id)
        cand_scores = []
        for c in cands:
            emit = emission_log_prob(c, obs)
            trans = transition_log_prob(prev_best, c, obs) if prev_best is not None else None
            cand_scores.append({
                "seg_id": c.seg_id, "dist_m": c.dist_m, "bearing_rad": c.bearing_rad,
                "emission_log_prob": round(emit, 4),
                "transition_log_prob": (round(trans, 4) if trans is not None else None),
            })
        local_greedy_pick = max(cand_scores, key=lambda cs: cs["emission_log_prob"])["seg_id"]
        matcher.step(obs, cands)
        trace.append({
            "tick": tick,
            "observation": {"lat": obs.lat, "lon": obs.lon, "speed_mps": obs.speed_mps},
            "candidates_raw": cands,
            "candidate_scores": cand_scores,
            "local_greedy_pick": local_greedy_pick,   # what a per-tick-only argmax would pick
            "step_had_candidates": True,
        })

    best_path = matcher.best_path()
    final_trace = []
    for tick, (t, state) in enumerate(zip(trace, best_path)):
        final_trace.append({
            "tick": tick,
            "local_greedy_pick": t["local_greedy_pick"],
            "viterbi_best_path_pick": state.candidate.seg_id if state.had_candidates else None,
            "accumulated_log_prob": round(state.log_prob, 4) if state.had_candidates else None,
            "corrected": (t["local_greedy_pick"] != (state.candidate.seg_id if state.had_candidates else None)),
        })

    return trace, final_trace


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NAVDRIFT-0 offline HMM/Viterbi evidence generator")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    trace, final_trace = run_demo()

    print(f"\n{'='*70}\n  NAVDRIFT-0 Offline Viterbi Map-Matching Evidence\n"
          f"  SYNTHETIC road/candidate fixture. Not real GNSS or OSM data.\n"
          f"  Not the live greedy LiveHMM matcher in frontend/mobile.html.\n{'='*70}")
    for row in final_trace:
        mark = "  <-- corrected by global evidence" if row["corrected"] else ""
        print(f"  tick {row['tick']}: local greedy pick = {row['local_greedy_pick']:10s}  "
              f"Viterbi best-path pick = {row['viterbi_best_path_pick']:10s}{mark}")

    n_corrected = sum(1 for r in final_trace if r["corrected"])
    print(f"\n  {n_corrected} tick(s) where the accumulated-path Viterbi reconstruction "
          f"differs from what a purely local/greedy per-tick decision would have picked.")

    report = {
        "validation_type": "OFFLINE_ONLY",
        "hardware_validation": "NOT_PERFORMED",
        "fixture_kind": "SYNTHETIC_HAND_CONSTRUCTED",
        "note": "No real-world map-matching accuracy claim. This demonstrates algorithmic "
                "correctness (Viterbi path reconstruction differs from and improves on a local "
                "greedy decision) against a hand-constructed fixture with one unambiguous "
                "correct answer, not accuracy against real GNSS/road data. Distinct from, and "
                "does not call or modify, frontend/mobile.html's live LiveHMM matcher.",
        "per_tick_trace": trace,
        "final_comparison": final_trace,
        "n_ticks_corrected_by_global_evidence": n_corrected,
    }

    if args.out_json:
        def _default(o):
            if isinstance(o, RoadCandidate):
                return {"seg_id": o.seg_id, "lat": o.lat, "lon": o.lon,
                        "bearing_rad": o.bearing_rad, "dist_m": o.dist_m}
            raise TypeError
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2, default=_default)
        print(f"\nWritten to {args.out_json}")
