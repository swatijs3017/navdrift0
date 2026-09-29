"""
edge/hmm_map_matcher.py — NAVDRIFT-0 Offline HMM/Viterbi Map-Matching Module
ISRO SIH 2026 PS #26168

A NEW, SEPARATE, OFFLINE Python module implementing a genuine bounded-history
Hidden Markov Model map matcher with full Viterbi best-path reconstruction
(candidate generation -> emission scoring -> transition scoring -> accumulated
path score -> backpointers -> best-path traceback).

Honest relationship to the live mobile matcher: frontend/mobile.html's
LiveHMM (see its own header comment) is an online, single-state, greedy
per-tick selector — it re-scores a fresh candidate set every tick using
emission + transition log-probabilities but keeps only the local argmax; it
does not accumulate a path score across time and does not backtrack. This
module is a genuinely different algorithm: it accumulates per-state path
scores across a bounded history window and reconstructs the single
maximum-likelihood path over that whole window via backpointers, the way a
textbook Viterbi decoder does. It does NOT replace, call, or share any
runtime state with LiveHMM or RoadGraph in frontend/mobile.html. It is not
wired into the live phone navigation pipeline in any way — it is an offline,
batch, Python-only analysis tool, run against a road-segment graph and a
sequence of observations supplied by the caller.

WHAT THIS MODULE DOES NOT CLAIM:
  - No claim of real-world map-matching accuracy is made anywhere in this
    module or its tests unless a real ground-truth benchmark dataset is
    supplied and scored — the unit tests here use small, explicitly labeled
    SYNTHETIC road/candidate fixtures to verify the ALGORITHM's correctness
    (does it pick the path an idealized human would pick, given synthetic
    candidates constructed to have one unambiguous best path), not to
    benchmark accuracy against real GNSS/road data.
  - Emission/transition scoring follows the same log-Gaussian family the live
    LiveHMM module uses (documented in its own header), reused here because
    it is a reasonable, already-reviewed choice for this domain — but the
    sigma constants below are independent, tunable parameters of THIS
    offline module, not copies of live runtime state.

Usage sketch:
    matcher = ViterbiMapMatcher(history_window=50)
    for obs in observation_stream:
        matcher.step(obs, candidate_fn=my_candidate_generator)
    best_path = matcher.best_path()   # list of MatchedState, oldest -> newest
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence as TSequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoadCandidate:
    """One candidate road-segment match for a single observation.

    `seg_id` identifies the road segment (any hashable value — an int index,
    an OSM way id, a string, etc). `lat`/`lon` are the projected point on
    that segment nearest the observation; `bearing_rad` is the segment's
    heading at that point; `dist_m` is the perpendicular distance from the
    raw observation to the segment.
    """
    seg_id: object
    lat: float
    lon: float
    bearing_rad: float
    dist_m: float


@dataclass(frozen=True)
class Observation:
    """One timestep's raw GNSS observation plus the motion context needed for
    transition scoring (real elapsed time and a speed estimate — the caller
    supplies these; this module fabricates neither)."""
    lat: float
    lon: float
    heading_rad: Optional[float]   # None if genuinely unknown — never guessed
    speed_mps: Optional[float]     # None if genuinely unknown — never guessed
    dt_s: float                    # real elapsed time since the previous observation


@dataclass
class MatchedState:
    """One reconstructed step of the best path after Viterbi traceback."""
    candidate: RoadCandidate
    log_prob: float                # accumulated path log-probability up to and including this step
    had_candidates: bool           # False if this step had no usable candidates (see below)


# ---------------------------------------------------------------------------
# Internal trellis node
# ---------------------------------------------------------------------------

@dataclass
class _TrellisNode:
    candidate: RoadCandidate
    score: float                   # accumulated best log-probability reaching this node
    backptr: Optional[int]         # index into the PREVIOUS step's candidate list, or None


# ---------------------------------------------------------------------------
# Scoring (same log-Gaussian family LiveHMM uses; independent constants)
# ---------------------------------------------------------------------------

DEFAULT_EMIT_DIST_SIGMA_M = 25.0
DEFAULT_EMIT_HEADING_SIGMA_RAD = 35.0 * math.pi / 180.0
DEFAULT_TRANS_CONTINUITY_SIGMA_M = 20.0
DEFAULT_TRANS_STAY_BONUS = 0.6
DEFAULT_MAX_SPEED_MPS = 40.0


def _wrapped_heading_diff(a: float, b: float) -> float:
    """Undirected segment heading agreement, folded into [0, pi/2] — a
    two-way road's bearing and its reverse are equally 'aligned'."""
    d = abs(((a - b + math.pi) % (2 * math.pi)) - math.pi)
    return min(d, math.pi - d)


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def emission_log_prob(cand: RoadCandidate, obs: Observation,
                       dist_sigma_m: float = DEFAULT_EMIT_DIST_SIGMA_M,
                       heading_sigma_rad: float = DEFAULT_EMIT_HEADING_SIGMA_RAD) -> float:
    """How well `cand` explains `obs`. Distance term always applies; the
    heading term is only included when obs.heading_rad is actually known —
    never substituted with an assumed value."""
    d_term = -(cand.dist_m ** 2) / (2 * dist_sigma_m ** 2)
    if obs.heading_rad is None:
        return d_term
    dh = _wrapped_heading_diff(obs.heading_rad, cand.bearing_rad)
    h_term = -(dh ** 2) / (2 * heading_sigma_rad ** 2)
    return d_term + h_term


def transition_log_prob(prev: RoadCandidate, cand: RoadCandidate, obs: Observation,
                         continuity_sigma_m: float = DEFAULT_TRANS_CONTINUITY_SIGMA_M,
                         stay_bonus: float = DEFAULT_TRANS_STAY_BONUS,
                         max_speed_mps: float = DEFAULT_MAX_SPEED_MPS) -> float:
    """How well moving from `prev` to `cand` is explained by the real elapsed
    time and (if known) real speed. If speed is unknown, only the
    same-segment continuity bonus applies — no speed is assumed."""
    same_seg = cand.seg_id == prev.seg_id
    actual_step_m = _haversine_m(prev.lat, prev.lon, cand.lat, cand.lon)
    if obs.speed_mps is not None:
        expected_step_m = max(0.0, min(obs.speed_mps, max_speed_mps)) * max(0.0, obs.dt_s)
        discrepancy = abs(actual_step_m - expected_step_m)
        cont_term = -(discrepancy ** 2) / (2 * continuity_sigma_m ** 2)
    else:
        cont_term = 0.0
    return cont_term + (stay_bonus if same_seg else 0.0)


# ---------------------------------------------------------------------------
# Viterbi map matcher
# ---------------------------------------------------------------------------

CandidateFn = Callable[[Observation], List[RoadCandidate]]


class ViterbiMapMatcher:
    """
    Bounded-history Viterbi map matcher.

    `history_window` bounds memory/computation: only the last N steps'
    trellis columns are retained. This is a genuine engineering bound (real
    driving map-matching is an online-ish task; an unbounded trellis would
    grow forever), documented explicitly rather than silently limiting
    correctness — see `best_path()`'s docstring for exactly what "best path"
    means when the window has scrolled past older observations.
    """

    def __init__(self, history_window: int = 50,
                 emit_dist_sigma_m: float = DEFAULT_EMIT_DIST_SIGMA_M,
                 emit_heading_sigma_rad: float = DEFAULT_EMIT_HEADING_SIGMA_RAD,
                 trans_continuity_sigma_m: float = DEFAULT_TRANS_CONTINUITY_SIGMA_M,
                 trans_stay_bonus: float = DEFAULT_TRANS_STAY_BONUS,
                 max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
                 min_candidates: int = 1):
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        self.history_window = history_window
        self.emit_dist_sigma_m = emit_dist_sigma_m
        self.emit_heading_sigma_rad = emit_heading_sigma_rad
        self.trans_continuity_sigma_m = trans_continuity_sigma_m
        self.trans_stay_bonus = trans_stay_bonus
        self.max_speed_mps = max_speed_mps
        self.min_candidates = min_candidates

        # Each element: (observation, [ _TrellisNode, ... ] or None if no usable candidates)
        self._trellis: List[tuple] = []

    def reset(self) -> None:
        self._trellis.clear()

    def step(self, obs: Observation, candidates: List[RoadCandidate]) -> None:
        """
        Advance the trellis by one observation, given an already-generated
        candidate list (candidate generation is the caller's responsibility
        — e.g. a road-graph nearest-K lookup; this module is graph-agnostic
        and never invents candidates itself).

        Gracefully handles too few/no candidates: the step is recorded as a
        gap (no trellis nodes), and the NEXT step with usable candidates
        starts a fresh sub-path (emission-only, no transition term) rather
        than crashing or silently fabricating a match.
        """
        usable = candidates if len(candidates) >= self.min_candidates else []

        if not usable:
            self._trellis.append((obs, None))
        else:
            prev_col = None
            for prev_obs, prev_nodes in reversed(self._trellis):
                if prev_nodes is not None:
                    prev_col = prev_nodes
                break  # only look at the immediately preceding column; a gap breaks continuity
            nodes: List[_TrellisNode] = []
            for cand in usable:
                emit = emission_log_prob(cand, obs, self.emit_dist_sigma_m, self.emit_heading_sigma_rad)
                if prev_col is None:
                    nodes.append(_TrellisNode(candidate=cand, score=emit, backptr=None))
                else:
                    best_score = -math.inf
                    best_bp = None
                    for bp_idx, prev_node in enumerate(prev_col):
                        trans = transition_log_prob(prev_node.candidate, cand, obs,
                                                     self.trans_continuity_sigma_m,
                                                     self.trans_stay_bonus, self.max_speed_mps)
                        cand_score = prev_node.score + trans + emit
                        if cand_score > best_score:
                            best_score = cand_score
                            best_bp = bp_idx
                    nodes.append(_TrellisNode(candidate=cand, score=best_score, backptr=best_bp))
            self._trellis.append((obs, nodes))

        if len(self._trellis) > self.history_window:
            self._trellis.pop(0)

    def best_path(self) -> List[MatchedState]:
        """
        Reconstruct the maximum-likelihood path over the CURRENT retained
        window via backpointer traceback, oldest -> newest.

        A gap (a step with no usable candidates, or the very first step
        after a gap) breaks continuity: traceback only follows backpointers
        within one unbroken run of usable-candidate columns. Steps within a
        gap are represented with had_candidates=False and no candidate
        selected — never a fabricated match.
        """
        result: List[MatchedState] = []
        i = len(self._trellis) - 1
        while i >= 0:
            obs, nodes = self._trellis[i]
            if nodes is None:
                # Traceback walks newest -> oldest, so a gap entry must be PREPENDED to keep
                # the final result oldest -> newest, matching the run branch below.
                result = [MatchedState(candidate=None, log_prob=-math.inf, had_candidates=False)] + result
                i -= 1
                continue

            # Find the run of consecutive usable-candidate columns ending at i.
            run_end = i
            run_start = i
            while run_start - 1 >= 0 and self._trellis[run_start - 1][1] is not None:
                run_start -= 1

            # Best terminal node of this run.
            best_idx = max(range(len(nodes)), key=lambda k: nodes[k].score)
            run_nodes = [self._trellis[j][1] for j in range(run_start, run_end + 1)]
            path_idx = best_idx
            run_result = []
            for col_idx in range(len(run_nodes) - 1, -1, -1):
                node = run_nodes[col_idx][path_idx]
                run_result.append(MatchedState(candidate=node.candidate, log_prob=node.score, had_candidates=True))
                path_idx = node.backptr if node.backptr is not None else 0
            run_result.reverse()
            result = run_result + result
            i = run_start - 1
        return result

    def __len__(self) -> int:
        return len(self._trellis)


# ---------------------------------------------------------------------------
# Convenience: run the matcher over a full observation stream in one call
# ---------------------------------------------------------------------------

def run_offline_match(observations: TSequence[Observation],
                       candidate_fn: CandidateFn,
                       history_window: int = 50,
                       **scoring_kwargs) -> List[MatchedState]:
    """
    GNSS observations -> candidate road segments -> emission scores ->
    transition scores -> Viterbi path -> reconstructed matched trajectory,
    in one call. `candidate_fn` is caller-supplied (e.g. a road-graph
    nearest-K lookup over real or synthetic road data) — this module never
    generates candidates itself.
    """
    matcher = ViterbiMapMatcher(history_window=history_window, **scoring_kwargs)
    for obs in observations:
        cands = candidate_fn(obs)
        matcher.step(obs, cands)
    return matcher.best_path()
