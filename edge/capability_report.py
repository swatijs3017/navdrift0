"""
edge/capability_report.py — NAVDRIFT-0 reproducible capability report
ISRO SIH 2026 PS #26168

Generates a machine-readable capability report from the repository's actual
state: the requirement evidence records (edge/requirement_evidence.py) plus
a set of direct, reproducible repository checks (which files exist, which
tests currently pass, whether the ONNX models on disk are present). Nothing
here is asserted from memory — every field is either read from
requirement_evidence.py or computed fresh by inspecting the repository or
running the actual test suite.

Status categories (exactly these, no other words — no "excellent",
"complete", "production-ready", etc.):
    IMPLEMENTED
    VALIDATED
    SOFTWARE_ONLY_VALIDATED
    OFFLINE_ONLY
    DATA_BLOCKED
    EXPERIMENTAL
    NOT_IMPLEMENTED

Usage:
    python -m edge.capability_report --out_json capability_report.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edge.requirement_evidence import REQUIREMENT_EVIDENCE, validate_evidence_records


# Live/production components explicitly out of scope for re-validation here
# (frozen per the engineering build's safety rules) — listed for completeness
# in the report, status taken as-is from requirement_evidence.py, never
# re-tested by this script.
FROZEN_LIVE_FILES = [
    "frontend/mobile.html",
    "frontend/desktop.html",
]

# Pre-existing ONNX models this report checks for presence only (existence,
# not accuracy) — a genuinely reproducible, cheap check.
KNOWN_ONNX_MODELS = [
    "models/driftformer_fp32.onnx",
    "models/adaptive_ekf_fp32.onnx",
    "models/tunnel_det_fp32.onnx",
]

# New offline modules built in this engineering phase.
NEW_OFFLINE_MODULES = [
    "edge/external_imu.py",
    "edge/vehicle_calibration.py",
    "edge/hmm_map_matcher.py",
    "edge/replay_200hz_demo.py",
    "edge/imu_replay_example.py",
    "edge/requirement_evidence.py",
    "edge/capability_report.py",
    "edge/model_inference_bench.py",
    "edge/calibration_evidence_demo.py",
    "edge/hmm_viterbi_evidence_demo.py",
]


def _file_exists(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


def _run_pytest(test_paths: List[str]) -> Dict:
    """Actually runs pytest against the given test files and reports the
    real pass/fail counts — not assumed from a prior run."""
    existing = [t for t in test_paths if _file_exists(t)]
    if not existing:
        return {"ran": False, "reason": "no test files found", "passed": 0, "failed": 0}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *existing, "-q", "--no-header"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout + proc.stderr
    passed = failed = 0
    for line in out.splitlines():
        line = line.strip()
        if line.endswith("passed") or " passed" in line:
            # pytest summary line, e.g. "64 passed in 2.94s" or "1 failed, 63 passed in 3s"
            import re
            m_pass = re.search(r"(\d+) passed", line)
            m_fail = re.search(r"(\d+) failed", line)
            if m_pass:
                passed = int(m_pass.group(1))
            if m_fail:
                failed = int(m_fail.group(1))
    return {"ran": True, "returncode": proc.returncode, "passed": passed, "failed": failed}


def generate_report(run_tests: bool = True) -> Dict:
    evidence_problems = validate_evidence_records()

    onnx_status = {p: _file_exists(p) for p in KNOWN_ONNX_MODELS}
    new_module_status = {p: _file_exists(p) for p in NEW_OFFLINE_MODULES}
    frozen_status = {p: _file_exists(p) for p in FROZEN_LIVE_FILES}

    requirement_summary = []
    status_counts: Dict[str, int] = {}
    for rec in REQUIREMENT_EVIDENCE:
        status_counts[rec.validation_type] = status_counts.get(rec.validation_type, 0) + 1
        requirement_summary.append({
            "requirement": rec.requirement,
            "status": rec.validation_type,
            "files": rec.files,
            "tests": rec.tests,
            "limitation": rec.limitation,
        })

    test_run_result = None
    if run_tests:
        all_test_files = sorted({t for rec in REQUIREMENT_EVIDENCE for t in rec.tests})
        test_run_result = _run_pytest(all_test_files)

    report = {
        "report_type": "NAVDRIFT-0 capability report",
        "generated_reproducibly_from": "edge/capability_report.py (reads edge/requirement_evidence.py "
                                        "and inspects the repository directly; nothing hardcoded from memory)",
        "status_categories_used": sorted(status_counts.keys()),
        "requirement_status_counts": status_counts,
        "requirements": requirement_summary,
        "requirement_evidence_consistency_problems": evidence_problems,
        "known_onnx_models_present": onnx_status,
        "new_offline_modules_present": new_module_status,
        "frozen_live_files_present_unmodified_by_this_report": frozen_status,
        "evidence_test_run": test_run_result,
        "notes": [
            "Frozen live files are listed for completeness only; this report does not "
            "re-validate or modify them.",
            "DATA_BLOCKED entries cannot be upgraded to VALIDATED/IMPLEMENTED without a "
            "real IO-VNBD (or equivalent) dataset, which does not exist in this repository.",
            "SOFTWARE_ONLY_VALIDATED means tested via software unit tests/benchmarks only; "
            "no physical hardware was used to produce any number in this report.",
        ],
    }
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NAVDRIFT-0 reproducible capability report")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--no_tests", action="store_true", help="Skip actually running the evidence test suite")
    args = parser.parse_args()

    report = generate_report(run_tests=not args.no_tests)

    print(f"\n{'='*70}\n  NAVDRIFT-0 Capability Report\n{'='*70}")
    print(f"\nStatus counts: {report['requirement_status_counts']}")
    print("\nPer-requirement status:")
    for r in report["requirements"]:
        print(f"  [{r['status']:24s}] {r['requirement']}")
    if report["requirement_evidence_consistency_problems"]:
        print("\nCONSISTENCY PROBLEMS:")
        for p in report["requirement_evidence_consistency_problems"]:
            print(f"  - {p}")
    if report["evidence_test_run"]:
        t = report["evidence_test_run"]
        print(f"\nEvidence test run: {t}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWritten to {args.out_json}")
