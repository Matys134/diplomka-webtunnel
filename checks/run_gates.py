#!/usr/bin/env python3
"""Run every build gate that can run against the given dataset, and report a summary.

Against the v1 dataset this is expected to FAIL on G2, G3 and G5 -- that is the harness
working, and the printed tripwire table is a thesis figure. G1, G4 and G6 need v2 artefacts
(CaptureManifest sidecars, FlowRecord objects) and will report SKIP until the collector is
rewritten.

Usage:
    python3 project/checks/run_gates.py \
        --dataset   project/data/processed/tabular_dataset.npz \
        --sequences project/data/processed/sequence_dataset.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)

import budget_parity            # noqa: E402
import null_controls            # noqa: E402
import split_integrity          # noqa: E402
import tripwire                 # noqa: E402

CLASS_NAMES = [
    "webtunnel", "direct_web_browsing", "websocket_ticker",
    "websocket_chat", "video_streaming", "web_assets", "quic_http3",
]

BAR = "=" * 78


def header(gate: str, title: str):
    print(f"\n{BAR}\n  {gate}  {title}\n{BAR}")


def main():
    ap = argparse.ArgumentParser(description="WebTunnel v2 build gates")
    ap.add_argument("--dataset", required=True, help="tabular_dataset.npz")
    ap.add_argument("--sequences", help="sequence_dataset.npz (optional)")
    ap.add_argument("--json", help="write a machine-readable summary here")
    ap.add_argument("--allow-fail", action="store_true",
                    help="exit 0 even when gates fail (use on the v1 corpus)")
    a = ap.parse_args()

    d = np.load(a.dataset, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]] if "feature_names" in d else \
            [f"f{i}" for i in range(d["X_train"].shape[1])]

    X_all = np.concatenate([d["X_train"], d["X_val"], d["X_test"]])
    y_all = np.concatenate([d["y_train"], d["y_val"], d["y_test"]])
    ym_all = np.concatenate([d["y_train_mul"], d["y_val_mul"], d["y_test_mul"]]) \
        if "y_train_mul" in d else None

    print(BAR)
    print("  WebTunnel v2 build gates")
    print(f"  dataset : {a.dataset}")
    print(f"  samples : {len(y_all)}  ({int(y_all.sum())} positive, "
          f"{int((y_all == 0).sum())} negative)")
    print(f"  features: {len(names)}")
    print(BAR)

    results = {}

    import corpus_gates

    flow_records_path = os.path.join(os.path.dirname(a.dataset), "flow_records.jsonl")
    manifest_path = "data/manifest.jsonl"
    flows = []
    if os.path.exists(flow_records_path):
        from common.contracts import FlowRecord
        with open(flow_records_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    flows.append(FlowRecord(**json.loads(line)))

    manifests = {}
    if os.path.exists(manifest_path):
        from common.contracts import CaptureManifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = CaptureManifest.from_json(line)
                    manifests[m.capture_id] = m

    header("G1", "stack parity (ClientHello identical across classes)")
    if flows:
        results["G1"] = corpus_gates.stack_parity(flows)
    else:
        print("  SKIP -- needs v2 FlowRecord objects with clienthello_len / ja4.")
        results["G1"] = {"status": "skipped", "reason": "needs v2 flow records"}

    header("G2", "leakage tripwire")
    results["G2"] = tripwire.run(d["X_train"], d["y_train"], d["X_test"], d["y_test"], names)

    header("G3", "null controls")
    results["G3"] = null_controls.run(X_all, y_all, ym_all)

    header("G4", "session-budget parity")
    if ym_all is None:
        print("  SKIP -- no multiclass labels in this dataset.")
        results["G4"] = {"status": "skipped"}
    else:
        results["G4"] = budget_parity.run(X_all, ym_all, names, CLASS_NAMES)

    header("G5", "split integrity")
    results["G5"] = split_integrity.run_v1_diagnostic(d)

    header("G6", "provenance")
    if flows and manifests:
        results["G6"] = corpus_gates.provenance(flows, manifests)
    else:
        print("  SKIP -- needs CaptureManifest sidecars written by the v2 collector.")
        results["G6"] = {"status": "skipped", "reason": "needs v2 manifests"}

    print(f"\n{BAR}\n  SUMMARY\n{BAR}")
    failed = []
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        r = results.get(g, {})
        if r.get("status") == "skipped":
            verdict = "SKIP"
        elif r.get("passed"):
            verdict = "PASS"
        else:
            verdict = "FAIL"
            failed.append(g)
        print(f"  {g}  {verdict}")

    if failed:
        print(f"\n  {len(failed)} gate(s) failed: {', '.join(failed)}")
        print("  On the v1 corpus this is the expected and correct outcome.")
        print("  See docs/01-audit-findings.md and docs/02-rebuild-plan.md section 5.")
    else:
        print("\n  All runnable gates passed.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  wrote {a.json}")

    sys.exit(0 if (not failed or a.allow_fail) else 1)


if __name__ == "__main__":
    main()
