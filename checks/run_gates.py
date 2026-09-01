#!/usr/bin/env python3
"""Run all six build gates. A failing gate BLOCKS the build (exit 1).

v2.1: every gate is falsifiable. See docs/04-v2-audit.md section 3 (V-01) for what each of these
used to assert and why four of them could not fail.

Usage:
    python3 checks/run_gates.py --dataset data/processed/tabular_dataset.npz
    python3 checks/run_gates.py --dataset ... --json data/processed/gates.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
for p in (HERE, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "2_data_pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

import budget_parity            # noqa: E402
import corpus_gates             # noqa: E402
import null_controls            # noqa: E402
import split_integrity          # noqa: E402
import tripwire                 # noqa: E402
from common.contracts import CaptureManifest, FlowRecord  # noqa: E402

CLASS_NAMES = ["webtunnel", "direct_web_browsing", "websocket_ticker",
               "websocket_chat", "video_streaming", "web_assets", "quic_http3"]
BAR = "=" * 78
GATE_TITLES = {
    "G1": "stack parity -- one ClientHello / JA4 / ALPN across every class",
    "G2": "leakage tripwire -- no unregistered single feature above AUC 0.90",
    "G3": "null controls -- label shuffle and same-generator both at chance",
    "G4": "session-budget parity -- paired on budget_id",
    "G5": "split integrity -- conn_id AND socket disjoint, groups aligned",
    "G6": "provenance -- authoritative ground truth, 5-tuple verified, drops logged",
}


def header(gate):
    print(f"\n{BAR}\n  {gate}  {GATE_TITLES[gate]}\n{BAR}")


def load_flows(path):
    flows = []
    if not os.path.exists(path):
        return flows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                known = FlowRecord.__dataclass_fields__.keys()  # type: ignore[attr-defined]
                flows.append(FlowRecord(**{k: v for k, v in d.items() if k in known}))
    return flows


def main():
    ap = argparse.ArgumentParser(description="WebTunnel v2.1 build gates")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequences")
    ap.add_argument("--json")
    ap.add_argument("--allow-fail", action="store_true",
                    help="exit 0 even when gates fail (diagnostic runs on a legacy corpus only)")
    ap.add_argument("--seeds", type=int, default=10, help="seeds for the G3 null controls")
    a = ap.parse_args()

    d = np.load(a.dataset, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    X_all = np.concatenate([d["X_train"], d["X_val"], d["X_test"]])
    y_all = np.concatenate([d["y_train"], d["y_val"], d["y_test"]])
    ym_all = np.concatenate([d["y_train_mul"], d["y_val_mul"], d["y_test_mul"]])
    labels_all = d["labels_all"] if "labels_all" in d.files else None
    socks_all = d["socket_ids_all"] if "socket_ids_all" in d.files else None
    budgets_all = d["budget_ids_all"] if "budget_ids_all" in d.files else None
    order_all = d["t_start_all"] if "t_start_all" in d.files else None

    data_dir = os.path.dirname(a.dataset)
    flows = load_flows(os.path.join(data_dir, "flow_records.jsonl"))

    manifests = {}
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw_pcap")
    for side in glob.glob(os.path.join(raw_dir, "*.manifest.json")):
        try:
            with open(side, encoding="utf-8") as f:
                m = CaptureManifest.from_json(f.read())
            manifests[m.capture_id] = m
        except Exception:
            continue
    n_on_disk = len(glob.glob(os.path.join(raw_dir, "*.pcap")))
    attrition_path = os.path.join(data_dir, "attrition_build.json")
    attrition = json.load(open(attrition_path)) if os.path.exists(attrition_path) else None

    print(BAR)
    print("  WebTunnel v2.1 build gates")
    print(f"  dataset : {a.dataset}")
    print(f"  samples : {len(y_all)}  ({int(y_all.sum())} positive, {int((y_all == 0).sum())} negative)")
    print(f"  features: {len(names)}   flows on file: {len(flows)}   captures on disk: {n_on_disk}")
    print(f"  negatives in the TEST split: {int((d['y_test'] == 0).sum())} "
          f"-> FPR resolution floor {1.0/max(1,int((d['y_test']==0).sum())):.2e}")
    print(BAR)

    results = {}

    header("G1")
    results["G1"] = corpus_gates.stack_parity(flows) if flows else \
        {"passed": False, "reason": "no flow_records.jsonl"}

    header("G2")
    results["G2"] = tripwire.run(d["X_train"], d["y_train"], d["X_test"], d["y_test"], names)

    header("G3")
    results["G3"] = null_controls.run(X_all, y_all, ym_all, labels=labels_all,
                                      groups=socks_all, order=order_all, n_seeds=a.seeds)

    header("G4")
    results["G4"] = budget_parity.run(X_all, ym_all, names, CLASS_NAMES,
                                      budget_ids=budgets_all)

    header("G5")
    results["G5"] = split_integrity.run(d)

    header("G6")
    results["G6"] = corpus_gates.provenance(flows, manifests, n_captures_on_disk=n_on_disk,
                                            attrition=attrition) if flows else \
        {"passed": False, "reason": "no flow_records.jsonl"}

    print(f"\n{BAR}\n  SUMMARY\n{BAR}")
    failed = []
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        ok = bool(results.get(g, {}).get("passed"))
        if not ok:
            failed.append(g)
        print(f"  {g}  {'PASS' if ok else 'FAIL'}   {GATE_TITLES[g]}")

    if failed:
        print(f"\n  {len(failed)} gate(s) failed: {', '.join(failed)}")
        print("  A failing gate BLOCKS the build. No dataset reaches a model until all six are green.")
    else:
        print("\n  All six gates passed. Corpus is admissible.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  wrote {a.json}")

    sys.exit(0 if (not failed or a.allow_fail) else 1)


if __name__ == "__main__":
    main()
