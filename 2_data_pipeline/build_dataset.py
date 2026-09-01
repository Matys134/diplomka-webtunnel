#!/usr/bin/env python3
"""
v2.1 dataset builder.

The split is keyed on SOCKET IDENTITY, not on the trailing integer of the filename.

v2.0 split on `sample_id` and then asserted conn_id disjointness -- but conn_id mixes in the SYN
timestamp, so it is unique per capture by construction and the assertion could never fail.
Meanwhile client port 56446 appeared in 234 captures spanning train, val and test.  Grouping on
`socket_id` ("ip:port->ip:port/proto") is what actually makes that visible, so that is the
grouping key here, and BOTH keys are asserted disjoint.

Also new: every capture that does not become a flow is counted with a reason (gate G6), and the
per-row key vectors needed by the gates (socket_id, conn_id, budget_id, dest_id, profile, epoch,
provenance) are written into the .npz so nothing downstream has to re-derive them.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (          # noqa: E402
    RAW_PCAP_DIR, PROCESSED_DATA_DIR, TABULAR_DATASET_PATH, SEQUENCE_DATASET_PATH,
    DATASET_SUMMARY_PATH, CLASSES, CLASS_MAP, MAX_SEQUENCE_LENGTH, RANDOM_SEED,
)
from common.contracts import assert_split_disjoint  # noqa: E402
from sanitizer import (              # noqa: E402
    FEATURE_NAMES, compute_flow_statistics, extract_flow_from_pcap, load_manifest_for,
    normalize_sequence_tensor,
)

FLOW_RECORDS_PATH = os.path.join(PROCESSED_DATA_DIR, "flow_records.jsonl")
ATTRITION_PATH = os.path.join(PROCESSED_DATA_DIR, "attrition_build.json")


def process_one(pcap_path: str, post_handshake: bool) -> Dict[str, Any]:
    manifest = load_manifest_for(pcap_path)
    label = manifest.label if manifest else "unknown"
    flow, reason = extract_flow_from_pcap(pcap_path, manifest=manifest,
                                          post_handshake_only=post_handshake)
    if flow is None:
        return {"ok": False, "label": label, "reason": reason or "unknown",
                "capture_id": os.path.basename(pcap_path)[:-5]}

    packets = [(t, d * l) for (t, d, l) in flow.records]
    if len(packets) < 3:
        return {"ok": False, "label": label, "reason": "too_few_records",
                "capture_id": flow.capture_id}

    return {
        "ok": True,
        "flow": flow,
        "tab": compute_flow_statistics(packets),
        "seq": normalize_sequence_tensor(packets, max_seq_len=MAX_SEQUENCE_LENGTH),
        "label": flow.label,
        "binary": 1 if flow.label == "webtunnel" else 0,
        "multi": CLASS_MAP.get(flow.label, 0),
    }


def split_by_group(groups: List[str], labels: List[str], seed: int,
                   frac=(0.70, 0.15, 0.15)) -> Dict[str, str]:
    """Assign whole GROUPS (sockets) to train/val/test, stratified within each class.

    A group never straddles a split, which is the property v2.0 claimed and did not have.
    """
    rng = np.random.RandomState(seed)
    by_class: Dict[str, Dict[str, int]] = defaultdict(Counter)
    for g, lab in zip(groups, labels):
        by_class[lab][g] += 1

    assignment: Dict[str, str] = {}
    for lab, counter in by_class.items():
        gs = sorted(counter)
        rng.shuffle(gs)
        total = sum(counter.values())
        seen, cuts = 0, (frac[0] * total, (frac[0] + frac[1]) * total)
        for g in gs:
            assignment[g] = "train" if seen < cuts[0] else ("val" if seen < cuts[1] else "test")
            seen += counter[g]
    return assignment


def main():
    ap = argparse.ArgumentParser(description="v2.1 dataset builder")
    ap.add_argument("--post-handshake", action="store_true",
                    help="drop everything up to the client's first post-Finished record")
    ap.add_argument("--group-key", default="socket_id", choices=["socket_id", "conn_id"],
                    help="split grouping key (socket_id is the strict one)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    pcaps = sorted(glob.glob(os.path.join(RAW_PCAP_DIR, "*.pcap")))
    if not pcaps:
        print(f"No PCAP files in {RAW_PCAP_DIR}")
        return

    print(f"=== v2.1 flow builder: {len(pcaps)} captures, grouping on {args.group_key} ===")
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(process_one, pcaps, [args.post_handshake] * len(pcaps), chunksize=8))

    valid = [r for r in results if r["ok"]]
    drops: Dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if not r["ok"]:
            drops[r["label"]][r["reason"].split(":")[0]] += 1

    print(f"    {len(valid)}/{len(pcaps)} captures became flows "
          f"({100*len(valid)/len(pcaps):.1f}%)")
    print("\n    attrition by class and reason (gate G6 consumes this):")
    for lab in sorted(drops):
        for reason, n in drops[lab].most_common():
            print(f"      {lab:<24}{reason:<36}{n:5d}")
    with open(ATTRITION_PATH, "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k, v in drops.items()}, f, indent=2)

    if not valid:
        print("\n    No usable flows. Nothing written.")
        return

    flows = [r["flow"] for r in valid]
    X_tab = np.asarray([r["tab"] for r in valid], dtype=np.float32)
    X_seq = np.asarray([r["seq"] for r in valid], dtype=np.float32)
    y_bin = np.asarray([r["binary"] for r in valid], dtype=np.int64)
    y_mul = np.asarray([r["multi"] for r in valid], dtype=np.int64)

    keys = {
        "socket_ids": np.asarray([f.socket_id for f in flows]),
        "conn_ids": np.asarray([f.conn_id for f in flows]),
        "capture_ids": np.asarray([f.capture_id for f in flows]),
        "budget_ids": np.asarray([f.budget_id for f in flows]),
        "dest_ids": np.asarray([f.dest_id for f in flows]),
        "profiles": np.asarray([f.profile for f in flows]),
        "behaviours": np.asarray([f.behaviour for f in flows]),
        "epochs": np.asarray([f.epoch for f in flows]),
        "provenance": np.asarray([f.provenance for f in flows]),
        "labels": np.asarray([f.label for f in flows]),
        "ja4": np.asarray([f.ja4 or "" for f in flows]),
        "clienthello_len": np.asarray([f.clienthello_len or 0 for f in flows], dtype=np.int64),
        "saw_client_syn": np.asarray([f.saw_client_syn for f in flows]),
        "t_start": np.asarray([f.t_start for f in flows], dtype=np.float64),
        "target_duration_s": np.asarray([f.target_duration_s for f in flows], dtype=np.float32),
        "target_bytes_up": np.asarray([f.target_bytes_up for f in flows], dtype=np.float32),
        "target_bytes_down": np.asarray([f.target_bytes_down for f in flows], dtype=np.float32),
    }

    group_vec = [getattr(f, args.group_key) for f in flows]
    assignment = split_by_group(group_vec, [f.label for f in flows], args.seed)
    which = np.asarray([assignment[g] for g in group_vec])
    idx = {s: np.where(which == s)[0] for s in ("train", "val", "test")}

    # The assertions v2.0 could not fail.  Both keys, both directions.
    assert_split_disjoint({s: keys["socket_ids"][i] for s, i in idx.items()}, "socket_id")
    assert_split_disjoint({s: keys["conn_ids"][i] for s, i in idx.items()}, "conn_id")

    order = np.concatenate([idx["train"], idx["val"], idx["test"]])
    print(f"\n    split (group={args.group_key}): train={len(idx['train'])} "
          f"val={len(idx['val'])} test={len(idx['test'])}")
    for s in ("train", "val", "test"):
        pos = int(y_bin[idx[s]].sum())
        print(f"      {s:<6} {len(idx[s]):5d} flows  {pos:4d} positive  "
              f"{len(set(keys['socket_ids'][idx[s]])):4d} sockets")

    with open(FLOW_RECORDS_PATH, "w", encoding="utf-8") as f:
        for fl in flows:
            f.write(json.dumps(fl.__dict__, ensure_ascii=False) + "\n")

    payload = {
        "X_train": X_tab[idx["train"]], "y_train": y_bin[idx["train"]], "y_train_mul": y_mul[idx["train"]],
        "X_val": X_tab[idx["val"]], "y_val": y_bin[idx["val"]], "y_val_mul": y_mul[idx["val"]],
        "X_test": X_tab[idx["test"]], "y_test": y_bin[idx["test"]], "y_test_mul": y_mul[idx["test"]],
        "feature_names": np.asarray(FEATURE_NAMES),
        "split_of_row": which[order],
        "group_key": np.asarray([args.group_key]),
    }
    for name, arr in keys.items():
        payload[f"{name}_all"] = arr[order]
        for s in ("train", "val", "test"):
            payload[f"{name}_{s}"] = arr[idx[s]]
    np.savez_compressed(TABULAR_DATASET_PATH, **payload)

    seq_payload = dict(payload)
    seq_payload.update(X_train=X_seq[idx["train"]], X_val=X_seq[idx["val"]], X_test=X_seq[idx["test"]])
    seq_payload.pop("feature_names", None)
    np.savez_compressed(SEQUENCE_DATASET_PATH, **seq_payload)

    summary = {
        "total_captures": len(pcaps),
        "total_flows": len(valid),
        "yield_pct": round(100 * len(valid) / len(pcaps), 2),
        "group_key": args.group_key,
        "split_strategy": f"v2.1 group-disjoint on {args.group_key} (70/15/15, stratified by class)",
        "classes": dict(Counter(f.label for f in flows)),
        "distinct_sockets": len(set(keys["socket_ids"])),
        "distinct_sockets_by_class": {
            c: len({f.socket_id for f in flows if f.label == c})
            for c in sorted({f.label for f in flows})},
        "flows_with_client_syn": int(sum(f.saw_client_syn for f in flows)),
        "flows_with_full_handshake": int(sum(f.saw_full_handshake for f in flows)),
        "provenance": dict(Counter(f.provenance for f in flows)),
        "splits": {s: int(len(i)) for s, i in idx.items()},
        "negatives_in_test": int((y_bin[idx["test"]] == 0).sum()),
        "fpr_resolution_floor": (1.0 / max(1, int((y_bin[idx["test"]] == 0).sum()))),
        "tabular_feature_count": len(FEATURE_NAMES),
        "sequence_shape": [MAX_SEQUENCE_LENGTH, 2],
        "attrition": {k: dict(v) for k, v in drops.items()},
    }
    with open(DATASET_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\n    distinct sockets: {summary['distinct_sockets']}  "
          f"(webtunnel: {summary['distinct_sockets_by_class'].get('webtunnel', 0)})")
    print(f"    FPR resolution floor on the test split: "
          f"{summary['fpr_resolution_floor']:.2e}")
    print(f"[OK] {TABULAR_DATASET_PATH}\n[OK] {SEQUENCE_DATASET_PATH}\n[OK] {FLOW_RECORDS_PATH}")


if __name__ == "__main__":
    main()
