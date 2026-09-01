#!/usr/bin/env python3
"""
v2 Multi-Core Dataset Builder & Flow Record Generator:
- Parses PCAPs and matches with CaptureManifest sidecars.
- Applies 5-tuple demux and TLS record extraction.
- Enforces strict conn_id disjoint split protocol.
- Emits flow_records.jsonl, tabular_dataset.npz, and sequence_dataset.npz.
"""
import os
import sys
import glob
import json
import argparse
import hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import (
    RAW_PCAP_DIR,
    PROCESSED_DATA_DIR,
    TABULAR_DATASET_PATH,
    SEQUENCE_DATASET_PATH,
    DATASET_SUMMARY_PATH,
    CLASSES,
    CLASS_MAP,
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED
)
from common.contracts import CaptureManifest, FlowRecord, assert_split_disjoint
from sanitizer import extract_flow_from_pcap, compute_flow_statistics, normalize_sequence_tensor, FEATURE_NAMES

FLOW_RECORDS_PATH = os.path.join(PROCESSED_DATA_DIR, "flow_records.jsonl")


def process_single_capture(pcap_path: str, post_handshake_only: bool = False) -> Optional[Dict[str, Any]]:
    manifest_path = os.path.splitext(pcap_path)[0] + ".manifest.json"
    manifest = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = CaptureManifest.from_json(f.read())
        except Exception:
            pass

    flow = extract_flow_from_pcap(pcap_path, manifest=manifest, post_handshake_only=post_handshake_only)
    if flow is None:
        return None

    # Packets as (t, signed_len)
    packets = [(t, d * l) for (t, d, l) in flow.records]
    if len(packets) < 3:
        return None

    feat_tab = compute_flow_statistics(packets)
    feat_seq = normalize_sequence_tensor(packets, max_seq_len=MAX_SEQUENCE_LENGTH)

    label_name = flow.label
    binary_label = 1 if label_name == "webtunnel" else 0
    multi_label = CLASS_MAP.get(label_name, 0)

    # Derive numeric session ID
    base = os.path.splitext(os.path.basename(pcap_path))[0]
    try:
        sid = int(base.split("_")[-1])
    except Exception:
        sid = int(hashlib.md5(flow.conn_id.encode()).hexdigest()[:6], 16) % 1000 + 1

    return {
        "pcap_path": pcap_path,
        "flow": flow,
        "class_name": label_name,
        "binary_label": binary_label,
        "multi_label": multi_label,
        "sample_id": sid,
        "conn_id": flow.conn_id,
        "tab": feat_tab,
        "seq": feat_seq,
        "pkt_count": len(packets)
    }


def main():
    parser = argparse.ArgumentParser(description="v2 Dataset Builder & Gate Preprocessor")
    parser.add_argument("--post-handshake", action="store_true", help="Isolate post-handshake packets")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    pcap_files = sorted(glob.glob(os.path.join(RAW_PCAP_DIR, "*.pcap")))

    if not pcap_files:
        print(f"No PCAP files found in {RAW_PCAP_DIR}!")
        return

    print(f"=== Found {len(pcap_files)} PCAP files. Processing with 5-tuple flow builder... ===")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_capture, p, args.post_handshake) for p in pcap_files]
        results = [f.result() for f in futures]

    valid = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid)} / {len(pcap_files)} valid flows.")

    if not valid:
        print("No valid samples extracted.")
        return

    X_tab = np.array([s["tab"] for s in valid], dtype=np.float32)
    X_seq = np.array([s["seq"] for s in valid], dtype=np.float32)
    y_bin = np.array([s["binary_label"] for s in valid], dtype=np.int64)
    y_mul = np.array([s["multi_label"] for s in valid], dtype=np.int64)
    sample_ids = np.array([s["sample_id"] for s in valid], dtype=np.int64)
    conn_ids = np.array([s["conn_id"] for s in valid])

    # Partition based on session ID (70% Train / 15% Val / 15% Test)
    max_sid = int(np.max(sample_ids)) if len(sample_ids) > 0 else 100
    train_cutoff = int(max_sid * 0.70)
    val_cutoff = int(max_sid * 0.85)

    train_idx = np.where(sample_ids <= train_cutoff)[0]
    val_idx = np.where((sample_ids > train_cutoff) & (sample_ids <= val_cutoff))[0]
    test_idx = np.where(sample_ids > val_cutoff)[0]

    # Verify split disjointness
    splits = {
        "train": conn_ids[train_idx],
        "val": conn_ids[val_idx],
        "test": conn_ids[test_idx]
    }
    assert_split_disjoint(splits)

    split_strategy = f"v2-Connection-Disjoint-Anti-Leakage (1-{train_cutoff} Train, {train_cutoff+1}-{val_cutoff} Val, {val_cutoff+1}-{max_sid} Test)"
    print(f"[{split_strategy}] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # 1. Save flow records JSONL
    with open(FLOW_RECORDS_PATH, "w", encoding="utf-8") as f:
        for item in valid:
            flow_obj = item["flow"]
            f.write(json.dumps(flow_obj.__dict__, ensure_ascii=False) + "\n")

    # 2. Save tabular dataset
    np.savez_compressed(
        TABULAR_DATASET_PATH,
        X_train=X_tab[train_idx], y_train=y_bin[train_idx], y_train_mul=y_mul[train_idx],
        X_val=X_tab[val_idx], y_val=y_bin[val_idx], y_val_mul=y_mul[val_idx],
        X_test=X_tab[test_idx], y_test=y_bin[test_idx], y_test_mul=y_mul[test_idx],
        sample_ids_train=sample_ids[train_idx], sample_ids_val=sample_ids[val_idx], sample_ids_test=sample_ids[test_idx],
        sample_ids_all=np.concatenate([sample_ids[train_idx], sample_ids[val_idx], sample_ids[test_idx]]),
        conn_ids_train=conn_ids[train_idx], conn_ids_val=conn_ids[val_idx], conn_ids_test=conn_ids[test_idx],
        conn_ids_all=np.concatenate([conn_ids[train_idx], conn_ids[val_idx], conn_ids[test_idx]]),
        feature_names=FEATURE_NAMES
    )

    # 3. Save sequence dataset
    np.savez_compressed(
        SEQUENCE_DATASET_PATH,
        X_train=X_seq[train_idx], y_train=y_bin[train_idx], y_train_mul=y_mul[train_idx],
        X_val=X_seq[val_idx], y_val=y_bin[val_idx], y_val_mul=y_mul[val_idx],
        X_test=X_seq[test_idx], y_test=y_bin[test_idx], y_test_mul=y_mul[test_idx],
        sample_ids_train=sample_ids[train_idx], sample_ids_val=sample_ids[val_idx], sample_ids_test=sample_ids[test_idx],
        sample_ids_all=np.concatenate([sample_ids[train_idx], sample_ids[val_idx], sample_ids[test_idx]]),
        conn_ids_train=conn_ids[train_idx], conn_ids_val=conn_ids[val_idx], conn_ids_test=conn_ids[test_idx],
        conn_ids_all=np.concatenate([conn_ids[train_idx], conn_ids[val_idx], conn_ids[test_idx]])
    )

    # 4. Save summary
    class_counts = {}
    for c in CLASSES:
        cnt = sum(1 for s in valid if s["class_name"] == c)
        if cnt > 0:
            class_counts[c] = cnt

    summary = {
        "total_samples": len(valid),
        "split_strategy": split_strategy,
        "classes": class_counts,
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        },
        "tabular_feature_count": len(FEATURE_NAMES),
        "sequence_shape": [MAX_SEQUENCE_LENGTH, 2]
    }

    with open(DATASET_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"[OK] Saved {TABULAR_DATASET_PATH} and {SEQUENCE_DATASET_PATH}")
    print(f"[OK] Flow records saved to {FLOW_RECORDS_PATH}")


if __name__ == "__main__":
    main()
