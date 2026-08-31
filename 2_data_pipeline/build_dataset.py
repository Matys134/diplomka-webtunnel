#!/usr/bin/env python3
"""
Multi-core dataset builder and feature extractor.
Applies:
- L7 Anti-Leakage sanitization (sanitizer.py)
- 48 Tabular flow statistical features extraction
- 200x2 Sequence tensor conversion
- Strict Session-Stratified-Anti-Leakage splitting (1-70 Train, 71-85 Val, 86-100 Test)
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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
    MAX_SEQUENCE_LENGTH
)
from sanitizer import extract_raw_packets_from_pcap, compute_flow_statistics, build_sequence_tensor, FEATURE_NAMES


def process_single_pcap(pcap_path: str, post_handshake_only: bool = False):
    basename = os.path.basename(pcap_path)
    label_name = None
    for k in CLASSES:
        if basename.startswith(k):
            label_name = k
            break

    if label_name is None:
        return None

    binary_label = 1 if label_name == "webtunnel" else 0
    multi_label = CLASS_MAP[label_name]

    packets = extract_raw_packets_from_pcap(pcap_path, post_handshake_only=post_handshake_only)
    if len(packets) < 3:
        return None

    feat_tab = compute_flow_statistics(packets)
    feat_seq = build_sequence_tensor(packets, max_seq_len=MAX_SEQUENCE_LENGTH)

    return {
        "path": pcap_path,
        "class_name": label_name,
        "binary_label": binary_label,
        "multi_label": multi_label,
        "tab": feat_tab,
        "seq": feat_seq,
        "pkt_count": len(packets)
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Core Dataset Builder and Sanitizer")
    parser.add_argument("--post-handshake", action="store_true", help="Isolate post-handshake packets (strip TLS ClientHello)")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    pcap_files = sorted(glob.glob(os.path.join(RAW_PCAP_DIR, "*.pcap")))

    if not pcap_files:
        print(f"No PCAP files found in {RAW_PCAP_DIR}!")
        return

    print(f"=== Found {len(pcap_files)} PCAP files. Extracting features with multi-core sanitization... ===")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_pcap, p, args.post_handshake) for p in pcap_files]
        results = [f.result() for f in futures]

    valid_samples = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_samples)} valid flows.")

    if not valid_samples:
        print("No valid samples extracted.")
        return

    X_tab = np.array([s["tab"] for s in valid_samples], dtype=np.float32)
    X_seq = np.array([s["seq"] for s in valid_samples], dtype=np.float32)
    y_bin = np.array([s["binary_label"] for s in valid_samples], dtype=np.int64)
    y_mul = np.array([s["multi_label"] for s in valid_samples], dtype=np.int64)
    class_names = [s["class_name"] for s in valid_samples]

    # Parse session IDs for strict Anti-Leakage splitting
    sample_ids = []
    for s in valid_samples:
        base = os.path.splitext(os.path.basename(s["path"]))[0]
        try:
            sid = int(base.split("_")[-1])
        except Exception:
            sid = -1
        sample_ids.append(sid)
    sample_ids = np.array(sample_ids)

    if not np.all(sample_ids > 0):
        invalid_count = int(np.sum(sample_ids <= 0))
        raise ValueError(f"CRITICAL ERROR: {invalid_count} PCAPs have unparseable session IDs! Cannot guarantee anti-leakage session split.")

    max_sid = int(np.max(sample_ids))
    train_cutoff = int(max_sid * 0.70)
    val_cutoff = int(max_sid * 0.85)

    train_idx = np.where(sample_ids <= train_cutoff)[0]
    val_idx = np.where((sample_ids > train_cutoff) & (sample_ids <= val_cutoff))[0]
    test_idx = np.where(sample_ids > val_cutoff)[0]
    split_strategy = f"Session-Stratified-Anti-Leakage (1-{train_cutoff} Train, {train_cutoff+1}-{val_cutoff} Val, {val_cutoff+1}-{max_sid} Test)"

    print(f"[{split_strategy}] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    np.savez_compressed(
        TABULAR_DATASET_PATH,
        X_train=X_tab[train_idx], y_train=y_bin[train_idx], y_train_mul=y_mul[train_idx],
        X_val=X_tab[val_idx], y_val=y_bin[val_idx], y_val_mul=y_mul[val_idx],
        X_test=X_tab[test_idx], y_test=y_bin[test_idx], y_test_mul=y_mul[test_idx],
        sample_ids_train=sample_ids[train_idx], sample_ids_val=sample_ids[val_idx], sample_ids_test=sample_ids[test_idx],
        sample_ids_all=sample_ids,
        feature_names=FEATURE_NAMES
    )

    np.savez_compressed(
        SEQUENCE_DATASET_PATH,
        X_train=X_seq[train_idx], y_train=y_bin[train_idx], y_train_mul=y_mul[train_idx],
        X_val=X_seq[val_idx], y_val=y_bin[val_idx], y_val_mul=y_mul[val_idx],
        X_test=X_seq[test_idx], y_test=y_bin[test_idx], y_test_mul=y_mul[test_idx],
        sample_ids_train=sample_ids[train_idx], sample_ids_val=sample_ids[val_idx], sample_ids_test=sample_ids[test_idx],
        sample_ids_all=sample_ids
    )

    summary = {
        "total_samples": len(valid_samples),
        "split_strategy": split_strategy,
        "classes": {k: int(np.sum(np.array(class_names) == k)) for k in CLASSES},
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        },
        "tabular_feature_count": len(FEATURE_NAMES),
        "sequence_shape": list(X_seq.shape[1:])
    }

    with open(DATASET_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[OK] Processed dataset saved to {PROCESSED_DATA_DIR}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
