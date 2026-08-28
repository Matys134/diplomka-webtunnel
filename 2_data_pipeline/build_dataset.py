#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sanitizer import extract_raw_packets_from_pcap, compute_flow_statistics, build_sequence_tensor, FEATURE_NAMES

PROCESSED_DIR = "data/processed"
RAW_PCAP_DIR = "data/raw_pcap"

CLASS_MAPPING = {
    "webtunnel": 1,             # Target circumvention class
    "direct_web_browsing": 0,  # Hard Negative 1 (Direct Browsing)
    "websocket_ticker": 0,      # Hard Negative 2
    "websocket_chat": 0,        # Hard Negative 3
    "video_streaming": 0,       # Hard Negative 4
    "web_assets": 0,            # Hard Negative 5
}

MULTI_CLASS_MAPPING = {
    "webtunnel": 0,
    "direct_web_browsing": 1,
    "websocket_ticker": 2,
    "websocket_chat": 3,
    "video_streaming": 4,
    "web_assets": 5,
}

def process_single_pcap(pcap_path: str, post_handshake_only: bool = False):
    basename = os.path.basename(pcap_path)
    label_name = None
    for k in CLASS_MAPPING.keys():
        if basename.startswith(k):
            label_name = k
            break
            
    if label_name is None:
        return None
        
    binary_label = CLASS_MAPPING[label_name]
    multi_label = MULTI_CLASS_MAPPING[label_name]
    
    packets = extract_raw_packets_from_pcap(pcap_path, post_handshake_only=post_handshake_only)
    if len(packets) < 3:
        return None
        
    feat_tab = compute_flow_statistics(packets)
    feat_seq = build_sequence_tensor(packets, max_seq_len=200)
    
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
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio from train")
    args = parser.parse_args()
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
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
    
    # Strict Anti-Leakage: Session/Temporal Split based on capture batch IDs
    # Guarantees that test flows originate from completely separate capture runs/sessions
    sample_ids = []
    for s in valid_samples:
        base = os.path.splitext(os.path.basename(s["path"]))[0]
        try:
            sid = int(base.split("_")[-1])
        except Exception:
            sid = -1
        sample_ids.append(sid)
    sample_ids = np.array(sample_ids)
    
    # If standard 1..100 numbering exists, split by session blocks (1-70 Train, 71-85 Val, 86-100 Test)
    if np.all(sample_ids > 0):
        train_idx = np.where(sample_ids <= 70)[0]
        val_idx = np.where((sample_ids > 70) & (sample_ids <= 85))[0]
        test_idx = np.where(sample_ids > 85)[0]
    else:
        # Fallback to stratified group split
        indices = np.arange(len(valid_samples))
        train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42, stratify=y_mul)
        train_idx, val_idx = train_test_split(train_idx, test_size=args.val_size, random_state=42, stratify=y_mul[train_idx])
        
    print(f"Session-Stratified Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "tabular_dataset.npz"),
        X_train=X_tab[train_idx], y_train=y_bin[train_idx], y_train_mul=y_mul[train_idx],
        X_val=X_tab[val_idx], y_val=y_bin[val_idx], y_val_mul=y_mul[val_idx],
        X_test=X_tab[test_idx], y_test=y_bin[test_idx], y_test_mul=y_mul[test_idx],
        feature_names=FEATURE_NAMES
    )
    
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "sequence_dataset.npz"),
        X_train=X_seq[train_idx], y_train=y_bin[train_idx], y_train_mul=y_mul[train_idx],
        X_val=X_seq[val_idx], y_val=y_bin[val_idx], y_val_mul=y_mul[val_idx],
        X_test=X_seq[test_idx], y_test=y_bin[test_idx], y_test_mul=y_mul[test_idx]
    )
    
    summary = {
        "total_samples": len(valid_samples),
        "classes": {k: int(np.sum(np.array(class_names) == k)) for k in MULTI_CLASS_MAPPING.keys()},
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        },
        "tabular_feature_count": len(FEATURE_NAMES),
        "sequence_shape": list(X_seq.shape[1:])
    }
    
    with open(os.path.join(PROCESSED_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n[OK] Processed dataset saved to {PROCESSED_DIR}/")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
