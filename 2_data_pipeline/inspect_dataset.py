#!/usr/bin/env python3
"""
Generates spectral packet length quantization and IAT distribution diagnostic plots.
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import RAW_PCAP_DIR, PLOTS_DIR, CLASSES, CLASS_DISPLAY_NAMES, setup_matplotlib_style
from sanitizer import extract_raw_packets_from_pcap, load_manifest_for


def main():
    setup_matplotlib_style()

    lengths_per_class = {c: [] for c in CLASSES}
    iats_per_class = {c: [] for c in CLASSES}

    for c in CLASSES:
        files = glob.glob(os.path.join(RAW_PCAP_DIR, f"{c}_*.pcap"))
        for f in files:
            manifest = load_manifest_for(f)
            if manifest is None:
                continue
            pkts = extract_raw_packets_from_pcap(f, manifest=manifest)
            if not pkts:
                continue
            lens = [abs(p[1]) for p in pkts]
            ts = [p[0] for p in pkts]
            iats = np.diff(ts)
            lengths_per_class[c].extend(lens)
            iats_per_class[c].extend(iats)

    # 1. Packet Length Distribution (Spectral Quantization Plot)
    plt.figure(figsize=(12, 6))
    for c in CLASSES:
        data = lengths_per_class[c]
        if data:
            sns.kdeplot(data, label=CLASS_DISPLAY_NAMES.get(c, c), bw_adjust=0.5, common_norm=False)

    plt.title("Packet Length Distribution: WebTunnel vs Hard Negatives (514-Byte Cell Quantization)")
    plt.xlabel("L7 Application Payload Length (Bytes)")
    plt.ylabel("Density")
    plt.axvline(x=560, color="r", linestyle="--", alpha=0.7, label="1x Tor Cell L7 (~560B: 514B + H2/TLS)")
    plt.axvline(x=1074, color="purple", linestyle="--", alpha=0.7, label="2x Tor Cell L7 (~1074B: 1028B + H2/TLS)")
    plt.axvline(x=1448, color="gray", linestyle=":", alpha=0.5, label="Max L7 MSS (1448B)")
    plt.xlim(0, 1550)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "packet_length_distribution.png"))
    plt.close()

    # 2. Log Inter-Arrival Time Distribution
    plt.figure(figsize=(12, 6))
    for c in CLASSES:
        data = np.array(iats_per_class[c])
        if len(data) > 0:
            clean_iats = np.maximum(data, 1e-7)
            log_iats = np.log10(clean_iats)
            sns.kdeplot(log_iats, label=CLASS_DISPLAY_NAMES.get(c, c), common_norm=False)

    plt.title("Log Inter-Arrival Time (IAT) Distribution across Traffic Classes")
    plt.xlabel("Log10(IAT in Seconds)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "iat_distribution.png"))
    plt.close()

    print(f"[OK] Diagnostic plots generated in {PLOTS_DIR}/")
    print(f"  - {os.path.join(PLOTS_DIR, 'packet_length_distribution.png')}")
    print(f"  - {os.path.join(PLOTS_DIR, 'iat_distribution.png')}")


if __name__ == "__main__":
    main()
