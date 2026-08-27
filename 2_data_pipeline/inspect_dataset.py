#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sanitizer import extract_raw_packets_from_pcap

RAW_PCAP_DIR = "data/raw_pcap"
PLOT_DIR = "4_evaluation/plots"

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    classes = ["webtunnel", "direct_web_browsing", "websocket_ticker", "websocket_chat", "video_streaming", "web_assets"]
    lengths_per_class = {c: [] for c in classes}
    iats_per_class = {c: [] for c in classes}
    
    for c in classes:
        files = glob.glob(os.path.join(RAW_PCAP_DIR, f"{c}_*.pcap"))
        for f in files:
            pkts = extract_raw_packets_from_pcap(f)
            if not pkts:
                continue
            lens = [abs(p[1]) for p in pkts]
            ts = [p[0] for p in pkts]
            iats = np.diff(ts)
            lengths_per_class[c].extend(lens)
            iats_per_class[c].extend(iats)
            
    # 1. Packet Length Distribution (Spectral Quantization Plot)
    plt.figure(figsize=(12, 6))
    for c in classes:
        data = lengths_per_class[c]
        if data:
            sns.kdeplot(data, label=c, bw_adjust=0.5, common_norm=False)
            
    plt.title("Packet Length Distribution: WebTunnel vs Hard Negatives (514-Byte Cell Quantization)", fontsize=14, fontweight="bold")
    plt.xlabel("Packet Length (Bytes)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.axvline(x=624, color="r", linestyle="--", alpha=0.7, label="1x Tor Cell (~624B)")
    plt.axvline(x=1138, color="purple", linestyle="--", alpha=0.7, label="2x Tor Cell (~1138B)")
    plt.axvline(x=1500, color="gray", linestyle=":", alpha=0.5, label="MTU (1500B)")
    plt.xlim(0, 1550)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "packet_length_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Log Inter-Arrival Time Distribution
    plt.figure(figsize=(12, 6))
    for c in classes:
        data = np.array(iats_per_class[c])
        if len(data) > 0:
            clean_iats = np.maximum(data, 1e-7)
            log_iats = np.log10(clean_iats)
            sns.kdeplot(log_iats, label=c, common_norm=False)
            
    plt.title("Log Inter-Arrival Time (IAT) Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Log10(IAT in Seconds)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "iat_distribution.png"), dpi=300)
    plt.close()
    
    print(f"[OK] Diagnostic plots generated in {PLOT_DIR}/")
    print(f"  - {PLOT_DIR}/packet_length_distribution.png")
    print(f"  - {PLOT_DIR}/iat_distribution.png")

if __name__ == "__main__":
    main()
