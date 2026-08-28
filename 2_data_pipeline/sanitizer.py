import dpkt
import socket
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any, Optional

FEATURE_NAMES = [
    # 0-9: Total packet lengths stats
    "len_min", "len_max", "len_mean", "len_std", "len_skew",
    "len_p10", "len_p25", "len_p50", "len_p75", "len_p90",
    # 10-18: Upstream packet lengths stats
    "up_len_min", "up_len_max", "up_len_mean", "up_len_std",
    "up_len_p10", "up_len_p25", "up_len_p50", "up_len_p75", "up_len_p90",
    # 19-27: Downstream packet lengths stats
    "down_len_min", "down_len_max", "down_len_mean", "down_len_std",
    "down_len_p10", "down_len_p25", "down_len_p50", "down_len_p75", "down_len_p90",
    # 28-36: Inter-arrival time (IAT) stats
    "iat_min", "iat_max", "iat_mean", "iat_std",
    "iat_p10", "iat_p25", "iat_p50", "iat_p75", "iat_p90",
    # 37-43: Burst dynamics
    "burst_count", "burst_len_mean_pkts", "burst_len_std_pkts",
    "burst_len_mean_bytes", "burst_len_std_bytes", "burst_dur_mean", "burst_dur_std",
    # 44-47: Flow ratios & totals
    "ratio_up_pkts", "ratio_up_bytes", "total_pkts", "total_bytes"
]

def extract_raw_packets_from_pcap(pcap_path: str, post_handshake_only: bool = False) -> List[Tuple[float, int]]:
    """
    Parses PCAP file and extracts sanitized (relative_time, signed_length) tuples.
    Applies strict Anti-Leakage stripping:
      - Ignores L2 (MAC), L3 (IP addresses, TTL), L4 (ports, sequence/ACK, TCP options).
      - Normalizes direction: +1 (Client -> Server), -1 (Server -> Client).
    """
    packets = []
    client_ip = None
    first_ts = None
    
    with open(pcap_path, 'rb') as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except Exception:
            f.seek(0)
            try:
                pcap = dpkt.pcapng.Reader(f)
            except Exception as e:
                print(f"Error opening PCAP {pcap_path}: {e}")
                return []
                
        for ts, buf in pcap:
            if first_ts is None:
                first_ts = ts
            rel_ts = ts - first_ts
            
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
                
            if not isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue
                
            ip = eth.data
            src_ip = ip.src
            dst_ip = ip.dst
            
            # Establish client IP from initial SYN or first packet
            if client_ip is None:
                client_ip = src_ip
                
            direction = 1 if src_ip == client_ip else -1
            pkt_len = len(ip)  # L3 packet length (clamped to MTU 1500)
            pkt_len = min(pkt_len, 1500)
            
            signed_len = direction * pkt_len
            packets.append((rel_ts, signed_len))
            
    # Post-handshake isolation (skip first 15 packets to eliminate TCP 3-way handshake & TLS 1.3 Key Exchange)
    if post_handshake_only and len(packets) > 15:
        packets = packets[15:]
        if packets:
            base_t = packets[0][0]
            packets = [(t - base_t, l) for t, l in packets]
            
    return packets

def compute_flow_statistics(packets: List[Tuple[float, int]]) -> np.ndarray:
    """Computes 48 statistical flow features for tree models (XGBoost/RandomForest)."""
    if len(packets) < 2:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        
    timestamps = np.array([p[0] for p in packets], dtype=np.float64)
    signed_lens = np.array([p[1] for p in packets], dtype=np.float64)
    abs_lens = np.abs(signed_lens)
    
    up_lens = abs_lens[signed_lens > 0]
    down_lens = abs_lens[signed_lens < 0]
    
    if len(up_lens) == 0: up_lens = np.array([0.0])
    if len(down_lens) == 0: down_lens = np.array([0.0])
    
    iats = np.diff(timestamps)
    if len(iats) == 0: iats = np.array([0.0])
    
    # Burst analysis
    bursts_pkts = []
    bursts_bytes = []
    bursts_dur = []
    
    curr_dir = np.sign(signed_lens[0])
    curr_pkts = 1
    curr_bytes = abs_lens[0]
    curr_start_t = timestamps[0]
    
    for i in range(1, len(signed_lens)):
        d = np.sign(signed_lens[i])
        if d == curr_dir:
            curr_pkts += 1
            curr_bytes += abs_lens[i]
        else:
            bursts_pkts.append(curr_pkts)
            bursts_bytes.append(curr_bytes)
            bursts_dur.append(timestamps[i-1] - curr_start_t)
            curr_dir = d
            curr_pkts = 1
            curr_bytes = abs_lens[i]
            curr_start_t = timestamps[i]
            
    bursts_pkts.append(curr_pkts)
    bursts_bytes.append(curr_bytes)
    bursts_dur.append(timestamps[-1] - curr_start_t)
    
    b_pkts = np.array(bursts_pkts)
    b_bytes = np.array(bursts_bytes)
    b_dur = np.array(bursts_dur)
    
    # 48 feature vector construction
    feat = [
        # Total lengths
        float(np.min(abs_lens)), float(np.max(abs_lens)), float(np.mean(abs_lens)), float(np.std(abs_lens)), float(stats.skew(abs_lens) if len(abs_lens) > 2 and np.std(abs_lens) > 1e-5 else 0.0),
        float(np.percentile(abs_lens, 10)), float(np.percentile(abs_lens, 25)), float(np.percentile(abs_lens, 50)), float(np.percentile(abs_lens, 75)), float(np.percentile(abs_lens, 90)),
        # Upstream lengths
        float(np.min(up_lens)), float(np.max(up_lens)), float(np.mean(up_lens)), float(np.std(up_lens)),
        float(np.percentile(up_lens, 10)), float(np.percentile(up_lens, 25)), float(np.percentile(up_lens, 50)), float(np.percentile(up_lens, 75)), float(np.percentile(up_lens, 90)),
        # Downstream lengths
        float(np.min(down_lens)), float(np.max(down_lens)), float(np.mean(down_lens)), float(np.std(down_lens)),
        float(np.percentile(down_lens, 10)), float(np.percentile(down_lens, 25)), float(np.percentile(down_lens, 50)), float(np.percentile(down_lens, 75)), float(np.percentile(down_lens, 90)),
        # IAT stats
        float(np.min(iats)), float(np.max(iats)), float(np.mean(iats)), float(np.std(iats)),
        float(np.percentile(iats, 10)), float(np.percentile(iats, 25)), float(np.percentile(iats, 50)), float(np.percentile(iats, 75)), float(np.percentile(iats, 90)),
        # Burst dynamics
        float(len(b_pkts)), float(np.mean(b_pkts)), float(np.std(b_pkts)),
        float(np.mean(b_bytes)), float(np.std(b_bytes)), float(np.mean(b_dur)), float(np.std(b_dur)),
        # Ratios & Totals
        float(len(up_lens) / max(len(abs_lens), 1)),
        float(np.sum(up_lens) / max(np.sum(abs_lens), 1.0)),
        float(len(abs_lens)),
        float(np.sum(abs_lens))
    ]
    return np.array(feat, dtype=np.float32)

def build_sequence_tensor(packets: List[Tuple[float, int]], max_seq_len: int = 200) -> np.ndarray:
    """
    Builds a 2D tensor of shape (max_seq_len, 2) for 1D-CNN and Transformers.
      - Channel 0: Normalized signed packet length in [-1.0, 1.0].
      - Channel 1: Scaled log inter-arrival time ln(1 + delta_t) / 10.0.
    """
    tensor = np.zeros((max_seq_len, 2), dtype=np.float32)
    if not packets:
        return tensor
        
    n = min(len(packets), max_seq_len)
    prev_t = packets[0][0]
    
    for i in range(n):
        t, signed_len = packets[i]
        delta_t = max(0.0, t - prev_t)
        prev_t = t
        
        # Normalization
        norm_len = signed_len / 1500.0  # Range [-1.0, 1.0]
        norm_iat = np.log1p(delta_t) / 10.0  # Log compressed
        
        tensor[i, 0] = np.clip(norm_len, -1.0, 1.0)
        tensor[i, 1] = np.clip(norm_iat, 0.0, 1.0)
        
    return tensor
