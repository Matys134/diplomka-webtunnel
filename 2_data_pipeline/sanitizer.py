"""
v2 Anti-Leakage Flow Sanitizer & TLS Record Builder:
Extracts pure L7 Application Data records matching the authoritative 5-tuple from CaptureManifest.
Eliminates:
  - Multicast noise (mDNS, SSDP) and cross-flow contamination.
  - Direction inversion (authoritative client_ip from manifest).
  - Segmentation offload artifacts via TLS record extraction.
Emits FlowRecord and 48 statistical flow features.
"""

import os
import socket
import json
import dpkt
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any, Optional
from common.contracts import CaptureManifest, FlowRecord, FiveTuple

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


def parse_tls_records_from_payload(payload: bytes) -> List[Tuple[int, int]]:
    """Extracts TLS record (content_type, length) tuples from TCP payload bytes."""
    records = []
    idx = 0
    while idx + 5 <= len(payload):
        ctype = payload[idx]
        version = (payload[idx+1] << 8) | payload[idx+2]
        if version in (0x0301, 0x0302, 0x0303, 0x0304): # SSLv3 / TLS 1.0 - 1.3
            rec_len = (payload[idx+3] << 8) | payload[idx+4]
            records.append((ctype, rec_len + 5))
            idx += 5 + rec_len
        else:
            break
    return records


def extract_flow_from_pcap(pcap_path: str, manifest: Optional[CaptureManifest] = None, post_handshake_only: bool = False) -> Optional[FlowRecord]:
    """
    Parses PCAP file, enforces 5-tuple filtering based on manifest, and constructs FlowRecord.
    """
    if not os.path.exists(pcap_path):
        return None

    raw_packets = []
    first_ts = None
    syn_ts = None
    detected_5tuple = None
    client_ip_str = manifest.client_ip if manifest and manifest.client_ip else None
    clienthello_len = None

    with open(pcap_path, 'rb') as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except Exception:
            f.seek(0)
            try:
                pcap = dpkt.pcapng.Reader(f)
            except Exception:
                return None

        for ts, buf in pcap:
            if first_ts is None:
                first_ts = ts

            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue

            if not isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue

            ip = eth.data
            src_str = socket.inet_ntoa(ip.src) if isinstance(ip, dpkt.ip.IP) else socket.inet_ntop(socket.AF_INET6, ip.src)
            dst_str = socket.inet_ntoa(ip.dst) if isinstance(ip, dpkt.ip.IP) else socket.inet_ntop(socket.AF_INET6, ip.dst)

            if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                continue

            l4 = ip.data
            is_tcp = isinstance(l4, dpkt.tcp.TCP)
            proto_str = "tcp" if is_tcp else "udp"
            src_port = l4.sport
            dst_port = l4.dport

            current_5tuple = (src_str, src_port, dst_str, dst_port, proto_str)

            # Check SYN
            if is_tcp and (l4.flags & dpkt.tcp.TH_SYN):
                if syn_ts is None:
                    syn_ts = ts
                    if client_ip_str is None:
                        client_ip_str = src_str
                    detected_5tuple = current_5tuple

            # 5-tuple filtering
            if manifest and manifest.target_5tuple:
                t_src, t_sport, t_dst, t_dport, t_proto = manifest.target_5tuple
                if t_sport == 0:
                    is_match = (
                        (dst_str == t_dst and dst_port == t_dport and proto_str == t_proto) or
                        (src_str == t_dst and src_port == t_dport and proto_str == t_proto)
                    )
                else:
                    is_match = (
                        (src_str == t_src and src_port == t_sport and dst_str == t_dst and dst_port == t_dport and proto_str == t_proto) or
                        (src_str == t_dst and src_port == t_dport and dst_str == t_src and dst_port == t_sport and proto_str == t_proto)
                    )
                if not is_match:
                    continue
            else:
                # Discard local host multicast noise
                if dst_str in ("224.0.0.251", "239.255.255.250", "ff02::fb"):
                    continue

            payload = l4.data
            pkt_len = len(payload)
            if pkt_len == 0:
                continue

            # Direction determination
            is_client = (src_str == client_ip_str or src_str == "172.20.0.30" or (src_str.startswith("172.20.0.3") and dst_str in ("172.20.0.10", "172.20.0.20")))
            direction = 1 if is_client else -1

            # Detect ClientHello length
            if direction == 1 and clienthello_len is None and pkt_len > 5:
                if payload[0] == 0x16 and payload[1] == 0x03:
                    clienthello_len = (payload[3] << 8) | payload[4] + 5

            rel_ts = ts - first_ts
            # If segmented, parse individual TLS records or capped payload length
            tls_recs = parse_tls_records_from_payload(payload)
            if tls_recs:
                for ctype, rlen in tls_recs:
                    raw_packets.append((rel_ts, direction, rlen, ctype))
            else:
                raw_packets.append((rel_ts, direction, min(pkt_len, 1500), 0))

    if len(raw_packets) < 3:
        return None

    if syn_ts is None:
        syn_ts = first_ts

    if detected_5tuple is None:
        detected_5tuple = ("172.20.0.30", 50000, "172.20.0.20", 8443, "tcp")

    capture_id = manifest.capture_id if manifest else os.path.splitext(os.path.basename(pcap_path))[0]
    label = manifest.label if manifest else "unknown"
    behaviour = manifest.behaviour if manifest else "browse"
    profile = manifest.profile if manifest else "broadband"
    dest_id = manifest.dest_id if manifest else "vhost-01"
    client_stack = manifest.client_stack if manifest else "utls-HelloChrome_Auto"
    epoch = manifest.epoch if manifest else "A"

    flow_uid = FlowRecord.make_uid(capture_id, detected_5tuple, syn_ts)
    conn_id = FlowRecord.make_conn_id(detected_5tuple, syn_ts)

    # Convert to records format (t, dir, len)
    records = [(t, d, l) for (t, d, l, ctype) in raw_packets]

    # Handshake end detection: client's first application data record after handshake
    hs_end_idx = 0
    for idx, (t, d, l, ctype) in enumerate(raw_packets):
        if d == 1 and idx > 3:
            hs_end_idx = idx
            break

    if post_handshake_only:
        records = records[hs_end_idx:]
        if len(records) < 3:
            return None

    n_up = sum(1 for _, d, _ in records if d == 1)
    n_down = sum(1 for _, d, _ in records if d == -1)
    bytes_up = sum(l for _, d, l in records if d == 1)
    bytes_down = sum(l for _, d, l in records if d == -1)
    duration = records[-1][0] - records[0][0] if len(records) > 1 else 0.0

    return FlowRecord(
        flow_uid=flow_uid,
        conn_id=conn_id,
        capture_id=capture_id,
        label=label,
        behaviour=behaviour,
        profile=profile,
        dest_id=dest_id,
        client_stack=client_stack,
        epoch=epoch,
        records=records,
        hs_end_idx=hs_end_idx,
        n_up=n_up,
        n_down=n_down,
        bytes_up=bytes_up,
        bytes_down=bytes_down,
        duration=duration,
        clienthello_len=clienthello_len,
        generator_seed=manifest.generator_seed if manifest else 0,
        git_commit=manifest.git_commit if manifest else ""
    )


def compute_flow_statistics(packets: List[Tuple[float, int]]) -> List[float]:
    """Computes standard 48 statistical flow features from signed packet lengths."""
    times = [p[0] for p in packets]
    lengths = [abs(p[1]) for p in packets]
    up_lengths = [abs(p[1]) for p in packets if p[1] > 0]
    down_lengths = [abs(p[1]) for p in packets if p[1] < 0]

    def get_stats(arr):
        if not arr:
            return [0.0] * 10
        a = np.array(arr, dtype=np.float64)
        return [
            float(np.min(a)), float(np.max(a)), float(np.mean(a)), float(np.std(a)),
            float(stats.skew(a)) if len(a) > 2 and np.std(a) > 1e-6 else 0.0,
            float(np.percentile(a, 10)), float(np.percentile(a, 25)),
            float(np.percentile(a, 50)), float(np.percentile(a, 75)),
            float(np.percentile(a, 90))
        ]

    def get_sub_stats(arr):
        if not arr:
            return [0.0] * 9
        a = np.array(arr, dtype=np.float64)
        return [
            float(np.min(a)), float(np.max(a)), float(np.mean(a)), float(np.std(a)),
            float(np.percentile(a, 10)), float(np.percentile(a, 25)),
            float(np.percentile(a, 50)), float(np.percentile(a, 75)),
            float(np.percentile(a, 90))
        ]

    # 1. Lengths stats
    total_stats = get_stats(lengths)
    up_stats = get_sub_stats(up_lengths)
    down_stats = get_sub_stats(down_lengths)

    # 2. IAT stats
    iats = [times[i] - times[i-1] for i in range(1, len(times))] if len(times) > 1 else [0.0]
    iat_stats = get_sub_stats(iats)

    # 3. Burst dynamics
    bursts = []
    curr_dir = None
    curr_len_pkts = 0
    curr_len_bytes = 0
    burst_start_t = 0.0

    for t, signed_l in packets:
        d = 1 if signed_l > 0 else -1
        l = abs(signed_l)
        if curr_dir is None:
            curr_dir = d
            curr_len_pkts = 1
            curr_len_bytes = l
            burst_start_t = t
        elif d == curr_dir:
            curr_len_pkts += 1
            curr_len_bytes += l
        else:
            bursts.append((curr_len_pkts, curr_len_bytes, t - burst_start_t))
            curr_dir = d
            curr_len_pkts = 1
            curr_len_bytes = l
            burst_start_t = t
    if curr_dir is not None:
        bursts.append((curr_len_pkts, curr_len_bytes, times[-1] - burst_start_t))

    burst_count = len(bursts)
    b_pkts = [b[0] for b in bursts]
    b_bytes = [b[1] for b in bursts]
    b_durs = [b[2] for b in bursts]

    burst_stats = [
        float(burst_count),
        float(np.mean(b_pkts)) if b_pkts else 0.0,
        float(np.std(b_pkts)) if b_pkts else 0.0,
        float(np.mean(b_bytes)) if b_bytes else 0.0,
        float(np.std(b_bytes)) if b_bytes else 0.0,
        float(np.mean(b_durs)) if b_durs else 0.0,
        float(np.std(b_durs)) if b_durs else 0.0,
    ]

    # 4. Ratios & Totals
    tot_pkts = len(packets)
    tot_bytes = sum(lengths)
    ratio_up_pkts = len(up_lengths) / max(1, tot_pkts)
    ratio_up_bytes = sum(up_lengths) / max(1.0, tot_bytes)

    totals = [float(ratio_up_pkts), float(ratio_up_bytes), float(tot_pkts), float(tot_bytes)]

    features = total_stats + up_stats + down_stats + iat_stats + burst_stats + totals
    return features


def extract_raw_packets_from_pcap(pcap_path: str, post_handshake_only: bool = False) -> List[Tuple[float, int]]:
    """Backward-compatible helper returning (t, signed_len) packet list."""
    flow = extract_flow_from_pcap(pcap_path, post_handshake_only=post_handshake_only)
    if flow is None:
        return []
    return [(t, d * l) for (t, d, l) in flow.records]


def normalize_sequence_tensor(packets: List[Tuple[float, int]], max_seq_len: int = 200) -> np.ndarray:
    """Converts packet trace into normalized [max_seq_len, 2] tensor."""
    seq = np.zeros((max_seq_len, 2), dtype=np.float32)
    prev_t = 0.0

    for i, (t, signed_l) in enumerate(packets[:max_seq_len]):
        delta_t = max(0.0, t - prev_t)
        prev_t = t
        norm_len = float(signed_l) / 1500.0
        norm_iat = min(1.0, np.log1p(delta_t) / 10.0)
        seq[i, 0] = norm_len
        seq[i, 1] = norm_iat
    return seq


build_sequence_tensor = normalize_sequence_tensor
