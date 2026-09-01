#!/usr/bin/env python3
"""
DIAGNOSTIC ONLY -- reconstruct 5-tuples for the pre-v2.1 pilot captures.

This exists so the rewritten pipeline (reassembly, strict demux, gates, lattice rule) can be
exercised end-to-end on the 2,016 captures that already exist, WITHOUT pretending that the
result is admissible evidence.

Every manifest it writes is stamped `provenance = "repaired-legacy"`, and gate G6 FAILS on any
corpus containing one.  That is deliberate: reconstructing ground truth from the artefact you
are trying to validate is exactly the circularity the audit objected to.  The real campaign gets
its 5-tuples from the collector (start_tor.sh / the generator), never from here.

Usage:
    python3 2_data_pipeline/repair_legacy_manifests.py [--dry-run]
"""
import argparse
import glob
import json
import os
import socket
import sys
from collections import defaultdict

import dpkt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from common.contracts import CaptureManifest  # noqa: E402

RAW = os.path.join(PROJECT_ROOT, "data", "raw_pcap")
BRIDGE = ("172.20.0.10", 443)
LEGIT = ("172.20.0.20", 8443)


def sockets_to(pcap_path, server_ip, server_port):
    """-> {client_port: {"bytes": n, "syn": bool, "first_ts": t}}"""
    out = defaultdict(lambda: {"bytes": 0, "syn": False, "first_ts": None})
    try:
        fh = open(pcap_path, "rb")
    except OSError:
        return {}
    with fh:
        try:
            pcap = dpkt.pcap.Reader(fh)
        except Exception:
            return {}
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP) or not isinstance(ip.data, dpkt.tcp.TCP):
                continue
            t = ip.data
            s, d = socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)
            if d == server_ip and t.dport == server_port:
                port, outbound = t.sport, True
            elif s == server_ip and t.sport == server_port:
                port, outbound = t.dport, False
            else:
                continue
            e = out[port]
            e["bytes"] += len(t.data)
            if e["first_ts"] is None:
                e["first_ts"] = ts
            if outbound and (t.flags & dpkt.tcp.TH_SYN) and not (t.flags & dpkt.tcp.TH_ACK):
                e["syn"] = True
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats = defaultdict(lambda: defaultdict(int))
    for pcap in sorted(glob.glob(os.path.join(RAW, "*.pcap"))):
        side = os.path.splitext(pcap)[0] + ".manifest.json"
        if not os.path.exists(side):
            continue
        with open(side, encoding="utf-8") as f:
            raw = json.load(f)
        label = raw.get("label", "")
        srv_ip, srv_port = BRIDGE if label == "webtunnel" else LEGIT

        socks = sockets_to(pcap, srv_ip, srv_port)
        live = {p: v for p, v in socks.items() if v["bytes"] > 0}
        stats[label]["captures"] += 1
        stats[label]["sockets"] += len(live)

        if not live:
            stats[label]["no_socket"] += 1
            raw.update(ok=False, drop_reason="legacy_no_socket", provenance="repaired-legacy")
        else:
            # Dominant socket by payload bytes; prefer one that actually has a client SYN.
            with_syn = {p: v for p, v in live.items() if v["syn"]}
            pool = with_syn or live
            port = max(pool, key=lambda p: pool[p]["bytes"])
            if len(live) > 1:
                stats[label]["multi_socket_capture"] += 1
            if not live[port]["syn"]:
                stats[label]["no_client_syn"] += 1
            raw.update(
                target_5tuple=["172.20.0.30", int(port), srv_ip, int(srv_port), "tcp"],
                client_ip="172.20.0.30",
                provenance="repaired-legacy",
                ok=True,
                drop_reason="",
                budget_id=f"{raw.get('profile','?')}-{raw.get('capture_id','?').split('_')[-1]}",
            )
        if not args.dry_run:
            with open(side, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2)

    print(f"{'class':<24}{'captures':>9}{'sockets':>9}{'multi':>8}{'no_syn':>8}{'no_sock':>9}")
    for label in sorted(stats):
        s = stats[label]
        print(f"{label:<24}{s['captures']:9d}{s['sockets']:9d}"
              f"{s['multi_socket_capture']:8d}{s['no_client_syn']:8d}{s['no_socket']:9d}")
    print("\nAll rewritten manifests are stamped provenance='repaired-legacy'. Gate G6 fails on them.")


if __name__ == "__main__":
    main()
