#!/usr/bin/env python3
"""Independent forensics over the raw PCAP corpus. Reproduces audit findings F-01..F-06.

Nothing here uses project/ code -- that is the point.

    python3 audit/pcap_forensics.py --pcap-dir /path/to/raw_pcap --mode all

Modes:
    ports        F-01  client TCP ports per split -> connection independence
    handshake    F-02  %captures with a ClientHello, SYN counts, contamination, offload
    clienthello  F-03  parsed ClientHello: length, ciphers, ALPN, GREASE, extensions
    histogram    F-06  most frequent TLS record sizes per class/direction

Requires: dpkt, numpy.
"""
from __future__ import annotations

import argparse, collections, glob, os, socket

import dpkt
import numpy as np

CLASSES = ["webtunnel", "direct_web_browsing", "websocket_ticker",
           "websocket_chat", "video_streaming", "web_assets"]
PROFILES = ["broadband", "lte", "lossy"]
CLIENT = "172.20.0.30"
WT_SRV = "172.20.0.10"
LEG_SRV = "172.20.0.20"


def ip2s(b):
    try:
        return socket.inet_ntop(socket.AF_INET, b)
    except Exception:
        return socket.inet_ntop(socket.AF_INET6, b)


def iter_tcp(path):
    """Yield (ts, src, dst, sport, dport, flags, payload) for every TCP packet."""
    with open(path, "rb") as f:
        try:
            rdr = dpkt.pcap.Reader(f)
        except Exception:
            return
        for ts, buf in rdr:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                continue
            L4 = ip.data
            if not isinstance(L4, dpkt.tcp.TCP):
                continue
            yield (ts, ip2s(ip.src), ip2s(ip.dst), L4.sport, L4.dport,
                   L4.flags, bytes(L4.data))


def parse_clienthello(pl):
    try:
        rec_len = int.from_bytes(pl[3:5], "big")
        hs = pl[5:5 + rec_len]
        if hs[0] != 0x01:
            return None
        i = 4 + 2 + 32
        i += 1 + hs[i]                                     # session id
        cs = int.from_bytes(hs[i:i + 2], "big"); i += 2 + cs
        i += 1 + hs[i]                                     # compression methods
        ext_len = int.from_bytes(hs[i:i + 2], "big"); i += 2
        end, alpn, exts = i + ext_len, [], []
        while i + 4 <= end:
            et = int.from_bytes(hs[i:i + 2], "big")
            el = int.from_bytes(hs[i + 2:i + 4], "big")
            body = hs[i + 4:i + 4 + el]
            i += 4 + el
            exts.append(et)
            if et == 16:
                j = 2
                while j < len(body):
                    L = body[j]
                    alpn.append(body[j + 1:j + 1 + L].decode(errors="replace"))
                    j += 1 + L
        return dict(record_len=len(pl), n_ciphers=cs // 2, alpn=alpn,
                    n_ext=len(exts), ext_ids=exts,
                    grease=any((e & 0x0f0f) == 0x0a0a for e in exts))
    except Exception:
        return None


def mode_ports(pd):
    print("=== F-01 :: client TCP ports on the bridge connection, by split ===")
    for prof in PROFILES:
        tr, te = set(), set()
        for i in range(1, 351, 7):
            for _, s, d, sp, dp, _, _ in iter_tcp(os.path.join(pd, f"webtunnel_{prof}_{i:04d}.pcap")):
                if d == WT_SRV: tr.add(sp)
                elif s == WT_SRV: tr.add(dp)
        for i in range(426, 501, 2):
            for _, s, d, sp, dp, _, _ in iter_tcp(os.path.join(pd, f"webtunnel_{prof}_{i:04d}.pcap")):
                if d == WT_SRV: te.add(sp)
                elif s == WT_SRV: te.add(dp)
        print(f"  {prof:<10} TRAIN={sorted(tr)}  TEST={sorted(te)}  SHARED={sorted(tr & te)}")
    print("  -> a shared port means the SAME TCP connection is in train and test.")


def mode_handshake(pd, n=40):
    print("=== F-02/F-04/F-05 :: handshake presence, contamination, offload ===")
    print(f"  {'class':<22}{'profile':<11}{'%CH':>7}{'SYNs':>7}{'%wt-contam':>12}{'%>1500B':>9}{'dur_s':>9}")
    for c in CLASSES:
        for p in PROFILES:
            ch = syn = contam = 0; tot = 0; over = []; durs = []
            for i in range(1, 501, max(1, 500 // n)):
                fp = os.path.join(pd, f"{c}_{p}_{i:04d}.pcap")
                if not os.path.exists(fp):
                    continue
                tot += 1
                has_ch = False; s_ct = 0; wt = 0; lens = []; ts = []
                for t, s, d, sp, dp, fl, pay in iter_tcp(fp):
                    ts.append(t)
                    if fl & 0x02: s_ct += 1
                    if WT_SRV in (s, d) and 443 in (sp, dp): wt += 1
                    if pay:
                        lens.append(len(pay))
                        if len(pay) > 10 and pay[0] == 0x16 and pay[1] == 0x03 and pay[5] == 0x01:
                            has_ch = True
                ch += has_ch; syn += s_ct; contam += (wt > 0 and c != "webtunnel")
                L = np.array(lens) if lens else np.array([0])
                over.append(float((L > 1500).mean()))
                durs.append(ts[-1] - ts[0] if len(ts) > 1 else 0.0)
            if tot:
                print(f"  {c:<22}{p:<11}{ch/tot*100:6.1f}%{syn/tot:7.2f}"
                      f"{contam/tot*100:11.1f}%{np.mean(over)*100:8.1f}%{np.mean(durs):9.2f}")


def mode_clienthello(pd):
    print("=== F-03 :: ClientHello fingerprints ===")
    def first_ch(fp, want):
        for _, s, d, sp, dp, _, pay in iter_tcp(fp):
            if len(pay) > 10 and pay[0] == 0x16 and pay[1] == 0x03 and pay[5] == 0x01:
                if d != want:
                    continue
                return parse_clienthello(pay)
        return None

    found = 0
    for prof in ["lte", "lossy", "broadband"]:
        for fp in sorted(glob.glob(os.path.join(pd, f"webtunnel_{prof}_*.pcap"))):
            r = first_ch(fp, WT_SRV)
            if r:
                print(f"  webtunnel  {os.path.basename(fp):<32} len={r['record_len']} "
                      f"ciphers={r['n_ciphers']} alpn={r['alpn']} grease={r['grease']} "
                      f"exts={r['ext_ids']}")
                found += 1
                break
        if found >= 2:
            break
    for c in ["direct_web_browsing", "websocket_ticker", "video_streaming"]:
        for fp in sorted(glob.glob(os.path.join(pd, f"{c}_broadband_00*.pcap")))[:3]:
            r = first_ch(fp, LEG_SRV)
            if r:
                print(f"  {c:<20} {os.path.basename(fp):<32} len={r['record_len']} "
                      f"ciphers={r['n_ciphers']} alpn={r['alpn']} grease={r['grease']}")
                break
    print("  -> 267 B / no ALPN / no GREASE is the Go crypto/tls default, NOT uTLS Chrome.")


def mode_histogram(pd, n=100):
    print("=== F-06 :: most frequent TLS record sizes (upstream) ===")
    for c, srv in ((("webtunnel", WT_SRV)), ("websocket_ticker", LEG_SRV),
                   ("websocket_chat", LEG_SRV), ("direct_web_browsing", LEG_SRV)):
        U = collections.Counter()
        for fp in sorted(glob.glob(os.path.join(pd, f"{c}_broadband_0*.pcap")))[:n]:
            for _, s, d, sp, dp, _, pay in iter_tcp(fp):
                if pay and d == srv:
                    U[len(pay)] += 1
        tot = sum(U.values()) or 1
        print(f"  {c:<22} {U.most_common(8)}")
        if c == "webtunnel":
            print(f"    share of exactly 558 B: {U[558]/tot*100:.1f}%   (n={tot})")
            print("    558 = 5 (TLS hdr) + 536 (514 B Tor cell + 22 B framing) + 1 + 16 (AEAD)")
            print(f"    2x multiple 1072 B: {U[1072]} occurrences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", required=True)
    ap.add_argument("--mode", default="all",
                    choices=["all", "ports", "handshake", "clienthello", "histogram"])
    a = ap.parse_args()
    pd_ = a.pcap_dir
    if a.mode in ("all", "ports"): mode_ports(pd_); print()
    if a.mode in ("all", "handshake"): mode_handshake(pd_); print()
    if a.mode in ("all", "clienthello"): mode_clienthello(pd_); print()
    if a.mode in ("all", "histogram"): mode_histogram(pd_)


if __name__ == "__main__":
    main()
