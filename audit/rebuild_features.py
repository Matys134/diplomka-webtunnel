#!/usr/bin/env python3
"""Rebuild the 48-feature matrix from raw PCAPs in four extraction variants (audit F-09).

    V0  all packets merged            -- replicates the v1 pipeline
    V1  single target TCP flow only   -- proper 5-tuple demultiplexing
    V2  V1 + first 10 payload packets dropped   -- crude but real post-handshake
    V3  V2 + segmentation offload undone (>MSS split into 1448 B chunks)

All four give 100.00% accuracy and AUC 1.0000, which is the point: the separation is created
by the experimental design, not by the packet representation.

    python3 audit/rebuild_features.py --pcap-dir /path/to/raw_pcap \
                                      --out audit/out/features_variants.npz

Runs single-threaded over 9,000 files in roughly 5-8 minutes. Use --limit to sample.
Requires: dpkt, numpy.
"""
from __future__ import annotations

import argparse, glob, os, socket

import dpkt
import numpy as np

CLASSES = ["webtunnel", "direct_web_browsing", "websocket_ticker",
           "websocket_chat", "video_streaming", "web_assets"]
PROFILES = ["broadband", "lte", "lossy"]
CLIENT, WT_SRV, LEG_SRV, MSS = "172.20.0.30", "172.20.0.10", "172.20.0.20", 1448

FEATURE_NAMES = [
    "len_min","len_max","len_mean","len_std","len_skew","len_p10","len_p25","len_p50","len_p75","len_p90",
    "up_len_min","up_len_max","up_len_mean","up_len_std","up_len_p10","up_len_p25","up_len_p50","up_len_p75","up_len_p90",
    "down_len_min","down_len_max","down_len_mean","down_len_std","down_len_p10","down_len_p25","down_len_p50","down_len_p75","down_len_p90",
    "iat_min","iat_max","iat_mean","iat_std","iat_p10","iat_p25","iat_p50","iat_p75","iat_p90",
    "burst_count","burst_len_mean_pkts","burst_len_std_pkts","burst_len_mean_bytes","burst_len_std_bytes","burst_dur_mean","burst_dur_std",
    "ratio_up_pkts","ratio_up_bytes","total_pkts","total_bytes"]


def ip2s(b):
    try: return socket.inet_ntop(socket.AF_INET, b)
    except Exception: return socket.inet_ntop(socket.AF_INET6, b)


def parse(path):
    flows = {}
    with open(path, "rb") as f:
        try: rdr = dpkt.pcap.Reader(f)
        except Exception: return {}
        for ts, buf in rdr:
            try: eth = dpkt.ethernet.Ethernet(buf)
            except Exception: continue
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP): continue
            L4 = ip.data
            if not isinstance(L4, dpkt.tcp.TCP): continue
            pl = bytes(L4.data)
            if not pl: continue
            s, d = ip2s(ip.src), ip2s(ip.dst)
            if s == CLIENT:   key, direc = (d, L4.dport, L4.sport), 1
            elif d == CLIENT: key, direc = (s, L4.sport, L4.dport), -1
            else: continue
            flows.setdefault(key, []).append((ts, direc, len(pl)))
    return flows


def stats48(pk):
    if len(pk) < 2: return None
    t = np.array([p[0] for p in pk], dtype=np.float64)
    sl = np.array([p[1] * p[2] for p in pk], dtype=np.float64)
    al = np.abs(sl); up, dn = al[sl > 0], al[sl < 0]
    nu, nd, nt = len(up), len(dn), len(al)
    upa = up if nu else np.array([0.]); dna = dn if nd else np.array([0.])
    iats = np.diff(t)
    if len(iats) == 0: iats = np.array([0.])
    bp, bb, bd = [], [], []
    cd, cp, cb, cs = np.sign(sl[0]), 1, al[0], t[0]
    for i in range(1, len(sl)):
        d = np.sign(sl[i])
        if d == cd: cp += 1; cb += al[i]
        else:
            bp.append(cp); bb.append(cb); bd.append(t[i-1]-cs); cd, cp, cb, cs = d, 1, al[i], t[i]
    bp.append(cp); bb.append(cb); bd.append(t[-1]-cs)
    bp, bb, bd = np.array(bp), np.array(bb), np.array(bd)
    def sk(a):
        if len(a) < 3 or a.std() < 1e-5: return 0.0
        return float((((a - a.mean()) ** 3).mean()) / (a.std() ** 3))
    f = [al.min(), al.max(), al.mean(), al.std(), sk(al)] + [np.percentile(al, q) for q in (10,25,50,75,90)]
    f += [upa.min() if nu else 0, upa.max() if nu else 0, upa.mean() if nu else 0, upa.std() if nu else 0]
    f += [np.percentile(upa, q) if nu else 0 for q in (10,25,50,75,90)]
    f += [dna.min() if nd else 0, dna.max() if nd else 0, dna.mean() if nd else 0, dna.std() if nd else 0]
    f += [np.percentile(dna, q) if nd else 0 for q in (10,25,50,75,90)]
    f += [iats.min(), iats.max(), iats.mean(), iats.std()] + [np.percentile(iats, q) for q in (10,25,50,75,90)]
    f += [len(bp), bp.mean(), bp.std(), bb.mean(), bb.std(), bd.mean(), bd.std()]
    f += [nu / max(nt, 1), up.sum() / max(al.sum(), 1.0), nt, al.sum()]
    return np.array(f, dtype=np.float32)


def segs(pk, split_mtu=False, skip=0):
    out = []
    for ts, direc, L in pk:
        if split_mtu and L > MSS:
            rem = L
            while rem > 0:
                c = min(MSS, rem); out.append((ts, direc, c)); rem -= c
        else:
            out.append((ts, direc, min(L, 1500)))
    return out[skip:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", required=True)
    ap.add_argument("--out", default="audit/out/features_variants.npz")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.pcap_dir, "*.pcap")))
    if a.limit: files = files[:a.limit]
    rows = {k: [] for k in ("V0", "V1", "V2", "V3")}
    meta = []

    for n, fp in enumerate(files):
        base = os.path.basename(fp)
        cls = next((c for c in CLASSES if base.startswith(c)), None)
        prof = next((p for p in PROFILES if f"_{p}_" in base), None)
        if not cls or not prof: continue
        try: sid = int(os.path.splitext(base)[0].split("_")[-1])
        except Exception: continue
        fl = parse(fp)
        if not fl: continue
        srv = WT_SRV if cls == "webtunnel" else LEG_SRV
        cand = {k: v for k, v in fl.items() if k[0] == srv}
        if not cand: continue
        key = max(cand, key=lambda k: sum(x[2] for x in cand[k]))
        pk = sorted(cand[key], key=lambda x: x[0])
        allpk = sorted([x for v in fl.values() for x in v], key=lambda x: x[0])
        variants = {"V0": segs(allpk), "V1": segs(pk),
                    "V2": segs(pk, skip=10), "V3": segs(pk, split_mtu=True, skip=10)}
        tmp = {}
        for name, v in variants.items():
            st = stats48(v) if len(v) >= 5 else None
            tmp[name] = st if st is not None else np.full(48, np.nan, dtype=np.float32)
        if bool(np.isnan(tmp["V1"]).any()): continue
        for k in rows: rows[k].append(tmp[k])
        meta.append((cls, prof, sid))
        if n % 1000 == 0:
            print(f"  {n}/{len(files)} files, {len(meta)} flows", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out,
        V0=np.array(rows["V0"]), V1=np.array(rows["V1"]),
        V2=np.array(rows["V2"]), V3=np.array(rows["V3"]),
        cls=np.array([m[0] for m in meta]), prof=np.array([m[1] for m in meta]),
        sid=np.array([m[2] for m in meta]), feature_names=np.array(FEATURE_NAMES))
    print(f"DONE  {len(meta)} flows -> {a.out}")


if __name__ == "__main__":
    main()
