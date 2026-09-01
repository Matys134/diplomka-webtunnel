#!/usr/bin/env python3
"""G1 (stack parity) and G6 (provenance).

Both operate on v2 artefacts -- CaptureManifest sidecars and FlowRecord objects -- so they
cannot run against the v1 dataset. They are written now so the collector rewrite has a
target to satisfy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, List


def stack_parity(flows: Iterable, verbose: bool = True) -> dict:
    """G1: the ClientHello must be indistinguishable across classes.

    v1 measurement (F-03): webtunnel 267 B / 19 ciphers / no ALPN / no GREASE (stock Go
    crypto/tls) vs negatives 517 B / 31-43 ciphers. A perfect discriminator.
    """
    by_class = defaultdict(Counter)
    ja4_by_class = defaultdict(set)
    for f in flows:
        if getattr(f, "clienthello_len", None) is not None:
            by_class[f.label][f.clienthello_len] += 1
        if getattr(f, "ja4", None):
            ja4_by_class[f.label].add(f.ja4)

    lengths = {c: set(ctr) for c, ctr in by_class.items()}
    all_lengths = set().union(*lengths.values()) if lengths else set()
    all_ja4 = set().union(*ja4_by_class.values()) if ja4_by_class else set()

    passed = len(all_lengths) <= 1 and len(all_ja4) <= 1
    if verbose:
        for c in sorted(by_class):
            print(f"    {c:<22} ClientHello lengths: {sorted(lengths[c])}  "
                  f"JA4: {sorted(ja4_by_class.get(c, []))}")
        print(f"  distinct ClientHello lengths across all classes: {sorted(all_lengths)}")
        print(f"  -> {'PASS' if passed else 'FAIL'}")
    return {"lengths_by_class": {c: sorted(v) for c, v in lengths.items()},
            "distinct_lengths": sorted(all_lengths),
            "distinct_ja4": sorted(all_ja4), "passed": bool(passed)}


def provenance(flows: Iterable, manifests_by_capture: dict, verbose: bool = True) -> dict:
    """G6: every flow matches its manifest; every capture has one; drops are logged."""
    missing_manifest: List[str] = []
    tuple_mismatch: List[str] = []
    n = 0
    for f in flows:
        n += 1
        m = manifests_by_capture.get(f.capture_id)
        if m is None:
            missing_manifest.append(f.flow_uid)
            continue
        if m.label != f.label or m.profile != f.profile or m.dest_id != f.dest_id:
            tuple_mismatch.append(f.flow_uid)

    passed = not missing_manifest and not tuple_mismatch
    if verbose:
        print(f"  flows checked                : {n}")
        print(f"  flows without a manifest     : {len(missing_manifest)}")
        print(f"  flows contradicting manifest : {len(tuple_mismatch)}")
        print(f"  -> {'PASS' if passed else 'FAIL'}")
    return {"n_flows": n, "missing_manifest": len(missing_manifest),
            "mismatch": len(tuple_mismatch), "passed": bool(passed)}
