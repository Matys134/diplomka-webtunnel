#!/usr/bin/env python3
"""G1 (stack parity) and G6 (provenance) -- rewritten so they can actually fail.

v2.0's versions (audit V-01):
  * G1 compared `len(set(clienthello_len)) <= 1` and `len(set(ja4)) <= 1`. `ja4` was never
    populated by the flow builder, so the JA4 clause was `len(empty) <= 1` -- always true. And a
    class contributing NO ClientHello was silently skipped, so the gate would have passed by
    comparing negatives to negatives.
  * G6 compared `manifest.label == flow.label` (and profile, and dest_id) -- but the flow builder
    COPIES those three fields out of the manifest, so the gate compared the manifest to itself.
    It never checked the 5-tuple its own docstring promised, never checked capture coverage, and
    never logged drop reasons.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

MIN_HELLOS_PER_CLASS = 25


def stack_parity(flows: Iterable, min_hellos: int = MIN_HELLOS_PER_CLASS,
                 verbose: bool = True) -> dict:
    """G1: the ClientHello must be indistinguishable across classes.

    Three assertions, all falsifiable:
      a) every class present in the corpus contributes at least `min_hellos` ClientHellos
         -- a class with none cannot be compared, and silence is not parity;
      b) exactly one JA4 fingerprint across all classes -- this is what a censor matches on;
      c) no class's ClientHello-length set is disjoint from every other class's -- v2.0 had
         webtunnel at {267} against negatives at {503..602}, a perfect discriminator.
    """
    flows = list(flows)
    classes = sorted({f.label for f in flows})
    lens: Dict[str, Counter] = defaultdict(Counter)
    ja4s: Dict[str, set] = defaultdict(set)
    alpns: Dict[str, set] = defaultdict(set)
    n_hellos: Counter = Counter()

    for f in flows:
        if getattr(f, "clienthello_len", None):
            lens[f.label][f.clienthello_len] += 1
            n_hellos[f.label] += 1
        if getattr(f, "ja4", None):
            ja4s[f.label].add(f.ja4)
        if getattr(f, "alpn_offered", None):
            alpns[f.label].add(tuple(f.alpn_offered))

    silent = [c for c in classes if n_hellos[c] < min_hellos]
    all_ja4 = set().union(*ja4s.values()) if ja4s else set()
    all_alpn = set().union(*alpns.values()) if alpns else set()

    disjoint: List[str] = []
    for c in classes:
        mine = set(lens[c])
        others = set().union(*[set(lens[o]) for o in classes if o != c]) if len(classes) > 1 else set()
        if mine and others and not (mine & others):
            disjoint.append(c)

    passed = (not silent) and len(all_ja4) == 1 and len(all_alpn) == 1 and not disjoint

    if verbose:
        for c in classes:
            print(f"    {c:<22} hellos={n_hellos[c]:5d}  lens={sorted(lens[c])[:6]}"
                  f"{'...' if len(lens[c]) > 6 else ''}")
            print(f"    {'':<22} ja4={sorted(ja4s.get(c, [])) or '(none)'}")
            print(f"    {'':<22} alpn={sorted(alpns.get(c, [])) or '(none)'}")
        if silent:
            print(f"  FAIL -- classes contributing < {min_hellos} ClientHellos: {silent}")
            print("         a class with no handshake cannot be compared; that IS the leak (F-02).")
        if len(all_ja4) != 1:
            print(f"  FAIL -- {len(all_ja4)} distinct JA4 fingerprints: {sorted(all_ja4)}")
        if len(all_alpn) != 1:
            print(f"  FAIL -- {len(all_alpn)} distinct ALPN offers: {sorted(all_alpn)}")
        if disjoint:
            print(f"  FAIL -- ClientHello length sets disjoint from all other classes: {disjoint}")
        print(f"  -> {'PASS' if passed else 'FAIL'}")

    return {
        "classes": classes,
        "hellos_per_class": dict(n_hellos),
        "silent_classes": silent,
        "distinct_ja4": sorted(all_ja4),
        "distinct_alpn": [list(a) for a in sorted(all_alpn)],
        "length_disjoint_classes": disjoint,
        "lengths_by_class": {c: sorted(v) for c, v in lens.items()},
        "passed": bool(passed),
    }


def provenance(flows: Iterable, manifests_by_capture: Dict[str, object],
               n_captures_on_disk: Optional[int] = None,
               attrition: Optional[Dict[str, Dict[str, int]]] = None,
               verbose: bool = True) -> dict:
    """G6: ground truth is recorded, authoritative, and matches the wire.

      a) every flow has a manifest;
      b) every manifest is AUTHORITATIVE (written by the collector) -- reconstructing a 5-tuple
         from the capture you are trying to validate is circular, so `repaired-legacy` and
         `legacy-pre-v2.1` fail here on purpose;
      c) every flow's observed socket_id equals the socket_id implied by its manifest 5-tuple;
      d) every capture on disk is accounted for: it became a flow, or it has a logged reason.
    """
    flows = list(flows)
    missing, non_auth, tuple_mismatch = [], [], []
    prov_counts: Counter = Counter()

    for f in flows:
        m = manifests_by_capture.get(f.capture_id)
        if m is None:
            missing.append(f.capture_id)
            continue
        prov_counts[getattr(m, "provenance", "?")] += 1
        if not getattr(m, "is_authoritative", False):
            non_auth.append(f.capture_id)
        ft = getattr(m, "target_5tuple", None)
        if ft:
            expected = f"{ft[0]}:{ft[1]}->{ft[2]}:{ft[3]}/{ft[4]}"
            if expected != f.socket_id:
                tuple_mismatch.append((f.capture_id, expected, f.socket_id))

    dropped = sum(sum(v.values()) for v in (attrition or {}).values())
    accounted = len(flows) + dropped
    unaccounted = (n_captures_on_disk - accounted) if n_captures_on_disk is not None else 0

    passed = (not missing and not non_auth and not tuple_mismatch and unaccounted == 0)

    if verbose:
        print(f"  flows checked                 : {len(flows)}")
        print(f"  flows without a manifest      : {len(missing)}")
        print(f"  flows with 5-tuple mismatch   : {len(tuple_mismatch)}")
        print(f"  manifest provenance           : {dict(prov_counts)}")
        print(f"  flows with NON-authoritative provenance: {len(non_auth)}")
        if n_captures_on_disk is not None:
            print(f"  captures on disk              : {n_captures_on_disk} "
                  f"({len(flows)} flows + {dropped} logged drops = {accounted})")
        if attrition:
            print("  drop reasons by class:")
            for cls in sorted(attrition):
                for reason, n in sorted(attrition[cls].items(), key=lambda kv: -kv[1]):
                    print(f"      {cls:<24}{reason:<36}{n:5d}")
        if non_auth:
            print(f"  FAIL -- {len(non_auth)} flow(s) rest on reconstructed ground truth.")
            print("         Only provenance='collector' is admissible (audit V-02/V-03).")
        if unaccounted:
            print(f"  FAIL -- {unaccounted} capture(s) neither became a flow nor were logged.")
        print(f"  -> {'PASS' if passed else 'FAIL'}")

    return {
        "n_flows": len(flows),
        "missing_manifest": len(missing),
        "non_authoritative": len(non_auth),
        "tuple_mismatch": len(tuple_mismatch),
        "tuple_mismatch_examples": tuple_mismatch[:5],
        "provenance_counts": dict(prov_counts),
        "captures_on_disk": n_captures_on_disk,
        "unaccounted": unaccounted,
        "passed": bool(passed),
    }
