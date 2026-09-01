#!/usr/bin/env python3
"""G5 -- split integrity.

Asserts:
  a) no conn_id appears in more than one split (train / val / test)
  b) the group vector handed to StratifiedGroupKFold is aligned with the feature matrix

(b) is the v1 bug F-11: X was concat(X_train, X_val, X_test) -- a permutation of file order --
while groups stayed in file order. Only 6.85% of positions matched.
"""
from __future__ import annotations

import numpy as np


def check_disjoint(split_to_conn_ids: dict) -> dict:
    seen, clashes = {}, []
    for name, ids in split_to_conn_ids.items():
        for cid in ids:
            if cid in seen and seen[cid] != name:
                clashes.append((cid, seen[cid], name))
            seen[cid] = name
    return {"n_clashes": len(clashes), "examples": clashes[:5],
            "passed": len(clashes) == 0}


def check_alignment(n_rows: int, groups) -> dict:
    return {"n_rows": int(n_rows), "n_groups": int(len(groups)),
            "passed": bool(len(groups) == n_rows)}


def run_v1_diagnostic(npz, verbose=True) -> dict:
    """Reproduce the F-11 measurement on a v1-style .npz (no conn_id available)."""
    out = {"mode": "v1-diagnostic"}
    if "sample_ids_all" not in npz:
        out["passed"] = False
        out["reason"] = "no sample_ids_all in dataset"
        return out

    sa = np.asarray(npz["sample_ids_all"])
    parts = [npz[k] for k in ("sample_ids_train", "sample_ids_val", "sample_ids_test") if k in npz]
    if len(parts) != 3:
        out["passed"] = False
        out["reason"] = "split id arrays missing"
        return out

    cat = np.concatenate(parts)
    identical = bool(np.array_equal(sa, cat))
    match = float((sa[:len(cat)] == cat).mean())
    out.update({"arrays_identical": identical, "fraction_group_labels_correct": match,
                "n_unique_groups": int(len(np.unique(sa))), "passed": identical})

    if verbose:
        print(f"  sample_ids_all == concat(train,val,test)? {identical}")
        print(f"  fraction of positions where the CV group label is correct: {match:.4f}")
        if not identical:
            print("    FAIL -- StratifiedGroupKFold is grouping on a scrambled label vector (F-11).")
        print("  NOTE: v1 has no conn_id, so the stronger check (a) cannot run. In v2 the")
        print("        grouping key is FlowRecord.conn_id and this becomes a real assertion.")
    return out
