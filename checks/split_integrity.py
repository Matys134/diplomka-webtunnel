#!/usr/bin/env python3
"""G5 -- split integrity, with the assertion that can actually fail.

v2.0 (audit V-01) called `run_v1_diagnostic()`, which checked
    sample_ids_all == concat(sample_ids_train, sample_ids_val, sample_ids_test)
-- an identity `build_dataset.py` constructs literally, so the gate passed by construction and
never touched `conn_ids_*`, which were sitting unused in the same .npz.

v2.1 asserts four things:
  a) no conn_id in two splits;
  b) no SOCKET in two splits -- the one that matters, because two windows of one long-lived
     socket carry different conn_ids. Client port 56446 spanned train/val/test in v2.0;
  c) the group vector handed to StratifiedGroupKFold is element-wise aligned with X (F-11a);
  d) the corpus has enough independent positive sockets to support a connection-level claim.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence

import numpy as np

MIN_POSITIVE_SOCKETS = 200          # below this a connection-level TPR interval is meaningless


def check_disjoint(split_to_keys: Dict[str, Sequence[str]], key_name: str = "key") -> dict:
    seen: Dict[str, str] = {}
    clashes = []
    for name, keys in split_to_keys.items():
        for k in keys:
            if k in seen and seen[k] != name:
                clashes.append((k, seen[k], name))
            seen[k] = name
    uniq = {c[0] for c in clashes}
    return {"key": key_name, "n_clashing_keys": len(uniq), "n_clashes": len(clashes),
            "examples": clashes[:5], "passed": not clashes}


def check_alignment(row_keys: Sequence, groups: Sequence) -> dict:
    if len(row_keys) != len(groups):
        return {"passed": False, "reason": "length mismatch",
                "n_rows": len(row_keys), "n_groups": len(groups)}
    bad = int(np.sum(np.asarray(row_keys) != np.asarray(groups)))
    return {"passed": bad == 0, "n_rows": len(row_keys), "n_mismatched": bad,
            "fraction_correct": 1.0 - bad / max(1, len(row_keys))}


def run(npz, verbose: bool = True) -> dict:
    out: Dict[str, object] = {"mode": "v2.1"}
    have = set(npz.files)

    need = {"socket_ids_train", "socket_ids_val", "socket_ids_test",
            "conn_ids_train", "conn_ids_val", "conn_ids_test"}
    if not need.issubset(have):
        out.update(passed=False, reason=f"dataset lacks {sorted(need - have)}; "
                                        "rebuild with 2_data_pipeline/build_dataset.py v2.1")
        if verbose:
            print(f"  FAIL -- {out['reason']}")
        return out

    socks = {s: npz[f"socket_ids_{s}"] for s in ("train", "val", "test")}
    conns = {s: npz[f"conn_ids_{s}"] for s in ("train", "val", "test")}
    a = check_disjoint(conns, "conn_id")
    b = check_disjoint(socks, "socket_id")

    # (c) alignment: socket_ids_all must equal concat(train, val, test) ELEMENT-WISE, and the
    # per-row key vector must equal the CV group vector actually used.
    cat = np.concatenate([socks[s] for s in ("train", "val", "test")])
    c = check_alignment(npz["socket_ids_all"], cat) if "socket_ids_all" in have else \
        {"passed": False, "reason": "no socket_ids_all"}

    # (d) independent positive sockets
    y = np.concatenate([npz["y_train"], npz["y_val"], npz["y_test"]])
    all_socks = npz["socket_ids_all"] if "socket_ids_all" in have else cat
    pos_socks = len(set(all_socks[y == 1]))
    biggest = Counter(all_socks[y == 1]).most_common(1)
    top_share = (biggest[0][1] / max(1, int(y.sum()))) if biggest else 0.0
    # Herfindahl effective number of independent positive sockets
    counts = np.asarray(list(Counter(all_socks[y == 1]).values()), dtype=float)
    eff_n = float(1.0 / np.sum((counts / counts.sum()) ** 2)) if counts.size else 0.0
    d_ok = pos_socks >= MIN_POSITIVE_SOCKETS and top_share <= 0.05

    passed = a["passed"] and b["passed"] and c.get("passed", False) and d_ok
    out.update(conn_disjoint=a, socket_disjoint=b, alignment=c,
               positive_sockets=pos_socks, effective_positive_sockets=round(eff_n, 2),
               largest_socket_share=round(top_share, 4), sufficiency_passed=bool(d_ok),
               passed=bool(passed))

    if verbose:
        a_msg = "PASS" if a["passed"] else "FAIL ({} conn_ids in 2 splits)".format(a["n_clashing_keys"])
        b_msg = "PASS" if b["passed"] else "FAIL ({} sockets in 2 splits)".format(b["n_clashing_keys"])
        print(f"  a) conn_id disjoint across splits      : {a_msg}")
        print(f"  b) SOCKET disjoint across splits       : {b_msg}")
        for k, s1, s2 in b["examples"]:
            print(f"       {k}  in {s1} and {s2}")
        print(f"  c) group vector aligned with X         : "
              f"{'PASS' if c.get('passed') else 'FAIL'}"
              f"{'' if c.get('passed') else '  ' + str(c.get('reason', c.get('n_mismatched')))}")
        print(f"  d) independent positive sockets        : {pos_socks} "
              f"(effective {eff_n:.1f}, largest carries {100*top_share:.1f}% of positives) "
              f"-- need >= {MIN_POSITIVE_SOCKETS} and <= 5% "
              f"{'PASS' if d_ok else 'FAIL'}")
        if not d_ok:
            print("       FAIL -- this is F-01. A connection-level confidence interval computed "
                  "from this many independent sockets is not meaningful.")
        print(f"  -> {'PASS' if passed else 'FAIL'}")
    return out


# kept so old callers do not break; it now delegates to the real check
def run_v1_diagnostic(npz, verbose: bool = True) -> dict:
    return run(npz, verbose=verbose)
