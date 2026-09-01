#!/usr/bin/env python3
"""G4 -- session-budget parity, as a PAIRED test.

The v2.1 collector draws ONE (duration, bytes_up, bytes_down) triple per (sample_id, profile)
and hands the identical triple to every class, tagging it with `budget_id`. So the right test is
paired: for each budget_id, compare the positive flow against each negative flow that was given
the SAME budget.

v2.0 drew a different budget per class from a shared distribution and ran a two-sample KS test.
At n=500 that rejects on any trivial difference (it reported p = 2e-165), and it could not
distinguish "the sampler differs" from "the generator did not honour its budget" -- which is the
only interesting failure.

The gate reports both the paired Wilcoxon test and the median ratio, because a ratio far from 1
with a non-significant p is still a design failure worth seeing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

ALPHA = 0.01
RATIO_TOLERANCE = 2.0            # positive/negative median ratio must sit in [1/2, 2]
DEFAULT_FEATURES = ("total_bytes", "total_pkts", "iat_max")


def run(X, y_multi, feature_names, class_names, positive_index=0,
        budget_ids: Optional[np.ndarray] = None, features=DEFAULT_FEATURES,
        alpha=ALPHA, verbose=True) -> dict:
    names = list(feature_names)
    pos_mask = np.asarray(y_multi) == positive_index
    rows: List[dict] = []
    worst_p, worst_ratio = 1.0, 1.0
    paired = budget_ids is not None and len(set(budget_ids)) > 1

    for feat in features:
        if feat not in names:
            continue
        j = names.index(feat)
        for c in sorted(set(int(v) for v in np.unique(y_multi)) - {positive_index}):
            neg_mask = np.asarray(y_multi) == c
            if pos_mask.sum() < 5 or neg_mask.sum() < 5:
                continue
            cname = class_names[c] if c < len(class_names) else str(c)

            if paired:
                bp: Dict[str, List[float]] = defaultdict(list)
                bn: Dict[str, List[float]] = defaultdict(list)
                for b, v in zip(np.asarray(budget_ids)[pos_mask], X[pos_mask, j]):
                    bp[b].append(float(v))
                for b, v in zip(np.asarray(budget_ids)[neg_mask], X[neg_mask, j]):
                    bn[b].append(float(v))
                common = sorted(set(bp) & set(bn))
                if len(common) < 10:
                    continue
                a = np.asarray([np.median(bp[b]) for b in common])
                bvals = np.asarray([np.median(bn[b]) for b in common])
                try:
                    p = float(stats.wilcoxon(a, bvals, zero_method="zsplit").pvalue)
                except ValueError:
                    p = 1.0
                ratio = float(np.median(a) / max(1e-9, np.median(bvals)))
                test = f"wilcoxon(n_pairs={len(common)})"
            else:
                p = float(stats.ks_2samp(X[pos_mask, j], X[neg_mask, j]).pvalue)
                ratio = float(np.median(X[pos_mask, j]) / max(1e-9, np.median(X[neg_mask, j])))
                test = "ks_2samp(UNPAIRED)"

            worst_p = min(worst_p, p)
            worst_ratio = max(worst_ratio, max(ratio, 1.0 / max(ratio, 1e-9)))
            rows.append({"feature": feat, "vs": cname, "p": p, "ratio": ratio, "test": test})

    ok_p = worst_p > alpha
    ok_ratio = worst_ratio <= RATIO_TOLERANCE
    passed = bool(rows) and ok_p and ok_ratio

    if verbose:
        mode = "PAIRED on budget_id" if paired else "UNPAIRED (no budget_id -- legacy corpus)"
        print(f"  mode: {mode};  need p > {alpha} AND median ratio within "
              f"[{1/RATIO_TOLERANCE:.2f}, {RATIO_TOLERANCE:.1f}]")
        for r in sorted(rows, key=lambda r: r["p"])[:10]:
            flag = "ok  " if (r["p"] > alpha and 1/RATIO_TOLERANCE <= r["ratio"] <= RATIO_TOLERANCE) else "FAIL"
            print(f"    {flag} {r['feature']:<13} vs {r['vs']:<22} "
                  f"p = {r['p']:.3e}   ratio = {r['ratio']:.2f}")
        if len(rows) > 10:
            print(f"    ... {len(rows) - 10} more comparisons")
        print(f"  worst p = {worst_p:.3e}   worst ratio = {worst_ratio:.2f}   "
              f"-> {'PASS' if passed else 'FAIL'}")

    return {"worst_p": worst_p, "worst_ratio": worst_ratio, "alpha": alpha,
            "paired": bool(paired), "n_comparisons": len(rows), "passed": passed}
