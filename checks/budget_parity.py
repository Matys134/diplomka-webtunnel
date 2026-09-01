#!/usr/bin/env python3
"""G4 -- session-budget parity.

Volume and duration must not be class-informative. In v2 the collector draws a
(duration, bytes_up, bytes_down) budget from a distribution SHARED by every class and each
generator must hit it, so these features carry no label information by construction.

The gate is a two-sample KS test between the positive class and each negative class on
total_bytes, total_pkts and a duration proxy. p > ALPHA for all of them.

In v1 this fails spectacularly: websocket_ticker runs 20.6 s and web_assets 1.18 s purely
because of how the generator scripts are written (F-06).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

ALPHA = 0.01
DEFAULT_FEATURES = ("total_bytes", "total_pkts", "iat_max")


def run(X, y_multi, feature_names, class_names, positive_index=0,
        features=DEFAULT_FEATURES, alpha=ALPHA, verbose=True):
    names = list(feature_names)
    rows, worst = [], 1.0
    pos_mask = y_multi == positive_index

    for feat in features:
        if feat not in names:
            continue
        j = names.index(feat)
        for c in sorted(set(int(v) for v in np.unique(y_multi)) - {positive_index}):
            neg_mask = y_multi == c
            if pos_mask.sum() < 5 or neg_mask.sum() < 5:
                continue
            p = float(stats.ks_2samp(X[pos_mask, j], X[neg_mask, j]).pvalue)
            worst = min(worst, p)
            cname = class_names[c] if c < len(class_names) else str(c)
            rows.append((feat, cname, p))

    passed = worst > alpha
    if verbose:
        print(f"  KS test, webtunnel vs each negative class (need p > {alpha}):")
        for feat, cname, p in sorted(rows, key=lambda r: r[2])[:10]:
            flag = "ok  " if p > alpha else "FAIL"
            print(f"    {flag} {feat:<14} vs {cname:<22} p = {p:.3e}")
        if len(rows) > 10:
            print(f"    ... {len(rows) - 10} more comparisons")
        print(f"  worst p = {worst:.3e}  ->  {'PASS' if passed else 'FAIL'}")
    return {"worst_p": worst, "alpha": alpha, "n_comparisons": len(rows), "passed": bool(passed)}
