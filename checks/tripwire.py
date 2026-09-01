#!/usr/bin/env python3
"""G2 -- the leakage tripwire.

Fits a depth-1 decision stump on EACH feature individually and reports its test AUC. If any
single feature separates the classes better than TRIPWIRE_AUC_LIMIT, the dataset is either
carrying a laboratory artefact or a genuine protocol invariant -- and you have to say which.

This is eleven lines of real logic. Run after every dataset build. In v1 it would have caught
the entire problem in week one instead of month eight:

    up_len_p25   98.97%   AUC 0.9868    <- Tor cell quantization (real, but see F-06)
    down_len_min 98.58%   AUC 0.9635    <- the 80-byte ChangeCipherSpec record (artefact, F-02)

The gate does NOT ban strong features. It forces them into
checks/expected_invariants.py with a written protocol derivation. That registry becomes a
table in the thesis, and it is the answer to the hardest defence question.

Usage:
    python3 project/checks/tripwire.py --dataset project/data/processed/tabular_dataset.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expected_invariants import (  # noqa: E402
    EXPECTED_INVARIANTS,
    KNOWN_ARTEFACTS,
    TRIPWIRE_AUC_LIMIT,
)


def stump_scan(X_train, y_train, X_test, y_test, feature_names, seed: int = 0):
    """Return [(feature, test_accuracy, test_auc, threshold)] sorted by |AUC - 0.5|."""
    rows = []
    for j, name in enumerate(feature_names):
        clf = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X_train[:, [j]], y_train)
        p = clf.predict_proba(X_test[:, [j]])[:, 1]
        acc = float(accuracy_score(y_test, (p >= 0.5).astype(int)))
        try:
            auc = float(roc_auc_score(y_test, p))
        except ValueError:
            auc = 0.5
        thr = float(clf.tree_.threshold[0])
        rows.append((name, acc, max(auc, 1.0 - auc), thr))
    rows.sort(key=lambda r: -r[2])
    return rows


def run(X_train, y_train, X_test, y_test, feature_names,
        limit: float = TRIPWIRE_AUC_LIMIT, top: int = 15, verbose: bool = True):
    rows = stump_scan(X_train, y_train, X_test, y_test, feature_names)
    offenders = [r for r in rows if r[2] > limit]
    unexplained = [r for r in offenders if r[0] not in EXPECTED_INVARIANTS]

    if verbose:
        print(f"  single-feature depth-1 stumps, {len(feature_names)} features, "
              f"limit AUC = {limit}")
        print(f"  {'feature':<24}{'acc':>9}{'AUC':>9}{'thr':>12}   status")
        for name, acc, auc, thr in rows[:top]:
            if auc <= limit:
                status = "ok"
            elif name in EXPECTED_INVARIANTS:
                status = "REGISTERED invariant"
            elif name in KNOWN_ARTEFACTS:
                status = "KNOWN ARTEFACT"
            else:
                status = "UNEXPLAINED"
            print(f"  {name:<24}{acc*100:8.2f}%{auc:9.4f}{thr:12.4g}   {status}")

        if unexplained:
            print()
            print(f"  FAIL -- {len(unexplained)} feature(s) above the limit are not registered:")
            for name, acc, auc, thr in unexplained:
                note = KNOWN_ARTEFACTS.get(name)
                print(f"    * {name} (AUC {auc:.4f})")
                if note:
                    print(f"        known artefact: {note}")
                else:
                    print("        Either add a protocol derivation to "
                          "checks/expected_invariants.py, or fix the testbed.")
        else:
            print("\n  PASS -- every strongly separating feature has a written justification.")

    return {
        "limit": limit,
        "top": [{"feature": n, "accuracy": a, "auc": u, "threshold": t}
                for n, a, u, t in rows[:top]],
        "n_over_limit": len(offenders),
        "unexplained": [r[0] for r in unexplained],
        "passed": len(unexplained) == 0,
    }


def _load(path):
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]] if "feature_names" in d else \
            [f"f{i}" for i in range(d["X_train"].shape[1])]
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"], names


def main():
    ap = argparse.ArgumentParser(description="G2 leakage tripwire")
    ap.add_argument("--dataset", required=True, help="tabular_dataset.npz")
    ap.add_argument("--limit", type=float, default=TRIPWIRE_AUC_LIMIT)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", help="write the result table here")
    a = ap.parse_args()

    Xtr, ytr, Xte, yte, names = _load(a.dataset)
    print(f"=== G2 leakage tripwire :: {os.path.basename(a.dataset)} ===")
    print(f"  train {Xtr.shape}  test {Xte.shape}  "
          f"positives(test) {int(yte.sum())}  negatives(test) {int((yte == 0).sum())}")
    res = run(Xtr, ytr, Xte, yte, names, limit=a.limit, top=a.top)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  wrote {a.json}")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
