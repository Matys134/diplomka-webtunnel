#!/usr/bin/env python3
"""Reproduce audit finding F-10: the countermeasure result is a pipeline artefact, and the
defences provide no protection against an adaptive adversary.

Uses the repository's OWN saved XGBoost model, its OWN test split and its OWN defence code
(re-implemented verbatim below so this script is self-contained), and adds the two controls
the repository never runs:

    (2) pipeline identity  -- recompute features from the UNMODIFIED tensor, no defence at all
    (5) adaptive adversary -- retrain the censor on defended traffic

Expected output (matches docs/03-evidence.md sec. 8):

    (1) original features, saved model  - repo "before"     acc 98.90%  recall 100.00%
    (2) CONTROL: recomputed, UNMODIFIED tensor              acc 86.66%  recall  24.89%
    (3) Mode 1 padding      - repo "after"                  acc 88.63%  recall  36.00%
    (4) Mode 2 coalescing   - repo "after"                  acc 88.87%  recall  37.33%
    adaptive Mode 1 / Mode 2                                acc 100.00% recall 100.00%

Requires: numpy, scipy, scikit-learn, xgboost. No PCAPs needed.

    python3 audit/defense_recheck.py --project project
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats as sps
from sklearn.metrics import (accuracy_score, average_precision_score,
                             recall_score, roc_auc_score)

N_FEATS = 48


# --------------------------------------------------------------------------- #
# verbatim re-implementation of project/2_data_pipeline/sanitizer.py
# --------------------------------------------------------------------------- #
def compute_flow_statistics(packets):
    if len(packets) < 2:
        return np.zeros(N_FEATS, dtype=np.float32)
    t = np.array([p[0] for p in packets], dtype=np.float64)
    sl = np.array([p[1] for p in packets], dtype=np.float64)
    al = np.abs(sl)
    up, dn = al[sl > 0], al[sl < 0]
    nu, nd, nt = len(up), len(dn), len(al)
    upa = up if nu else np.array([0.0])
    dna = dn if nd else np.array([0.0])
    iats = np.diff(t)
    if len(iats) == 0:
        iats = np.array([0.0])

    bp, bb, bd = [], [], []
    cd, cp, cb, cs = np.sign(sl[0]), 1, al[0], t[0]
    for i in range(1, len(sl)):
        d = np.sign(sl[i])
        if d == cd:
            cp += 1
            cb += al[i]
        else:
            bp.append(cp); bb.append(cb); bd.append(t[i - 1] - cs)
            cd, cp, cb, cs = d, 1, al[i], t[i]
    bp.append(cp); bb.append(cb); bd.append(t[-1] - cs)
    bp, bb, bd = np.array(bp), np.array(bb), np.array(bd)

    f = [al.min(), al.max(), al.mean(), al.std(),
         (sps.skew(al) if len(al) > 2 and al.std() > 1e-5 else 0.0)]
    f += [np.percentile(al, q) for q in (10, 25, 50, 75, 90)]
    f += [upa.min() if nu else 0.0, upa.max() if nu else 0.0,
          upa.mean() if nu else 0.0, upa.std() if nu else 0.0]
    f += [np.percentile(upa, q) if nu else 0.0 for q in (10, 25, 50, 75, 90)]
    f += [dna.min() if nd else 0.0, dna.max() if nd else 0.0,
          dna.mean() if nd else 0.0, dna.std() if nd else 0.0]
    f += [np.percentile(dna, q) if nd else 0.0 for q in (10, 25, 50, 75, 90)]
    f += [iats.min(), iats.max(), iats.mean(), iats.std()]
    f += [np.percentile(iats, q) for q in (10, 25, 50, 75, 90)]
    f += [len(bp), bp.mean(), bp.std(), bb.mean(), bb.std(), bd.mean(), bd.std()]
    f += [nu / max(nt, 1), up.sum() / max(al.sum(), 1.0), nt, al.sum()]
    return np.array(f, dtype=np.float32)


def recompute_tab(Xs):
    """project/4_evaluation/evaluate_before_after_defenses.py::recompute_tabular_features

    THIS IS THE ARTEFACT GENERATOR. It truncates the flow to 200 packets and round-trips the
    IAT channel through expm1(x*10) on a clipped value, so features computed this way are not
    comparable with the stored X_test. Delete it in v2.
    """
    out = []
    for i in range(len(Xs)):
        pk, ct = [], 0.0
        for s in range(Xs.shape[1]):
            ct += max(0.0, float(np.expm1(float(Xs[i, s, 1]) * 10.0)))
            pk.append((ct, int(round(Xs[i, s, 0] * 1500.0))))
        if len(pk) < 3:
            pk = [(0.0, 500), (0.05, -500), (0.1, 500)]
        out.append(compute_flow_statistics(pk))
    return np.array(out, dtype=np.float32)


def pad_mode1(Xs, y, rng):
    """Mode 1: adaptive intra-frame padding, 1-128 B."""
    Xd = Xs.copy()
    tot = pad = 0.0
    for i in range(len(y)):
        if y[i] != 1:
            continue
        for s in range(Xd.shape[1]):
            nl, ni = Xd[i, s, 0], Xd[i, s, 1]
            if abs(nl) < 1e-4:
                continue
            ob = abs(nl) * 1500.0
            tot += ob
            fb = min(1480.0, ob + rng.randint(1, 129))
            pad += fb - ob
            Xd[i, s, 0] = (1.0 if nl > 0 else -1.0) * (fb / 1500.0)
            Xd[i, s, 1] = min(1.0, ni + rng.uniform(0.001, 0.03))
    return Xd, pad / max(tot, 1.0) * 100.0


def mode2(Xs, y, rng):
    """Mode 2: cell coalescing into MTU frames + cover traffic shaping."""
    Xd = np.zeros_like(Xs)
    to = td = 0.0
    for i in range(len(y)):
        if y[i] == 0:
            Xd[i] = Xs[i]
            continue
        raw = []
        for s in range(Xs.shape[1]):
            v = Xs[i, s, 0]
            if abs(v) < 1e-4:
                continue
            L = int(round(abs(v) * 1500.0))
            raw.append((1 if v > 0 else -1, L, float(Xs[i, s, 1])))
            to += L
        if not raw:
            continue
        co, cd, cb, ci = [], None, 0, 0.0
        for d, L, it in raw:
            if cd is None:
                cd, cb, ci = d, L, it
            elif cd == d:
                if cb + L <= 1448:
                    cb += L
                    ci += it * 0.5
                else:
                    co.append((cd, cb, ci)); cb, ci = L, it
            else:
                co.append((cd, cb, ci)); cd, cb, ci = d, L, it
        if cb > 0:
            co.append((cd, cb, ci))
        sh = []
        if co:
            sh.append((1, rng.randint(450, 750), 0.0))
            sh.append((-1, rng.randint(600, 1100), rng.uniform(0.01, 0.03)))
        for d, L, it in co:
            if rng.random() < 0.6:
                L = min(1448, L + rng.randint(16, 128))
            sh.append((d, L, it + rng.uniform(0.005, 0.04)))
        for k, (d, L, it) in enumerate(sh[:Xd.shape[1]]):
            td += L
            Xd[i, k, 0] = (d * L) / 1500.0
            Xd[i, k, 1] = min(1.0, it)
    return Xd, max(0.0, (td - to) / max(to, 1.0) * 100.0)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Reproduce audit finding F-10")
    ap.add_argument("--project", default="project", help="path to the v1 project directory")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import xgboost as xgb

    proc = os.path.join(a.project, "data", "processed")
    tab = np.load(os.path.join(proc, "tabular_dataset.npz"), allow_pickle=True)
    seq = np.load(os.path.join(proc, "sequence_dataset.npz"), allow_pickle=True)
    model_path = os.path.join(a.project, "3_models", "saved_models", "xgboost_baseline.json")

    rng = np.random.RandomState(a.seed)
    np.random.seed(a.seed)

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    Xte, yte, Ste = tab["X_test"], tab["y_test"], seq["X_test"]
    Str, ytr, Sva, yva = seq["X_train"], tab["y_train"], seq["X_val"], tab["y_val"]

    def ev(tag, X, model=clf, y=yte):
        p = model.predict_proba(X)[:, 1]
        print(f"  {tag:<56} acc={accuracy_score(y, p >= .5)*100:7.2f}%  "
              f"recall={recall_score(y, p >= .5)*100:7.2f}%  "
              f"AUC={roc_auc_score(y, p):.4f}")

    print("=== F-10 :: reproduce the repo numbers and add the MISSING control ===")
    ev("(1) original tabular features, saved model  [repo 'before']", Xte)
    Xrec = recompute_tab(Ste)
    ev("(2) CONTROL: features recomputed from UNMODIFIED tensor", Xrec)
    Xm1, oh1 = pad_mode1(Ste, yte, rng)
    Xm1t = recompute_tab(Xm1)
    ev(f"(3) Mode 1 padding, overhead={oh1:.1f}%  [repo 'after']", Xm1t)
    Xm2, oh2 = mode2(Ste, yte, rng)
    Xm2t = recompute_tab(Xm2)
    ev(f"(4) Mode 2 coalescing, overhead={oh2:.1f}%  [repo 'after']", Xm2t)

    print("\n  ^ condition (2) applies NO defence yet scores lower than (3) and (4).")
    print("    The reported 'defence effect' is a feature-pipeline artefact.\n")

    print("=== ADAPTIVE ADVERSARY :: retrain the censor on defended traffic ===")
    spw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    def retrain(Xtr_t, Xva_t):
        m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              random_state=a.seed, n_jobs=-1, eval_metric="logloss",
                              early_stopping_rounds=30)
        m.fit(Xtr_t, ytr, eval_set=[(Xva_t, yva)], verbose=False)
        return m

    for name, fn, Xdef_t in (("Mode 1 padding", pad_mode1, Xm1t),
                             ("Mode 2 coalescing", mode2, Xm2t)):
        Xtr_d, _ = fn(Str, ytr, rng)
        Xva_d, _ = fn(Sva, yva, rng)
        m = retrain(recompute_tab(Xtr_d), recompute_tab(Xva_d))
        p = m.predict_proba(Xdef_t)[:, 1]
        print(f"  retrained vs {name:<20} acc={accuracy_score(yte, p >= .5)*100:7.2f}%  "
              f"recall={recall_score(yte, p >= .5)*100:7.2f}%  "
              f"AUC={roc_auc_score(yte, p):.4f}  AP={average_precision_score(yte, p):.4f}")

    m = retrain(recompute_tab(Str), recompute_tab(Sva))
    p = m.predict_proba(Xrec)[:, 1]
    print(f"  reference: retrained on UNdefended    acc={accuracy_score(yte, p >= .5)*100:7.2f}%  "
          f"recall={recall_score(yte, p >= .5)*100:7.2f}%  AUC={roc_auc_score(yte, p):.4f}")

    print("\n  Conclusion: both defences give 100% recall against a censor that retrains.")
    print("  Report this, plus the mechanism: padding 1-128 B moves the 558 B mode to")
    print("  559-686 B, which never enters the legitimate upstream support of 40-81 B.")


if __name__ == "__main__":
    main()
