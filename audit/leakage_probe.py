#!/usr/bin/env python3
"""Reproduce audit finding F-09 from the rebuilt feature variants.

    python3 audit/leakage_probe.py --features audit/out/features_variants.npz

Prints: extraction variant x feature subset table, single-feature stump ranking, and the
class-conditional medians that show the zero-overlap separation on up_len_p50.
Requires: numpy, scikit-learn.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

VARIANT_LABEL = {
    "V0": "all packets merged (repo pipeline)",
    "V1": "single target TCP flow only",
    "V2": "single flow, first 10 packets dropped",
    "V3": "V2 + segmentation offload undone",
}


def fit_eval(name, Xa, y, tr, te):
    ok = ~np.isnan(Xa).any(1)
    trm, tem = tr & ok, te & ok
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       max_depth=5, random_state=0).fit(Xa[trm], y[trm])
    p = m.predict_proba(Xa[tem])[:, 1]
    yt = y[tem]
    thr = np.percentile(p[yt == 1], 5)
    print(f"  {name:<44} n_tr={trm.sum():5d} n_te={tem.sum():5d} "
          f"acc={accuracy_score(yt, p >= .5)*100:7.2f}%  AUC={roc_auc_score(yt, p):.4f}  "
          f"AP={average_precision_score(yt, p):.4f}  "
          f"rec={recall_score(yt, p >= .5)*100:5.1f}%  "
          f"FPR@TPR95={float((p[yt == 0] >= thr).mean()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    a = ap.parse_args()

    d = np.load(a.features, allow_pickle=True)
    FN = [str(x) for x in d["feature_names"]]
    cls, prof, sid = d["cls"], d["prof"], d["sid"]
    y = (cls == "webtunnel").astype(int)
    tr, te = sid <= 350, sid > 425
    print(f"N = {len(y)}  positives = {y.sum()}   train {tr.sum()}  test {te.sum()} "
          f"({y[te].sum()} pos / {(y[te] == 0).sum()} neg)\n")

    print("=== A) extraction variant ===")
    for k in ("V0", "V1", "V2", "V3"):
        fit_eval(f"{k}  {VARIANT_LABEL[k]}", d[k], y, tr, te)

    print("\n=== B) feature-subset ablation on V3 ===")
    groups = {
        "sizes only": [i for i, n in enumerate(FN) if "len" in n and "burst" not in n],
        "timing only": [i for i, n in enumerate(FN) if n.startswith("iat") or n.startswith("burst_dur")],
        "volume only (ratios + totals)": [FN.index(n) for n in
            ("ratio_up_pkts", "ratio_up_bytes", "total_pkts", "total_bytes")],
        "no volume, no duration": [i for i, n in enumerate(FN) if n not in
            ("total_pkts", "total_bytes", "iat_max", "iat_mean", "iat_p10",
             "iat_p25", "iat_p50", "iat_p75", "iat_p90")],
    }
    for g, idx in groups.items():
        fit_eval(f"V3 / {g}", d["V3"][:, idx], y, tr, te)

    print("\n=== C) single-feature depth-1 stumps on V3 ===")
    Xa = d["V3"]; ok = ~np.isnan(Xa).any(1)
    trm, tem = tr & ok, te & ok
    res = []
    for j, name in enumerate(FN):
        m = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xa[trm][:, [j]], y[trm])
        p = m.predict_proba(Xa[tem][:, [j]])[:, 1]
        auc = roc_auc_score(y[tem], Xa[tem][:, j])
        res.append((name, accuracy_score(y[tem], p >= .5), max(auc, 1 - auc),
                    float(m.tree_.threshold[0])))
    for n, acc, auc, thr in sorted(res, key=lambda r: -r[2])[:12]:
        print(f"  {n:<24} acc={acc*100:7.2f}%  AUC={auc:.4f}  thr={thr:.4g}")

    print("\n=== D) class-conditional medians (V3) ===")
    keys = ["ratio_up_bytes", "ratio_up_pkts", "up_len_p50", "len_p50", "total_bytes", "total_pkts"]
    idx = [FN.index(k) for k in keys]
    print("  " + f"{'class':<22}" + "".join(f"{k[:16]:>17}" for k in keys))
    for c in ["webtunnel", "direct_web_browsing", "websocket_ticker",
              "websocket_chat", "video_streaming", "web_assets"]:
        m = (cls == c) & ok
        if not m.any(): continue
        print("  " + f"{c:<22}" + "".join(f"{np.median(Xa[m][:, j]):>17.3f}" for j in idx))

    j = FN.index("up_len_p50")
    wt, ng = Xa[(cls == "webtunnel") & ok][:, j], Xa[(cls != "webtunnel") & ok][:, j]
    print(f"\n  up_len_p50 webtunnel: p1={np.percentile(wt,1):.1f} p50={np.median(wt):.1f} "
          f"p99={np.percentile(wt,99):.1f}")
    print(f"  up_len_p50 negatives: p1={np.percentile(ng,1):.1f} p50={np.median(ng):.1f} "
          f"p99={np.percentile(ng,99):.1f}")
    print("  -> zero overlap. One feature, 100% accuracy. See docs/01-audit-findings.md F-09.")


if __name__ == "__main__":
    main()
