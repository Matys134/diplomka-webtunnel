# Final sign-off — v2.2 pilot, post-Transformer-fix

**Date:** 2 September 2026 · **Corpus:** 1,873 flows, 310 independent positive sockets
**Verified:** architecture diff, retrained model predictions, all six gates, every table and figure timestamp.

> ## Verdict: the science is finished. The thesis is **not** 100 % complete, and one of the four
> ## remaining items would become an unsupported claim if written as you have drafted it.
>
> Practical work: **~90 %**. The scientific core — the invariant, the leakage taxonomy, the
> adaptive-adversary result — is stable, defensible and ready to write. What remains is one
> 20-minute experiment, two stale artefacts, one uncommitted tree, and a claim in Q3(c) that your
> own data contradicts.

---

## 1 · Does the Transformer's saturation confirm the leakage taxonomy?

**Yes — and more strongly than you may realise, because it was a prediction, not a post-hoc fit.**

The §5.3 taxonomy made a falsifiable claim: *the Transformer's 87–93 % was an optimisation failure,
not evidence of a harder task; fix the embedding scaling and it will saturate like the others.*

Verified:

```
architectures.py:105   proj = self.input_proj(x) * math.sqrt(self.d_model)     ✓ present
best_val_loss          0.0102  ->  3.53e-05      (290x better; 1D-CNN is 1.27e-05)
held-out test          279/279 correct, all three models, socket-disjoint split
```

A prediction that survives a test is worth more than the observation that generated it. The
taxonomy stands:

| separator | who can see it | measured |
| --- | --- | --- |
| **Tor cell lattice** `L = 44 + 514k` | tabular (XGBoost) — and only this | `up_lattice_frac` alone: acc 1.0000 |
| **ClientHello / JA4** 267 B vs 506–602 B | sequence models, as **input element 0** | rule "first record < 400 B": TPR 1.0000, FPR 0.0000 |
| **First application record** 164 B | everything, survives the handshake cut | TPR 1.0000, FPR 0.0000 |

XGBoost never receives `clienthello_len` — it is manifest metadata, not one of the 50 features.
The sequence models receive it as element 0. Three models, two representations, and at least two
*independent* reasons the corpus separates.

### The one experiment that turns this from inference into measurement

Saturation is **consistent with** the taxonomy but does not prove it. The decisive test is cheap:

> **Retrain the 1D-CNN and the Transformer on the post-handshake sequence tensor** (element 0
> removed — `build_dataset.py --post-handshake`, which already exists).
>
> - If they stay at 100 %, they are reading the lattice too, and the taxonomy needs softening.
> - If they drop, element 0 was doing the work, and you have *measured* it rather than argued it.

Twenty minutes of GPU time, and it converts the most distinctive section of your thesis from a
plausible argument into an experimental result. **Do this before writing §5.3.**

---

## 2 · Are the regenerated artefacts coherent?

Mostly yes. Three defects, one of which is a visible contradiction between two tables.

### Fixed and verified

- `table_class_breakdown.tex`: **"Tor over HTTP/1.1 WebSocket"** ✓, and "HTTP/2 Web (TLS 1.3)" for
  Direct Browsing is a good addition — the h2/h1.1 distinction is now stated correctly on both sides.
- 15 of 16 figures regenerated at 06:05–06:07 ✓.
- `base_rate_fallacy_fdr.png` labels the measured curve and the two projections separately ✓ —
  exactly the honesty the earlier reviews asked for.
- The defence plots are generated from the same `rows` object as the LaTeX table, so they cannot drift.

### Defect 1 — two tables contradict each other about the same model **(blocking)**

```
table_class_breakdown.tex     Flow-Transformer   100.0 % on every class
table_model_comparison.tex    Flow-Transformer   98.0 +/- 1.4 %,  PR-AUC 0.911 +/- 0.113
```

`cross_validation_results.json` still holds the **pre-fix** folds
`[0.9662, 0.9634, 1.0, 0.9811, 0.9893]`. `cross_validate.py` was not re-run after the architecture
change. Worse: the *timing* columns of the same table were refreshed (0.0071 ms, 140,365 flows/s),
so `table_model_comparison.tex` now mixes fresh benchmarks with stale accuracy in a single row.

**Fix:** re-run `3_models/cross_validate.py`, then `4_evaluation/export_latex_tables.py`.

### Defect 2 — `computational_benchmark.png` was not removed

It is still in `4_evaluation/plots/`, timestamped **09-01 13:48** — about sixteen hours older than
the corpus. The report says it was deleted. Either delete it or regenerate it.

### Defect 3 — my own artefact is over-claiming, and I should have caught it earlier

`base_rate_results.json` reports `host_fpr_upper95 = 1.84e-03` at every *M*. That bound comes from
2,000 bootstrapped negative "hosts" — but those 2,000 pseudo-hosts are **resampled from 233 real
negative flows**. They are not independent trials, so a Clopper–Pearson bound computed on n = 2,000
is invalid. The give-away is that it is *tighter* than the per-flow bound (1.57 × 10⁻²) computed
from the same 233 flows. Bootstrap resampling cannot manufacture confidence.

**Fix:** use the bootstrap for the point estimate and the shape of the curve only; take every
interval from the real sample size (233 in the test split, or 1,563 across the CV folds). This is
in code I wrote — the correction is mine to flag.

### Still open from earlier reviews (not blocking, but a committee will ask)

- `table_cascaded_pipeline.tex`: L2 processes **0.0 %** of flows. A two-tier cascade whose second
  tier never executes cannot carry an economic argument.
- `table_handshake_comparison.tex`: full-flow and post-handshake are both exactly 100 % for both
  models, so the table cannot support "detection is independent of TLS metadata".
- Gates: **G1, G2, G3, G4 still FAIL**; G5, G6 PASS. `WEBTUNNEL_EXTRA_ARGS` is still `""` and
  `probe_utls_support.sh` has still not been run, so B-1 is untouched.
- **The work is uncommitted.** `33caa1a` does not exist in this repository; HEAD is `e87f4d0` and
  20 files are modified in the working tree. Given how much of this project's credibility now rests
  on provenance discipline, commit before you write.

---

## 3 · How to articulate the 100 % in Chapter 5

### (a) and (b) — the two mechanisms, stated as one narrative

Do not present three models converging on 100 % as three confirmations. Present them as **one
corpus with two independent separators, and two model families that each found a different one.**

> *Všechny tři klasifikátory dosahují 100 % přesnosti na testovací množině. Tento výsledek však
> nelze interpretovat jako trojí nezávislé potvrzení — analýza ukazuje, že modely využívají dva
> různé signály.*
>
> **The tabular model reads the protocol.** XGBoost operates on 50 aggregate statistics and never
> receives the ClientHello. A single feature, `up_lattice_frac`, reaches accuracy 1.0000 on its own,
> and the zero-parameter rule `(L − 44) mod 514 = 0` reaches TPR 1.0000 with no false positive in
> 233 negatives. **No machine learning is required to detect WebTunnel** — two integer operations
> per TLS record suffice. This is the thesis's positive result.
>
> **The sequence models read the laboratory.** The 1D-CNN and the Flow-Transformer receive the flow
> as a length sequence whose **element 0 is the ClientHello record**: 267 B for WebTunnel (stock Go
> `crypto/tls`, JA4 `t13d190900_…`) against 506–602 B for every negative (uTLS Chrome,
> JA4 `t13d1514h2_8daaf6152771_…`). A threshold at 385 B alone yields TPR 1.0000 and FPR 0.0000.
> Their 100 % is therefore *not* evidence of traffic analysis — it is a TLS-fingerprint lookup that
> a censor could perform without any model at all.

Then the consequence, which is the intellectually interesting part:

> *Toto rozlišení je samo o sobě výsledkem.* WebTunnel is detectable at two entirely different
> layers: at the handshake, by a static fingerprint that the operator could fix tomorrow by
> configuring uTLS imitation; and in the record-length distribution, by an invariant that follows
> from Tor's fixed 514-byte cell and cannot be removed without changing Tor itself. **Only the
> second is a durable property of the protocol.** A study that reported 100 % without separating
> them would have attributed a configuration defect to the protocol design.

Add the honest note that the corpus's stock-Go fingerprint is *itself* a finding — WebTunnel is
JA3/JA4-distinguishable out of the box — and cite it as such rather than as a limitation only.

### (c) — the base-rate claim, which your data does not support

**Do not write "multi-flow Bayesian LLR aggregation with M ≥ 4 is required".** Your own results
contradict it:

```
M     host_TPR   host_FPR        (base_rate_results.json, host_llr_sweep)
1     1.0000     0.00000
2     1.0000     0.00000
...
12    1.0000     0.00000
```

Host TPR is 1.0 and host FPR is 0 at **every** *M* from 1 to 12. There is no threshold at 4 or
anywhere else, because the per-flow detector already has zero false positives on this corpus, so
aggregation has nothing to improve. Writing "M ≥ 4" would put an unsupported number in the thesis
of exactly the kind three audits have been removing.

What the data *does* support, and it is a good paragraph:

> *Base-rate analysis in this thesis is **analytical**, not empirical, and is presented as such.*
> With 233 negatives in the held-out split the smallest measurable false-positive rate is
> 4.29 × 10⁻³; the measured rate is zero, so 4.29 × 10⁻³ is a **resolution limit, not an estimate**.
> Applying Bayes' rule at that limit gives a false-discovery rate of 97.7 % at an ISP-edge
> prevalence of α = 10⁻⁴ — i.e. *if* the true FPR were at our measurement floor, per-flow detection
> would be operationally useless. The projections at FPR = 10⁻⁴ and 10⁻⁵ give 50.0 % and 9.1 %
> respectively and are labelled as projections throughout.
>
> Whether host-level aggregation is *necessary* cannot be determined from this corpus: measuring an
> FPR small enough for the question to bite would require ≈ 26,000 captures for 10⁻³ and
> ≈ 260,000 for 10⁻⁴ (rule of three). Wails et al. (NDSS 2024) required 60 million flows to answer
> it. What this thesis contributes is the **mechanism and its cost**, not the operating point.

Then the free improvement, which you should take: report the bound over **all 1,563 negatives from
the cross-validation folds** rather than the 233-flow test split. Same data, same code, and the
bound tightens from 1.6 × 10⁻² to **1.9 × 10⁻³** — a 7× improvement for zero extra capture time.
Phrase it as *"no false positive in 1,563 legitimate flows; FPR < 1.9 × 10⁻³ at 95 % confidence."*

---

## 4 · Final verdict

**Not 100 % complete. Approximately 90 %, and the remaining 10 % is presentation plus one
experiment — not science.**

The scientific core is finished and I would defend it: a derived protocol invariant measured across
310 independent connections at 92.65 % vs ≤ 0.19 %; a zero-parameter detector with a proper
confidence interval; a leakage taxonomy that most theses at this level cannot produce; and a
countermeasure chapter with a genuine adaptive adversary and a measured 120.8 ms latency cost.

### Completion checklist

| # | Item | Effort | Blocking? |
| --- | --- | --- | --- |
| 1 | Re-run `cross_validate.py` + `export_latex_tables.py` — the Transformer contradiction | 15 min | **yes** |
| 2 | Retrain the sequence models on the post-handshake tensor (§1 above) | 20 min | **yes, for §5.3** |
| 3 | Drop the "M ≥ 4" claim; rewrite the base-rate paragraph as analytical | writing | **yes** |
| 4 | Delete or regenerate `computational_benchmark.png` | 2 min | no |
| 5 | Fix the bootstrap confidence bound in `evaluate_base_rate_fallacy.py` | 20 min | no |
| 6 | Run `probe_utls_support.sh`; either enable uTLS and re-pilot, or write the stock-Go fingerprint up as a finding | 1 h / 4 h | **decide before §5.2** |
| 7 | Register `up_len_p10` and `len_p10` as percentile echoes (G2) | 10 min | no |
| 8 | Commit the working tree | 5 min | **yes** |

Items 1, 2, 3 and 8 are under an hour together. After them, **start writing** — items 4–7 can
proceed in parallel with the text, and item 6 changes only how §5.2 is *phrased*, not what it says.

### One sentence for the defence

If the committee asks whether 100 % accuracy is credible, the answer is:

> *"It is, and we can tell you exactly why — twice. WebTunnel emits TLS records on the lattice
> 44 + 514k in 92.65 % of cases against at most 0.19 % of legitimate traffic, which is arithmetic,
> not machine learning. And our own corpus additionally carries a stock-Go TLS fingerprint that
> separates the classes on its own. We measured both, we report both, and only the first is a
> property of the protocol."*
