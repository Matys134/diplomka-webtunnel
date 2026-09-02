# v2.2 pilot review — answers to the five questions

**Date:** 1 September 2026 · **Corpus:** 2,016 captures → 1,873 flows, 310 independent positive sockets
**Method:** gates re-executed; ClientHellos, JA4s and record lattices re-parsed from `flow_records.jsonl`;
classifiers re-trained under ablation; sequence tensors inspected element-wise.

> **Verdict: 8/10. The invariant is now proven, and the defence chapter has become the best part
> of the thesis. Two things still block a clean defence, and one of them is not a data problem —
> it is a single environment variable that was never set.**
>
> `WEBTUNNEL_EXTRA_ARGS` is still `""`. `probe_utls_support.sh` was still not run. So B-1 is
> untouched, G1 still fails, and a one-line JA4 lookup is still a perfect classifier.

---

## What v2.2 genuinely fixed — verified

| Fix | Evidence |
| --- | --- |
| **B-2 ALPN parity** | All five negative classes now share **one** JA4 (`t13d1514h2_8daaf6152771_02713d6af862`), one ALPN offer, one ClientHello length set {506, 538, 570, 602}. Previously two JA4s and two ALPN offers. |
| **B-4 server stack** | `down_len_min` is **gone** — it now has AUC 0.5000 and a constant value across all classes. The nginx front end did exactly what it was supposed to do. |
| **B-3 local target** | WebTunnel's byte budget is now the *only* one that is roughly honoured (up 1.22×, down 0.78× of target). The public-web confound is gone. |
| **Lattice** | 92.65 % of WebTunnel upstream records and 86.03 % of downstream records on `L = 44 + 514k`, versus **0.00 – 0.19 %** for every negative class. The strongest separation measured in this project. |
| **G5 / G6** | 310 independent positive sockets, effective *n* = 310.0, largest carries 0.3 %; 0 five-tuple mismatches, 100 % capture accounting, 143 logged drops with reasons. Both PASS. |

**Current gate state:** G1 FAIL · G2 FAIL · G3 FAIL · G4 FAIL · G5 PASS · G6 PASS.

---

## Q1 · Is the near-100 % genuine, or is there residual leakage?

**Both. The invariant is genuine and the leakage is total — and they are different leaks for
different models.** This is the single most important thing to get straight before writing.

### 1a. The invariant is real, and it is sufficient on its own

```
up_lattice_frac ONLY (1 feature)      acc 1.0000   AUC 1.0000   AP 1.0000
lattice features ONLY (2 features)    acc 1.0000   AUC 1.0000   AP 1.0000
lattice rule, thr 0.1 … 0.7           TPR 1.0000   0 FP in 233 negatives   ROC-AUC 1.0000
```

The rule is now insensitive to its threshold across the whole range 0.1–0.7 — in v2.1 it collapsed
above 0.3. That robustness is what a real invariant looks like, and it is theoretically expected:
`L = 44 + 514k` follows from the 514-byte Tor cell, 22-byte WS/HTTPT framing, 5-byte TLS header,
1-byte inner content type and 16-byte AEAD tag. **This part of the thesis is defensible as it stands.**

### 1b. But a JA4 lookup is still a perfect classifier

```
clienthello_len   webtunnel [267]        negatives [506, 538, 570, 602]
JA4               webtunnel t13d190900_9dc949149365_e7c285222651   (stock Go crypto/tls)
                  negatives t13d1514h2_8daaf6152771_02713d6af862   (Chrome, via uTLS)

rule "clienthello_len < 385":   TPR 1.0000   FPR 0.0000
```

`WEBTUNNEL_EXTRA_ARGS` in `docker-compose.yml` is still empty and `probe_utls_support.sh` has still
not been run. B-1 is exactly where it was two reviews ago.

### 1c. The 1D-CNN's 100 % is the JA4 leak, not the lattice

This is the finding that matters most, and it is easy to miss:

```
sequence tensor element 0 (the first TLS record of the flow):
    webtunnel  267 B          negatives  506 / 538 / 570 / 602 B
rule "first record < 400 B":   TPR 1.0000   FPR 0.0000
```

The 1D-CNN and the Flow-Transformer are handed the ClientHello length as **input element 0**. They
do not need the lattice, the timing, or anything else. XGBoost is different: `clienthello_len` is
stored as metadata and is *not* one of its 50 features, so **XGBoost's 100 % genuinely is the
lattice** while **the 1D-CNN's 100 % is the stack leak**. Two models, two headline numbers, two
completely different causes. Reporting them side by side as mutual corroboration would be wrong.

### 1d. Post-handshake is over-determined too

The handshake cut is clean (1 of 1,873 flows retains a handshake record; median `hs_end_idx` = 9).
But the *first application record* is a per-class constant:

```
webtunnel 164 B (310/310 flows)   direct/video/assets 92 B   ws_chat 216 B   ws_ticker 218 B
rule "first application record == 164 B":   TPR 1.0000   FPR 0.0000
```

So "post-handshake accuracy is 100 %" does not isolate the lattice either. It is a real transport
difference (HTTP/1.1 Upgrade versus an HTTP/2 client preface), not an artefact — but it is not the
Tor cell, and the thesis must not conflate them.

### 1e. Volume is still class-informative — G4 fails harder than in v2.1

| class | observed down / target | observed duration / target |
| --- | ---: | ---: |
| `video_streaming` | **31.88 ×** | 2.84 / 3.20 |
| `web_assets` | 9.05 × | 2.41 / 3.20 |
| `webtunnel` | 0.78 × | **6.13 / 3.20** |
| `direct_web_browsing` | **0.06 ×** | 2.24 / 3.20 |

`total_bytes` spans 4.7 KB to 2.79 MB across classes — a 585× range. Two causes, both fixable:
the bulk classes check the budget *before* issuing a request and overshoot on a single large
response; the interactive classes are throttled by `b.think()` and never reach it. WebTunnel
overshoots on *duration* because an in-flight onion request outlives `targetDur`. Worst median
ratio is now 24.79 (was 13.87).

### 1f. Two new G2 offenders, one real and one worth a look

- `up_len_p10` (AUC 0.9569) and `len_p10` (0.9436) are further **percentile echoes** — when 92.65 %
  of upstream records sit on 558 B, even the 10th percentile is 558. Same derivation as the ones
  already registered; extend the registry.
- `iat_p10` (0.9241): WebTunnel 6.6 × 10⁻⁵ s, every negative exactly 0. Measured cause: negatives
  pack several small TLS records into one TCP segment (16–60 % of record pairs share a timestamp)
  while WebTunnel flushes roughly one 558 B cell per segment (6 %). That is a genuine consequence
  of Tor's per-cell scheduling and belongs in the thesis as a *second* protocol observable — but
  derive it explicitly before registering it, because it is partly a property of how the flow
  builder timestamps a record.

---

## Q2 · Why does Flow-Transformer show 11–13 % false positives?

**It is an implementation bug, not realism. Do not present it as realistic confusion.**

The Transformer sees the *same* input tensor as the 1D-CNN, in which element 0 alone separates the
classes perfectly. A model that fails on that input is under-fitted. The evidence:

```
1D-CNN        best_val_loss = 1.27e-05
Transformer   best_val_loss = 1.02e-02      <- 800x higher, identical data
```

The cause is in `3_models/architectures.py`:

```python
self.input_proj = nn.Linear(in_features=2, d_model=64)   # output RMS ~ 0.048
...
return x + self.pe[:, :x.size(1), :]                     # sinusoids, RMS 0.707
```

Measured on the actual tensors: the projected content has RMS ≈ 0.048 and the positional encoding
has RMS ≈ 0.707. **The positional signal is about 15× louder than the content it is added to.**
Vaswani et al. §3.4 multiply the embedding by `sqrt(d_model)` (= 8 here) before adding the
positional encoding for exactly this reason; this implementation does not. The encoder is being
shown position clearly and content faintly.

A second, smaller issue compounds it: after `log1p(Δt)/5` the IAT channel has mean |x| = 0.0027 and
p99 = 0.038 — it is very nearly a constant-zero channel, so half the input carries almost nothing.

**Fix (one line):** `proj = self.input_proj(x) * math.sqrt(self.d_model)`, or insert a
`nn.LayerNorm(d_model)` after the projection. Then retrain. Expect the Transformer to join the
other two at ~100 %, which is the honest outcome.

**Why this matters for the defence.** A committee member who knows Transformers will spot the
missing scaling in under a minute. If the thesis has meanwhile built an argument on "the
Transformer shows realistic confusion", that argument collapses and takes credibility with it.
The correct framing is the opposite: *on a task this separable, every correctly-trained model
should saturate; the Transformer did not, and here is the bug.* Reporting that you found and fixed
it is a strength.

---

## Q3 · Is 1,873 flows enough, or do you need 5,000?

**Neither more flows nor 5,000 captures is the right answer. The binding limit is not sample size.**

### The bottleneck is test-split negatives, and it moves slowly

```
0 FP in    233 negatives (current test split)  -> 95% upper bound on FPR  1.3e-02
0 FP in  1,563 negatives (all negatives, via CV) ->                        1.9e-03
0 FP in  3,000 negatives                        ->                        1.0e-03
0 FP in 30,000 negatives                        ->                        1.0e-04
```

A 5,000-capture campaign yields roughly 577 test negatives → a bound of **5.2 × 10⁻³**. That is a
2.5× improvement for 2.5× the capture time, and it still cannot claim 10⁻³. To *measure* 10⁻³ you
need about 3,000 test negatives ≈ **26,000 captures**; for 10⁻⁴, about ten times that. Both are
out of scope, and the rebuild plan already said so.

### Free improvement you should take first

Report the FPR bound over **all 1,563 negatives using the existing cross-validation** rather than
the single 233-flow test split. Same data, same code, and the bound tightens from 1.3 × 10⁻² to
**1.9 × 10⁻³** — a 7× improvement for zero additional capture hours. State the operating point as
*"no false positive in 1,563 legitimate flows; FPR < 1.9 × 10⁻³ at 95 % confidence"*.

### The positive side is already comfortable

310 independent sockets, all classified correctly → Clopper–Pearson **[0.9882, 1.0000]** at the
connection level. That is a publishable TPR interval. More positive flows would not improve the
claim materially.

### What to spend the hours on instead

The limitation a committee will press on is **external validity**, not statistical power:
one bridge, one legitimate vhost, one capture epoch, no QUIC class. In descending value:

1. **Fix B-1** (an environment variable and a re-pilot) — worth more than 20,000 extra captures.
2. **A second capture epoch** — gives a genuine temporal split for ~3.5 h of unattended time.
3. **Two more bridge instances and 3–4 more vhosts** — makes the *unseen-destination* split
   possible, which is the axis the roadmap promised and no version has ever delivered.
4. A ~500-flow QUIC/HTTP-3 negative class, which the assignment names explicitly.

---

## Q4 · Are the tables and plots defensible?

Partly. The defence table has become the best artefact in the project. Six figures are stale, and
three tables still carry defects flagged in earlier reviews.

### What is now excellent

`table_before_after_defense.tex` — this is the result to lead with:

| defence | adversary | recall | bytes | latency |
| --- | --- | ---: | ---: | ---: |
| none | static | 100.0 % | — | — |
| padding | **static** | **0.0 %** | 3.5 % | 0 ms |
| padding | **adaptive** | **100.0 %** | 3.5 % | 0 ms |
| coalescing + chatter | **static** | **0.0 %** | 1.5 % | **120.8 ms** |
| coalescing + chatter | **adaptive** | **100.0 %** | 1.5 % | 120.8 ms |

In v2.1 padding left the static model at 100 % recall — proof it was riding artefacts. Now both
defences defeat the frozen model completely and **both are worthless against a censor that
retrains**. That is a clean, quotable, genuinely interesting result, with a measured cost
(120.8 ms of buffering latency for 1.5 % of bandwidth) that nobody else has reported.

### Stale figures — six of sixteen

The corpus and every table were rebuilt at 22:59–23:01. These figures were not:

```
12:29  before_vs_after_distributions.png   before_vs_after_metrics.png     <- v2.1 corpus or older
13:48  base_rate_fallacy_fdr.png           computational_benchmark.png
19:00  1d_cnn_saliency_map.png             transformer_attention_map.png
       xgboost_shap_summary.png            xgboost_feature_importance.png
       iat_distribution.png                packet_length_distribution.png
```

The two `before_vs_after_*` figures are the worst case: they illustrate the defence chapter and
they **contradict the defence table sitting next to them**. The four XAI figures show feature
attributions computed on a different dataset. Re-run those phases before anything goes into the
thesis.

### Table defects still open

- **`table_class_breakdown.tex` still says "Tor over HTTP/2 WSS".** This has survived three
  reviews. WebTunnel offers *no* ALPN (its JA4 ends `00`), so it runs HTTP/1.1 Upgrade — and now
  the negatives genuinely do run h2, so the label is wrong in a way a reader can check in one line.
- **`table_cascaded_pipeline.tex`: L2 handles 0.0 % of flows.** A two-tier cascade whose second
  tier never executes cannot support an economic argument. Either report the escalation rate as
  *zero on a saturated task, therefore uninformative*, or cut the chapter.
- **`table_model_comparison.tex`: `$100.0 \pm 0.0\%$`** is not a confidence interval, and it mixes
  5-fold CV means with single-run test numbers in one table. Use Clopper–Pearson at the connection
  level, and say which split each column comes from.
- **`table_handshake_comparison.tex`:** post-handshake equals full-flow at 100 % for both models.
  Per Q1d this cannot support "detection is independent of TLS metadata" — the first application
  record separates perfectly on its own.

---

## Q5 · Recommendations for the write-up

### Do these three things first (about half a day, plus one re-pilot)

1. **Run `probe_utls_support.sh` and act on the answer.** Either set `WEBTUNNEL_EXTRA_ARGS` and
   re-pilot, or write the stock-Go fingerprint up as a finding in its own right — *WebTunnel is
   JA3/JA4-detectable out of the box, before any traffic analysis* — and make the handshake-stripped
   analysis the **primary** result rather than an ablation. Both are defensible. Leaving it
   unaddressed is not.
2. **Fix the Transformer's embedding scaling** and retrain, then delete any prose describing its
   errors as realistic.
3. **Regenerate the six stale figures.**

### The claims that are safe, and the ones that are not

| Safe to claim | Not safe |
| --- | --- |
| `L = 44 + 514k` holds for 92.65 % of WebTunnel upstream records and ≤ 0.19 % of legitimate ones | "Our classifiers detect WebTunnel with 100 % accuracy" |
| A zero-parameter rule achieves TPR 1.0 with no false positive in 1,563 legitimate flows (FPR < 1.9 × 10⁻³, 95 %) | Any FPR figure at 10⁻⁴ or below as a *measurement* |
| Measured across 310 independent connections, three network profiles, socket-disjoint splits | "Detection is independent of TLS metadata" |
| Both countermeasures defeat a frozen model and neither survives a retraining censor | "The 1D-CNN confirms the XGBoost result" (different signals) |
| Coalescing costs 120.8 ms of buffering latency for 1.5 % bandwidth | Cascade throughput as an operational figure (L2 never runs) |

### Chapter 5 structure — refined from the previous review

**§5.1 The apparatus and what it rejected.** Six gates, `test_gates.py` 15/15, and the honest table
of rejections: v1 corpus, v2.0 corpus, v2.1 corpus on G1–G4, v2.2 on G1–G4. Then the accepted
corpus. This is the assignment's *"předzpracování dat pro zamezení učení modelu z irelevantních
znaků"* requirement, discharged visibly.

**§5.2 The invariant.** Derive `44 + 514k`, show the ladder with counts, the class table
(92.65 % vs ≤ 0.19 %), and the threshold-insensitivity sweep (TPR 1.0 from 0.1 to 0.7). State it as
a *detector*: TPR 1.0000 [0.9229, 1.0000], zero false positives, no machine learning.

**§5.3 What the models are actually using — the leakage taxonomy.** New section, and it will be the
most distinctive part of the thesis. Present the ablation honestly: `up_lattice_frac` alone gives
1.0000; the 1D-CNN reads the ClientHello at input element 0; the first application record is a
per-class constant. Three independent perfect separators, one of which is a protocol invariant and
two of which are laboratory properties. Most theses cannot tell these apart; yours can, and saying
so is the methodological contribution.

**§5.4 Countermeasures and the adaptive adversary.** The table above, then the mechanism —
padding destroys the lattice (fraction 0.9265 → 0.0000) and coalescing only halves it
(→ 0.5632), which is why a retrained model recovers. Close with the design requirement: move the
distribution *into* the legitimate support, i.e. MTU-sized coalescing plus injected HTTP/2 control
chatter. Cite Huma (NDSS 2026) as independent corroboration.

**§5.5 Base rate and resolution.** Report the CV-wide bound (1.9 × 10⁻³), label the 10⁻⁴/10⁻⁵
columns as analytical projections, and state what measuring them would have cost (26,000 and
260,000 captures).

**§5.6 Threats to validity.** Write it yourself: one bridge, one vhost, one epoch, no QUIC, the
uTLS gap if B-1 stays open, and the budget-parity failure (G4). Every one already has a number
from the gates — use them. A committee rewards a limitation you quantified far more than one they
had to find.

### On the student's worry

The instinct that "100 % looks too high" is the correct instinct, and it has now been followed to
the right place. The answer is not more data. It is that **the task genuinely is close to trivially
separable once WebTunnel's cell quantization is visible** — that is the finding — **and that this
particular corpus is separable for two further reasons that have nothing to do with Tor.** Say all
three out loud, show which is which, and the 100 % stops being a weakness and becomes the point.
