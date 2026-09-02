# Order-shuffle audit — what the control proves, and what it cannot

**Date:** 2 September 2026 · **HEAD:** `c0c9c60`, working tree clean
**Verified independently:** `shuffle_flow_records()` implementation, the sequence-tensor round-trip,
set-membership predicates on the authoritative `flow_records.jsonl`, train/test contamination,
and a permutation-invariant emulation of the shuffled sequence models.

> ## The experiment is correctly designed and correctly implemented. Its conclusion does not follow.
>
> Order-shuffling ablates **position**. It preserves the **multiset**. In this corpus *every*
> candidate signal — including both artefacts you were trying to rule out — is a property of the
> multiset. The control was therefore guaranteed to return 100 % under every hypothesis, and it
> cannot rank them.

---

## Q1 · Does the order-shuffle result prove the models read the lattice?

**No.** The implementation is sound — I checked it:

```python
active_mask = np.abs(flow).sum(axis=-1) > 1e-6      # permutes only the active prefix
perm = rng.permutation(active_len)                  # padding structure preserved
X_shuffled[i, :active_len] = flow[:active_len][perm]
```

Independent seeds per split, no leakage between train and test. The tensor round-trip is exact
(max |recovered − true| = 0 over 51,992 records), so nothing is lost in the representation.

The problem is what the transformation leaves untouched.

### Every signal in this corpus is order-invariant

| Signal | Predicate | TPR | FPR | Order-invariant? |
| --- | --- | ---: | ---: | :---: |
| Tor cell lattice | fraction of records with `(L−44) mod 514 = 0` | 1.0000 | 0.0000 | **yes** |
| ClientHello | *does the flow contain a 267 B record?* | **1.0000** | **0.0070** | **yes** |
| First app record | *does the flow contain a 164 B record?* | **1.0000** | **0.0141** | **yes** |
| Flow length | count of active records (from the padding mask) | — | — | **yes** (stump AUC 0.7918) |

Measured on `flow_records.jsonl`, not on the tensor. **A permutation of the record order changes
none of these four numbers**, because set membership and multiset fractions do not depend on order.
The 267 B ClientHello sits at index 0 before the shuffle, so it is always inside the 200-record
window and always in the shuffled multiset afterwards.

So the experiment could only ever have returned 100 %, whichever hypothesis is true. It has no
discriminative power for the question it was built to answer.

### What it *does* establish — and this is worth keeping

Positional and n-gram cues are **not necessary**. That is a real negative result and it eliminates
a family of explanations (the models are not memorising "index 0 = 267 B", nor any prefix pattern).
Report it as such. It is simply weaker than *"they read the lattice"*.

### The over-determination is worse than two hypotheses

I emulated a permutation-invariant sequence model directly — a bag-of-record-lengths classifier,
which is exactly the information a shuffled 1D-CNN or [CLS]-pooled Transformer can access:

```
full multiset                                                    acc 1.0000  AUC 1.0000
multiset minus the 267 B ClientHello                             acc 1.0000  AUC 1.0000
multiset minus 267 B and 164 B                                   acc 1.0000  AUC 1.0000
multiset minus 267 B, 164 B and EVERY lattice value              acc 1.0000  AUC 1.0000
   ... same, with the flow-length feature also removed           acc 1.0000  AUC 1.0000
```

Removing every named signal still leaves perfect separation. The residual length distributions are
simply disjoint enough on their own. **Ablation-based attribution does not converge on this corpus**,
and no further ablation experiment will make it converge. Stop running them.

### What *is* proven cleanly, today, with no new experiment

`up_lattice_frac` **alone** — a single order-free feature — reaches accuracy 1.0000 and AUC 1.0000
on the socket-disjoint test split. This is airtight as a **sufficiency** claim, because neither
267 nor 164 lies on the lattice: they can only enter that feature through its denominator, and the
class gap is ≥ 0.40 against ≤ 0.002. One or two non-lattice records cannot manufacture that.

**Reframe the claim to match what is provable.** The thesis is about the detectability of a
*protocol*, not about which feature a particular neural network attends to:

- ✅ *"The Tor cell lattice is sufficient for perfect detection on this corpus."* — proven, by a
  single feature and by a zero-parameter rule.
- ✅ *"The corpus additionally contains laboratory separators that are not protocol properties."* —
  proven, with TPR/FPR for each.
- ❌ *"The 1D-CNN and Transformer detect the lattice."* — not determined, and **not needed**. Drop it.

---

## Q2 · How to prove the 100 % is genuine rather than a broken split

Run the contamination checks and put the numbers in the text. I ran them:

```
exact duplicate feature vectors, train ∩ test        0
exact duplicate vectors within train                 0
test → train 1-NN distance (standardised)            min 0.190   p05 0.422   median 0.985
test rows with a train neighbour closer than 0.01    0
```

Combined with what the gates already assert:

- **G5** — socket-disjoint *and* conn_id-disjoint splits, group vector element-wise aligned,
  310 independent positive sockets, effective *n* = 310.0, largest socket carries 0.3 % of positives.
- **G6** — 1,873 flows, 0 missing manifests, **0 five-tuple mismatches**, 100 % capture accounting,
  143 logged drops with per-class reasons.
- **G3** — label-shuffle control at AUC 0.5027 [0.4700, 0.5353]. If there were an index-level leak,
  a shuffled-label model would find it. It does not.

### The rhetorical move that actually wins the room

Do not defend 100 % as an accuracy figure. **Defend it as the consequence of a closed-form property,
and present the derivation before the measurement.**

> *An accuracy that was predicted from arithmetic is not suspicious. An accuracy that was discovered
> is.* Derive `L = 44 + 514k` from the Tor cell specification, state that it implies near-perfect
> separability against any traffic that does not quantise its records the same way, **then** show
> that the measurement agrees. The reader watches a prediction being confirmed rather than a number
> being explained away.

Then volunteer the weakness before anyone asks for it: *"our corpus contains at least three
separators and only one of them is a property of the protocol — here is the taxonomy, and here is
which of them a real deployment could remove."* A committee that sees you rank your own confounds
stops looking for the one you hid.

---

## Q3 · The final taxonomy for §5.3

State each claim with its epistemic status. That ladder is the contribution.

| | Signal | Nature | Status | Evidence |
| --- | --- | --- | --- | --- |
| **S1** | Tor cell lattice `L = 44 + 514k` | **protocol invariant** | **sufficiency proven** | `up_lattice_frac` alone: acc 1.0000. Zero-parameter rule: TPR 1.0000, 0 FP in 233 test / 1,563 CV negatives. 92.65 % of WebTunnel upstream records vs ≤ 0.19 % |
| **S2** | 267 B ClientHello (stock Go `crypto/tls`) | **laboratory artefact** — removable by configuration | **present, quantified** | presence rule TPR 1.0000 / FPR 0.0070; JA4 `t13d190900_…` vs `t13d1514h2_8daaf6152771_…`; G1 FAIL |
| **S3** | 164 B first application record | **transport property** — HTTP/1.1 Upgrade vs h2 preface | **present, quantified** | presence rule TPR 1.0000 / FPR 0.0141; survives the handshake cut |
| **S4** | Flow length / volume | **laboratory artefact** — the G4 budget failure | **present, quantified** | padding-mask stump AUC 0.7918; `total_bytes` spans 4.7 KB – 2.79 MB across classes |

Two cross-cutting results to state explicitly:

1. **All four are order-invariant** (the shuffle control). Positional and n-gram cues are not
   necessary for any of them.
2. **The corpus is over-determined**, so which signal a given model uses cannot be attributed by
   ablation. Say this outright; it is the honest limit of the method, and stating it is what
   distinguishes a careful study from a lucky one.

### Suggested §5.3 structure

1. **The problem.** Three models reach 100 %. In a well-designed study that is a warning, not a result.
2. **The method.** A separator is any predicate with TPR ≈ 1 and FPR ≈ 0. Enumerate them; classify
   each as protocol or laboratory; state what a real deployment could remove.
3. **The four separators** — the table above, with derivations.
4. **The two controls and their limits.** Order-shuffle: rules out position, cannot rank the rest.
   Post-handshake: rules out the ClientHello, leaves S1/S3/S4. Say what each *cannot* show.
5. **What survives.** Only S1 is a property of Tor. S2 disappears the day the operator configures
   uTLS imitation; S3 disappears if WebTunnel moves to HTTP/2; S4 is an artefact of our budget
   controller. **The durable claim is S1, and it needs no machine learning.**
6. **The methodological point.** A study reporting 100 % without this decomposition would have
   attributed a configuration defect to the protocol design. The decomposition *is* the result.

---

## Q4 · Traps a hostile reviewer will find

| # | Trap | Severity | What to do |
| --- | --- | --- | --- |
| 1 | **B-1 is still open.** `WEBTUNNEL_EXTRA_ARGS: ""`; `probe_utls_support.sh` never run. Deferred five times. It is the *only* clean removal of S2, and it is one environment variable. | **high** | Run the probe. Either enable uTLS and re-pilot (S2 disappears at source, G1 goes green) or write the stock-Go fingerprint up as a finding. Do not leave it undecided in the text. |
| 2 | **G1–G4 still FAIL.** Anyone who opens `checks/` sees four red gates behind a chapter of 100 % results. | **high** | Report the gate table honestly in §5.1 with a sentence per failure. A stated failure is a limitation; a hidden one is a finding against you. |
| 3 | **The 200-record window is class-dependent.** `video_streaming` 95.2 % truncated, `web_assets` 75.9 %, `webtunnel` 2.9 %. The padding mask alone gives AUC 0.79. | medium | Name it as S4's mechanism, or re-run with a longer window and show the result is unchanged. |
| 4 | **The IAT channel is nearly dead.** Mean \|x\| = 0.0027, p99 = 0.038 after `log1p(Δt)/5`. | medium | Do not claim the models use timing. They use sizes. Say so, or rescale and re-run. |
| 5 | **`table_order_shuffle.tex` has no caveat.** Four rows of 100 % with no statement of what the control cannot show. | medium | Add one sentence to the caption: *"the control ablates order only; every signal in the corpus is order-invariant, so this rules out positional cues without ranking the remaining separators."* |
| 6 | **Cascade L2 handles 0.0 % of flows.** | low | Report the escalation rate as zero-on-a-saturated-task, or cut the chapter. |
| 7 | **Single bridge, single vhost, one epoch, no QUIC.** | low–medium | §5.6, with the numbers already in the gate output. |

### Is the codebase ready?

**The code and the corpus: yes.** Provenance is authoritative and verified, splits are
socket-disjoint with a proven-falsifiable harness (`test_gates.py` 15/15), there is no train/test
contamination at any distance I could measure, and the pipeline is reproducible from a clean tree.
This is better engineering than most work at this level.

**The interpretation: not yet.** The drafted §5.3 conclusion overreaches on exactly the point this
audit examined. Fix the claim rather than the corpus — the corrected version is stronger, because
"we found three separators and only one is the protocol" is a more interesting sentence than
"our classifier got 100 %".

**One item of real work remains, and it is item 1.** Everything else is writing.
