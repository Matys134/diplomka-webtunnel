# audit/ — re-runnable verification

Every script here is independent of `project/` code. That independence is the point: the audit's
credibility rests on not reusing the pipeline it is auditing.

| Script | Reproduces | Needs PCAPs? |
| --- | --- | --- |
| `defense_recheck.py` | F-10 — the defence artefact and the adaptive adversary | **no** |
| `pcap_forensics.py` | F-01 … F-06 — connections, handshakes, fingerprints, histograms | yes |
| `rebuild_features.py` | builds the four extraction variants used by F-09 | yes |
| `leakage_probe.py` | F-09 — variant × subset × stump analysis | no (needs the .npz above) |

## Without the raw captures

```bash
pip install -r ../requirements-working.txt
python3 defense_recheck.py --project ../project
```

Expected: condition (2) — *no defence applied* — scores 24.89 % recall, below the 36.0 %
reported for "after defence"; and both defences give ~100 % recall against a retrained censor.

## With the raw captures

```bash
PCAPS=/path/to/webtunnel_pcaps_9000/raw_pcap

python3 pcap_forensics.py   --pcap-dir "$PCAPS" --mode all
python3 rebuild_features.py --pcap-dir "$PCAPS" --out out/features_variants.npz
python3 leakage_probe.py    --features out/features_variants.npz
```

`rebuild_features.py` takes 5–8 minutes single-threaded over 9,000 files; pass `--limit N` to
sample. All measured values are tabulated in `../docs/03-evidence.md`.
