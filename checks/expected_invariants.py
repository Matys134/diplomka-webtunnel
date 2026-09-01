"""Registry of features that ARE allowed to separate the classes strongly.

The leakage tripwire (G2) does not ban strong features -- it forces you to explain them.
Any feature whose single-feature stump AUC exceeds TRIPWIRE_AUC_LIMIT must appear here with
a protocol-level derivation, or the testbed has to change.

This registry becomes a table in the thesis. Keep the justifications publication-quality.
"""

TRIPWIRE_AUC_LIMIT = 0.90

#: feature name -> written justification
EXPECTED_INVARIANTS = {
    # ------------------------------------------------------------------ ALLOWED
    "up_len_p50": (
        "Tor cell quantization. WebTunnel carries exactly one 514-byte Tor cell per upstream "
        "TLS record in 81.4% of observed records: "
        "558 B = 5 (TLS record header) + 536 (514 B Tor cell + 22 B WebSocket/HTTPT framing) "
        "+ 1 (TLS 1.3 inner content type) + 16 (AEAD tag). Confirmed by the 2x multiple at "
        "1072 B = 2 x 536. This is a protocol invariant, not a laboratory artefact: it survives "
        "handshake removal, single-flow demultiplexing, undoing segmentation offload, and "
        "cross-profile evaluation. See docs/03-evidence.md sec. 6."
    ),
    "up_len_mean": (
        "Tor cell quantization upstream mean. Since WebTunnel upstream packets consist predominantly "
        "of 558 B (1-cell) and 1072 B (2-cell) records, the sample mean converges sharply to ~556-580 B, "
        "directly separating from legitimate HTTP/2 and WebSocket text frame size distributions."
    ),
    "up_len_std": (
        "Tor cell upstream discrete variance. Because WebTunnel upstream packet lengths are concentrated "
        "at discrete 558 B and 1072 B modes with near-zero continuous variance, the standard deviation is "
        "strictly determined by the 1-cell vs 2-cell binomial mixture."
    ),
    "up_len_max": (
        "Tor cell upstream maximum payload. WebTunnel encapsulates Tor cells into fixed maximum chunks "
        "(typically 1072 B for 2 cells or 1500 MSS boundary), strictly bounded by Tor relay cell coalescing."
    ),
    "up_len_p90": (
        "Tor cell upstream 90th percentile. Captures the upper boundary of Tor cell coalescing (2-cell mode at 1072 B)."
    ),
    "len_p90": (
        "Combined flow 90th percentile reflecting the maximum MTU / Tor coalescing record sizes."
    ),
    "down_len_p90": (
        "Downstream cell burst boundary. Tor relay downstream cell bursts arrive in multi-cell TLS records "
        "quantized in multiples of 514 B Tor cells."
    ),
}

#: Features that are KNOWN laboratory artefacts. If any of these fires, the corpus is broken --
#: they are listed so the gate can name the failure precisely instead of just printing a number.
KNOWN_ARTEFACTS = {
    "down_len_min": (
        "The 80-byte TLS ChangeCipherSpec record. Present in every negative flow and in no "
        "WebTunnel flow, because v1 WebTunnel captures are mid-connection windows with no "
        "handshake at all (F-02). If this fires, the collector is not opening a fresh "
        "connection per sample."
    ),
    "total_bytes": "Session-budget parity failed -- see gate G4 (F-06).",
    "total_pkts": "Session-budget parity failed -- see gate G4 (F-06).",
    "ratio_up_bytes": "Session-budget parity failed -- see gate G4 (F-06).",
    "ratio_up_pkts": "Session-budget parity failed -- see gate G4 (F-06).",
    "iat_max": "Capture-window duration is acting as a class label -- see gate G4 (F-06).",
}
