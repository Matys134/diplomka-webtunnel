"""Build gates for the WebTunnel v2 dataset.

A dataset that does not pass these never reaches a model. Each gate maps to a finding
from the September 2026 audit; see docs/01-audit-findings.md.

    G1  stack_parity     ClientHello identical across classes            F-02, F-03
    G2  tripwire         no unexplained single-feature separation        F-09
    G3  null_controls    label-shuffle and same-generator collapse       F-09
    G4  budget_parity    volume/duration not class-informative           F-06
    G5  split_integrity  no conn_id in two splits; groups aligned        F-01, F-11
    G6  provenance       every flow matches its manifest                 F-04, F-08

Run them all with:  python3 project/checks/run_gates.py --help
"""

GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
