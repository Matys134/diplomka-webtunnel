#!/usr/bin/env python3
"""
Master Experimental Benchmark Orchestrator:
Executes the end-to-end WebTunnel Traffic Analysis and Defense Benchmark:
1. Docker Testbed Health Check
2. Scaled PCAP Capture across 3 Network Profiles (Broadband, LTE, Lossy)
3. Anti-Leakage Sanitization & Multi-Core Dataset Building
4. Diagnostic Spectral & IAT Distribution Plotting
5. Model Training (XGBoost, 1D-CNN, Flow-Transformer)
6. 5-Fold Stratified Group Cross-Validation
7. Explainable AI (SHAP & Gradient Saliency Maps)
8. Pre- vs. Post-Handshake Ablation Analysis
9. Cross-Profile Domain Generalization Experiment
10. Protocol Countermeasures & Before-vs-After Defense Simulation
11. 2-Tier Cascaded Line-Rate Inspection Pipeline
12. Logarithmic DET Curve Generation
13. Multi-Class Hard Negatives Confusion Breakdown
14. Base Rate Fallacy & Host-Based Aggregation
15. Synchronized LaTeX Tables Export
"""
import os
import sys
import time
import json
import subprocess
import argparse

VENV_PYTHON = "venv/bin/python3"
LOG_FILE = "benchmark_run.log"


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_step(step_name: str, cmd: str):
    print(f"\n{'='*75}")
    print(f"[*] FÁZE: {step_name}")
    print(f"[*] PŘÍKAZ: {cmd}")
    print(f"[*] ČAS: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*75}\n")

    start_t = time.time()
    res = subprocess.run(cmd, shell=True, text=True)
    elapsed = time.time() - start_t

    if res.returncode != 0:
        print(f"\n[CHYBA] Fáze '{step_name}' selhala s návratovým kódem {res.returncode} po {elapsed:.2f}s!")
        sys.exit(res.returncode)
    else:
        print(f"\n[OK] Fáze '{step_name}' úspěšně dokončena za {elapsed:.2f}s ({elapsed/60:.2f} min).")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Master WebTunnel Resilience Benchmark Orchestrator")
    parser.add_argument("--samples-per-profile", type=int, default=100,
                        help="Počet PCAP vzorků na třídu a síťový profil (100 = 1800 celkem)")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Přeskočit sběr PCAPů a spustit pipeline na existujících datech")
    parser.add_argument("--allow-failing-gates", action="store_true",
                        help="DIAGNOSTIC ONLY: continue past a failing build gate. Never use this "
                             "for results that go into the thesis (audit principle P3).")
    args = parser.parse_args()

    sys.stdout = Logger(LOG_FILE)
    sys.stderr = sys.stdout

    total_start = time.time()
    timings = {}

    total_pcaps = 3 * 6 * args.samples_per_profile
    print("\n" + "#"*75)
    print("#  WEBTUNNEL RESILIENCE MASTER BENCHMARK PIPELINE")
    print(f"#  Cíl: {args.samples_per_profile} vzorků/třídu/profil (3 profily x 6 tříd = {total_pcaps} PCAPů)")
    print(f"#  Čas zahájení: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Logování do souboru: {os.path.abspath(LOG_FILE)}")
    print("#"*75 + "\n")

    # 1. Kontrola Docker testbedu
    if not args.skip_capture:
        print("[*] Kontrola Docker Compose testbedu...")
        chk = subprocess.run("docker compose -f 1_testbed/docker-compose.yml ps", shell=True, capture_output=True, text=True)
        if "testbed-client" not in chk.stdout:
            print("[*] Spouštím Docker kontejnery...")
            subprocess.run("docker compose -f 1_testbed/docker-compose.yml up -d", shell=True, check=True)
            time.sleep(4)

        cmd_capture = f"{VENV_PYTHON} 1_testbed/capture/collect_scaled_dataset.py --samples-per-profile {args.samples_per_profile}"
        timings["capture"] = run_step("Sběr 1800 PCAPů v Docker testbedu (Broadband, LTE, Lossy)", cmd_capture)
    else:
        print("[*] Fáze sběru PCAPů přeskočena na vyžádání (--skip-capture).")

    # 2. Datová pipeline
    timings["build_dataset"] = run_step(
        "Flow builder v2.1 (TCP reassembly, strict 5-tuple demux, socket-disjoint split)",
        f"{VENV_PYTHON} 2_data_pipeline/build_dataset.py")

    # ---------------------------------------------------------------------
    # BUILD GATES -- BLOCKING.  Audit principle P3: "a dataset that does not pass the gates
    # never reaches a model".  v2.0 had 15 phases and this was in none of them, so a corpus
    # with G1 and G4 red produced three trained models and six LaTeX tables unobstructed.
    # run_step() exits non-zero on failure, which is exactly the desired behaviour.
    # ---------------------------------------------------------------------
    gate_cmd = (f"{VENV_PYTHON} checks/run_gates.py "
                f"--dataset data/processed/tabular_dataset.npz "
                f"--sequences data/processed/sequence_dataset.npz "
                f"--json data/processed/gates.json")
    if args.allow_failing_gates:
        gate_cmd += " --allow-fail"
        print("\n[!] --allow-failing-gates is set. Results from this run are DIAGNOSTIC ONLY "
              "and must not be quoted in the thesis.\n")
    timings["build_gates"] = run_step("Validační brány G1-G6 (BLOKUJÍCÍ)", gate_cmd)

    timings["inspect_dataset"] = run_step("Generování spektrálních a IAT distribučních grafů", f"{VENV_PYTHON} 2_data_pipeline/inspect_dataset.py")

    # 3. Trénování modelů
    timings["train_xgboost"] = run_step("Trénování XGBoost Baseline modelu s kalibrací prahu", f"{VENV_PYTHON} 3_models/train_xgboost.py")
    timings["train_1d_cnn"] = run_step("Trénování PyTorch 1D-CNN (CUDA) s Focal Loss", f"{VENV_PYTHON} 3_models/train_1d_cnn.py")
    timings["train_transformer"] = run_step("Trénování Flow-Transformeru ([CLS] Token Attention)", f"{VENV_PYTHON} 3_models/train_transformer.py")
    # The zero-parameter baseline: two integer operations, no training. See docs/04-v2-audit.md 4.4.
    timings["lattice_rule"] = run_step(
        "Deterministické pravidlo Tor cell lattice ((L-44) mod 514 == 0)",
        f"{VENV_PYTHON} 3_models/lattice_rule.py --split test")

    # 4. 5-Fold Stratified Group Cross-Validation
    timings["cross_validation"] = run_step("5-Fold Session-Stratifikovaná křížová validace", f"{VENV_PYTHON} 3_models/cross_validate.py")

    # 5. XAI Explainability
    timings["explainability"] = run_step("Explainable AI (SHAP & Input Gradient Saliency)", f"{VENV_PYTHON} 3_models/explain_models.py")

    # 6. Evaluační experimenty
    timings["post_handshake"] = run_step("Pre- vs. Post-Handshake ablace (klientsky Finished, manifest-aware)", f"{VENV_PYTHON} 4_evaluation/evaluate_post_handshake.py")
    timings["cross_profile"] = run_step("Doménový posun a generalizace (Broadband -> LTE & Lossy)", f"{VENV_PYTHON} 4_evaluation/evaluate_cross_profile.py")
    timings["defenses"] = run_step("Obrany na urovni TLS zaznamu: staticky i adaptivni protivnik", f"{VENV_PYTHON} 4_evaluation/evaluate_before_after_defenses.py")
    timings["cascaded_pipeline"] = run_step("2-Tier kaskádová architektura (L1 CPU -> L2 GPU)", f"{VENV_PYTHON} 4_evaluation/evaluate_cascaded_pipeline.py")
    timings["det_curve"] = run_step("Logaritmická DET křivka cenzora", f"{VENV_PYTHON} 4_evaluation/evaluate_det_curve.py")
    timings["confusion_breakdown"] = run_step("Dekompozice chybovosti po třídách a konfuzní matice", f"{VENV_PYTHON} 4_evaluation/evaluate_confusion_matrix.py")
    timings["base_rate_fallacy"] = run_step("Base Rate Fallacy a empiricka LLR agregace podle destinace", f"{VENV_PYTHON} 4_evaluation/evaluate_base_rate_fallacy.py")
    timings["order_shuffle"] = run_step("Kontrolní experiment permutace pořadí (Order-Shuffle Control)", f"{VENV_PYTHON} 4_evaluation/evaluate_order_shuffle.py")
    timings["censor_simulation"] = run_step("Simulace nasazení cenzora na páteřní síti (1M toků, Base-Rate & obrany)", f"{VENV_PYTHON} 4_evaluation/simulate_censor_deployment.py")

    # 7. Export LaTeX tabulek
    timings["latex_tables"] = run_step("Generování synchronizovaných LaTeX tabulek", f"{VENV_PYTHON} 4_evaluation/export_latex_tables.py")

    total_elapsed = time.time() - total_start
    timings["total_pipeline_time_sec"] = total_elapsed

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/master_run_timings.json", "w") as f:
        json.dump(timings, f, indent=4)

    print("\n" + "#"*75)
    print(f"#  KOMPLETNÍ BENCHMARK DOKONČEN ZA {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)!")
    print(f"#  Čas dokončení: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Výstupy uloženy v: data/processed/, 4_evaluation/plots/ a 0_thesis_text/tables/")
    print("#"*75 + "\n")


if __name__ == "__main__":
    main()
