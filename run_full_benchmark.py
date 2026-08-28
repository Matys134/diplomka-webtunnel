#!/usr/bin/env python3
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
    print(f"\n{'='*70}")
    print(f"[*] STEP: {step_name}")
    print(f"[*] COMMAND: {cmd}")
    print(f"[*] TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    start_t = time.time()
    res = subprocess.run(cmd, shell=True, text=True)
    elapsed = time.time() - start_t
    
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode} after {elapsed:.2f}s!")
        sys.exit(res.returncode)
    else:
        print(f"\n[OK] Step '{step_name}' completed successfully in {elapsed:.2f}s ({elapsed/60:.2f} min).")
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="Master Experimental Benchmark Orchestrator")
    parser.add_argument("--samples-per-profile", type=int, default=100, 
                        help="Number of PCAP samples per traffic class per network profile (e.g. 100 = 1500 total PCAPs, 200 = 3000 PCAPs)")
    parser.add_argument("--skip-capture", action="store_true", 
                        help="Skip PCAP capture and run training/eval on existing raw PCAPs")
    args = parser.parse_args()

    # Setup automatic tee logging
    sys.stdout = Logger(LOG_FILE)
    sys.stderr = sys.stdout

    total_start = time.time()
    timings = {}

    total_pcaps = 3 * 6 * args.samples_per_profile
    print("\n" + "#"*70)
    print("#  WEBTUNNEL RESILIENCE MASTER BENCHMARK PIPELINE")
    print(f"#  Target: {args.samples_per_profile} samples/class/profile (3 profiles x 6 classes = {total_pcaps} PCAPs)")
    print(f"#  Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Logging all output to: {os.path.abspath(LOG_FILE)}")
    print("#"*70 + "\n")

    # 0. Health check
    print("[*] Checking Docker testbed health...")
    chk = subprocess.run("docker compose -f 1_testbed/docker-compose.yml ps", shell=True, capture_output=True, text=True)
    if "testbed-client" not in chk.stdout:
        print("[*] Starting Docker Compose containers...")
        subprocess.run("docker compose -f 1_testbed/docker-compose.yml up -d", shell=True, check=True)
        time.sleep(5)

    # 1. Capture Dataset (unless skipped)
    if not args.skip_capture:
        cmd_capture = f"{VENV_PYTHON} 1_testbed/capture/collect_scaled_dataset.py --samples-per-profile {args.samples_per_profile}"
        timings["capture"] = run_step("Scaled PCAP Capture (Broadband, LTE, Lossy)", cmd_capture)
    else:
        print("[*] Skipping capture as requested.")

    # 2. Sanitization & Feature Extraction
    cmd_build = f"{VENV_PYTHON} 2_data_pipeline/build_dataset.py"
    timings["build_dataset"] = run_step("Anti-Leakage Sanitization & Multi-Core Dataset Building", cmd_build)

    # 3. Inspection & Spectral Plots
    cmd_inspect = f"{VENV_PYTHON} 2_data_pipeline/inspect_dataset.py"
    timings["inspect_dataset"] = run_step("Cell Quantization & IAT Distribution Plotting", cmd_inspect)

    # 4. Model Training (XGBoost, 1D-CNN, Flow-Transformer)
    timings["train_xgboost"] = run_step("XGBoost Baseline Training & Profiling", f"{VENV_PYTHON} 3_models/train_xgboost.py")
    timings["train_1d_cnn"] = run_step("PyTorch 1D-CNN (CUDA) Training with Focal Loss", f"{VENV_PYTHON} 3_models/train_1d_cnn.py")
    timings["train_transformer"] = run_step("PyTorch Flow-Transformer (CUDA) Training", f"{VENV_PYTHON} 3_models/train_transformer.py")

    # 5. 5-Fold Stratified Cross-Validation
    timings["cross_validation"] = run_step("5-Fold Stratified Cross-Validation", f"{VENV_PYTHON} 3_models/cross_validate.py")

    # 6. Model Explainability & Saliency
    timings["explainability"] = run_step("Explainable AI (SHAP & Saliency Maps)", f"{VENV_PYTHON} 3_models/explain_models.py")

    # 7. Post-Handshake, Cross-Profile & Defense Simulation
    timings["post_handshake"] = run_step("Pre- vs Post-Handshake Analysis", f"{VENV_PYTHON} 4_evaluation/evaluate_post_handshake.py")
    timings["cross_profile"] = run_step("Cross-Profile Domain Generalization Evaluation", f"{VENV_PYTHON} 4_evaluation/evaluate_cross_profile.py")
    timings["defense_sim"] = run_step("Advanced Defenses & Before-vs-After Simulation", f"{VENV_PYTHON} 4_evaluation/evaluate_before_after_defenses.py")

    # 8. Base Rate Fallacy, Cascaded Pipeline, DET Curve & Class Breakdown
    timings["cascaded_pipeline"] = run_step("2-Tier Cascaded Classification Pipeline Evaluation", f"{VENV_PYTHON} 4_evaluation/evaluate_cascaded_pipeline.py")
    timings["det_curve"] = run_step("Logarithmic DET Curve Generation", f"{VENV_PYTHON} 4_evaluation/evaluate_det_curve.py")
    timings["confusion_breakdown"] = run_step("Multi-Class Confusion Matrix & Class Breakdown", f"{VENV_PYTHON} 4_evaluation/evaluate_confusion_matrix.py")
    timings["evaluation"] = run_step("Base Rate Fallacy & Host-Based Aggregation Evaluation", f"{VENV_PYTHON} 4_evaluation/evaluate_base_rate_fallacy.py")

    # 9. LaTeX Table Export
    timings["latex_export"] = run_step("LaTeX Table Generation", f"{VENV_PYTHON} 4_evaluation/export_latex_tables.py")

    total_elapsed = time.time() - total_start
    timings["total_pipeline_time_sec"] = total_elapsed

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/master_run_timings.json", "w") as f:
        json.dump(timings, f, indent=4)

    print("\n" + "#"*70)
    print(f"#  ALL BENCHMARK PHASES COMPLETED SUCCESSFULLY in {total_elapsed/60:.2f} minutes!")
    print(f"#  Finish Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Artifacts saved in data/processed/, 4_evaluation/plots/ and 0_thesis_text/tables/")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()
