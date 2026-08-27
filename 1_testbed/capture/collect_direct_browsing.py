#!/usr/bin/env python3
import os
import time
import subprocess
import argparse
import random

COMPOSE_FILE = "1_testbed/docker-compose.yml"
RAW_PCAP_DIR = "data/raw_pcap"
NETEM_PROFILES = ["broadband", "lte", "lossy"]

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def apply_netem(profile):
    print(f"\n>>> [tc-netem] Switching network profile to: {profile.upper()} <<<")
    cmd = f"docker compose -f {COMPOSE_FILE} exec client /app/netem_profiles.sh {profile}"
    res = run_cmd(cmd)
    print(res.stdout.strip())
    time.sleep(1)

def capture_session(class_name, traffic_type, profile, sample_id):
    pcap_filename = f"{class_name}_{profile}_{sample_id:04d}.pcap"
    container_pcap_path = f"/data/raw_pcap/{pcap_filename}"
    host_pcap_path = os.path.join(RAW_PCAP_DIR, pcap_filename)
    
    if os.path.exists(host_pcap_path) and os.path.getsize(host_pcap_path) > 500:
        return True
        
    tcpdump_cmd = f"docker compose -f {COMPOSE_FILE} exec -d client tcpdump -i eth0 -s 0 -w {container_pcap_path}"
    run_cmd(tcpdump_cmd)
    time.sleep(0.4)
    
    gen_cmd = f"docker compose -f {COMPOSE_FILE} exec client python3 traffic_generator.py --mode {traffic_type}"
    run_cmd(gen_cmd)
    time.sleep(0.6)
    
    stop_cmd = f"docker compose -f {COMPOSE_FILE} exec client pkill -f tcpdump"
    run_cmd(stop_cmd)
    time.sleep(0.3)
    
    if os.path.exists(host_pcap_path) and os.path.getsize(host_pcap_path) > 500:
        print(f"  [{profile.upper()}] [{sample_id:04d}] OK: {pcap_filename} ({os.path.getsize(host_pcap_path)} B)")
        return True
    else:
        sz = os.path.getsize(host_pcap_path) if os.path.exists(host_pcap_path) else 0
        print(f"  [{profile.upper()}] [{sample_id:04d}] WARN: size={sz} B")
        return False

def main():
    parser = argparse.ArgumentParser(description="Direct Web Browsing Dataset Collector")
    parser.add_argument("--samples-per-profile", type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(RAW_PCAP_DIR, exist_ok=True)
    print(f"=== Starting Direct Browsing Collection: {len(NETEM_PROFILES)} profiles x {args.samples_per_profile} samples ===")
    
    for profile in NETEM_PROFILES:
        apply_netem(profile)
        print(f"\n--- Generating class: direct_web_browsing under {profile} ---")
        for i in range(1, args.samples_per_profile + 1):
            capture_session("direct_web_browsing", "direct_browsing", profile, i)
            time.sleep(random.uniform(0.1, 0.3))
            
    apply_netem("reset")
    print(f"\n=== [DONE] Direct Browsing collection completed! ===")

if __name__ == "__main__":
    main()
