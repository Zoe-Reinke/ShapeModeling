#!/usr/bin/env python3
"""
Background resource logger started by `template.sh`:

    python3 monitor_resources.py LOG_CSV INTERVAL_S &

Appends one CSV row every INTERVAL_S seconds with CPU %, RAM (GB) of the
parent job's process tree, and -- if `nvidia-smi` is available -- GPU
utilization and memory. Runs until killed (the template traps EXIT).
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time

import psutil


def _gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        if out:
            util, mem = out[0].split(",")
            return float(util), float(mem)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None, None


def main(argv):
    path, interval = argv[1], float(argv[2])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    parent = psutil.Process(os.getppid())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "cpu_percent", "rss_gb", "gpu_util_percent", "gpu_mem_mb"])
        while True:
            procs = [parent] + parent.children(recursive=True)
            cpu, rss = 0.0, 0
            for p in procs:
                try:
                    cpu += p.cpu_percent(interval=None)
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            gu, gm = _gpu()
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), round(cpu, 1), round(rss / 1e9, 3), gu, gm])
            f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main(sys.argv)
