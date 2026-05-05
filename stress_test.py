# stress_test.py
# ============================================================
# OPTIONAL STRESS TEST - 1000 REQUESTS
#
# This runs 1000 requests against the distributed system.
# Because we use a real LLM (tinyllama via Ollama), this will
# take significantly longer than the main demo (potentially
# 30-60 minutes depending on hardware).
#
# The main demo (main.py) uses 10/20/24 users to show the
# system working clearly within a reasonable time.
# This file proves the system ARCHITECTURE supports 1000+
# by queueing and processing them through the worker pool.
#
# How to run:
#     python stress_test.py
# ============================================================

import time
import threading
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


def main():
    print("=" * 60)
    print("  STRESS TEST: 1000 REQUESTS WITH REAL LLM")
    print("  WARNING: This will take a long time (30-60 min)")
    print("  Each request uses real tinyllama inference via Ollama")
    print("=" * 60)

    # 8 workers x 10 capacity = 80 active LLM slots
    # The queue handles the remaining requests
    workers = [GPUWorker(i, max_capacity=10, enable_batching=False) for i in range(8)]

    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb

    try:
        print("\n===== STRESS TEST: 1000 USERS (Real LLM) =====")
        print("System will queue and process all 1000 requests.")
        print("Workers process requests as they complete.\n")

        start = time.time()
        run_load_test(lb, num_users=1000, strategy="least_connections")
        elapsed = time.time() - start

        print(f"\nTotal wall time: {elapsed:.1f}s")
        print("\n========== FINAL WORKER METRICS ==========")
        for worker in workers:
            summary = worker.get_metrics_summary()
            print(
                f"  GPU-{worker.id} | "
                f"Processed: {summary['total_processed']:>4} | "
                f"Avg: {summary['avg_latency_s']:.2f}s"
            )

    finally:
        print("\nShutting down...")
        for worker in workers:
            worker.stop()
        lb.stop()
        scheduler.stop()


if __name__ == "__main__":
    main()
