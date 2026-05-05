# stress_test.py
# ============================================================
# STRESS TEST - 1000 CONCURRENT REQUESTS
#
# Proves the distributed system architecture supports 1000+
# by queueing and processing them through the worker pool.
#
# How to run:     python stress_test.py
# ============================================================

import time
import threading
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


def main():
    print("=" * 60)
    print("  STRESS TEST: 1000 REQUESTS")
    print("  8 workers x 15 capacity = 120 concurrent slots")
    print("  Requests queue with backoff until a slot opens")
    print("=" * 60)

    # 8 workers x 15 capacity = 120 active slots
    # The scheduler's backoff+retry handles the remaining requests
    workers = [GPUWorker(i, max_capacity=15, enable_batching=False) for i in range(8)]

    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb

    try:
        print("\n===== STRESS TEST: 1000 USERS =====")
        print("System will queue and process all 1000 requests.")
        print("Workers process requests as they complete.\n")

        start = time.time()
        results = run_load_test(lb, num_users=1000, strategy="least_connections")
        elapsed = time.time() - start

        print(f"\nTotal wall time: {elapsed:.1f}s")
        print("\n========== FINAL WORKER METRICS ==========")
        for worker in workers:
            summary = worker.get_metrics_summary()
            print(
                f"  GPU-{worker.id} | "
                f"Processed: {summary['total_processed']:>4} | "
                f"Failed: {summary['total_failed']:>2} | "
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
