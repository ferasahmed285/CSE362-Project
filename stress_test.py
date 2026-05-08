# stress_test.py
# ============================================================
# STRESS TEST - 1000 CONCURRENT REQUESTS WITH REMOTE WORKERS
#
# Generates 1000 concurrent client requests and distributes them
# across remote worker laptops running Ollama tinyllama through
# Cloudflare Tunnel.
#
# How to run:
#     python stress_test.py
# ============================================================

import time
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


OLLAMA_ENDPOINTS = [
    "https://clearly-integrate-engines-cooperative.trycloudflare.com",
    "https://windsor-dream-adapters-fighter.trycloudflare.com",
]

NUM_USERS = 1000


def main():
    print("=" * 60)
    print(f"  STRESS TEST: {NUM_USERS} CONCURRENT REQUESTS")
    print(f"  Remote Ollama workers: {len(OLLAMA_ENDPOINTS)}")
    print("  Requests queue with backoff until a worker is available")
    print("  This test may take a long time because inference is real")
    print("=" * 60)

    workers = [
        GPUWorker(
            i,
            max_capacity=50,
            enable_batching=False,
            request_timeout=7200,
            ollama_url=OLLAMA_ENDPOINTS[i],
        )
        for i in range(len(OLLAMA_ENDPOINTS))
    ]

    lb = LoadBalancer(workers)

    scheduler = Scheduler(
        task_timeout=7200,
        capacity_wait_timeout=7200,
        capacity_retry_delay=0.5,
    )

    lb.scheduler = scheduler
    scheduler.lb = lb

    try:
        print(f"\n===== STRESS TEST: {NUM_USERS} USERS =====")
        print("System will queue and process requests through remote workers.\n")

        start = time.time()
        results = run_load_test(lb, num_users=NUM_USERS, strategy="least_connections")
        elapsed = time.time() - start

        success_count = sum(1 for r in results if r.get("success"))
        failed_count = len(results) - success_count

        print(f"\nTotal results returned: {len(results)}")
        print(f"Successful requests:   {success_count}")
        print(f"Failed requests:       {failed_count}")
        print(f"Total wall time:       {elapsed:.1f}s")

        print("\n========== FINAL WORKER METRICS ==========")
        for worker in workers:
            summary = worker.get_metrics_summary()
            
            # Count how many results were handled by this worker
            completed = sum(1 for r in results if r.get("worker_id") == worker.id)

            print(
                f"  GPU-{worker.id} | "
                f"Endpoint: {OLLAMA_ENDPOINTS[worker.id]} | "
                f"Completed Requests: {completed:>4} | "
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