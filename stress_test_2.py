# fault_stress_test.py
# ============================================================
# FAULT-TOLERANCE STRESS TEST
#
# What this test does:
# 1. Starts 2 remote GPU workers through Cloudflare/Ollama.
# 2. Starts many concurrent client requests.
# 3. After a delay, simulates failure of one GPU worker.
# 4. Checks whether the system continues using the remaining worker.
#
# How to run:
# python fault_stress_test.py
#
# Optional Windows CMD environment variables:
# set FAULT_NUM_USERS=200
# set FAIL_WORKER_ID=0
# set FAIL_AFTER_SECONDS=10
# set WORKER_CAPACITY=50
# set OLLAMA_ENDPOINTS=https://url1.trycloudflare.com,https://url2.trycloudflare.com
# python fault_stress_test.py
# ============================================================

import os
import time
import threading

from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DEFAULT_ENDPOINTS = [
    "https://clearly-integrate-engines-cooperative.trycloudflare.com",
    "https://windsor-dream-adapters-fighter.trycloudflare.com",
]


def get_endpoints():
    """
    Reads Cloudflare Ollama endpoints from environment variable if provided.
    Otherwise, uses the default endpoints above.
    """
    env_value = os.getenv("OLLAMA_ENDPOINTS", "").strip()

    if env_value:
        endpoints = [
            url.strip().rstrip("/")
            for url in env_value.split(",")
            if url.strip()
        ]
    else:
        endpoints = DEFAULT_ENDPOINTS

    if len(endpoints) < 2:
        raise ValueError(
            "Fault-tolerance test needs at least 2 worker endpoints. "
            "Set OLLAMA_ENDPOINTS=url1,url2"
        )

    return endpoints


OLLAMA_ENDPOINTS = get_endpoints()

NUM_USERS = int(os.getenv("FAULT_NUM_USERS", "1000"))
FAIL_WORKER_ID = int(os.getenv("FAIL_WORKER_ID", "0"))
FAIL_AFTER_SECONDS = float(os.getenv("FAIL_AFTER_SECONDS", "10"))

WORKER_CAPACITY = int(os.getenv("WORKER_CAPACITY", "50"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "7200"))

STRATEGY = os.getenv("LB_STRATEGY", "least_connections")


# ------------------------------------------------------------
# Fault simulation
# ------------------------------------------------------------

def simulate_worker_failure_later(workers, worker_id, delay_seconds):
    """
    Simulates a worker failure while the load test is running.
    This uses the existing GPUWorker.simulate_failure() method.
    """

    def fail_later():
        print(
            f"\n[FaultTest] Worker {worker_id} will fail "
            f"after {delay_seconds} seconds...\n"
        )

        time.sleep(delay_seconds)

        print("\n" + "!" * 70)
        print(
            f"[FaultTest] SIMULATING FAILURE: GPU-{worker_id} is now OFFLINE"
        )
        print("!" * 70 + "\n")

        workers[worker_id].simulate_failure()

    thread = threading.Thread(target=fail_later, daemon=True)
    thread.start()


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------

def print_worker_status(workers, lb, results):
    print("\n========== FINAL WORKER STATUS ==========")

    for worker in workers:
        summary = worker.get_metrics_summary()

        with lb.lock:
            stat = lb.worker_stats[worker.id]
            is_alive = stat.is_alive
            active_connections = stat.active_connections
            gpu_util = stat.gpu_utilization

        completed_by_worker = sum(
            1 for r in results
            if r.get("success") and r.get("worker_id") == worker.id
        )

        print(
            f"GPU-{worker.id} | "
            f"Alive: {is_alive} | "
            f"Endpoint: {worker.ollama_url} | "
            f"Successful Results: {completed_by_worker} | "
            f"Internal Processed: {summary['total_processed']} | "
            f"Internal Failed: {summary['total_failed']} | "
            f"Active Connections: {active_connections} | "
            f"GPU Util: {gpu_util}% | "
            f"Avg Latency: {summary['avg_latency_s']}s | "
            f"P95 Latency: {summary['p95_latency_s']}s"
        )

    print("=========================================\n")


def print_fault_test_summary(results, elapsed):
    total = len(results)
    success = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    success_rate = (len(success) / total * 100) if total else 0

    print("\n========== FAULT-TOLERANCE SUMMARY ==========")
    print(f"Total requests returned: {total}")
    print(f"Successful requests: {len(success)}")
    print(f"Failed requests: {len(failed)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Total wall time: {elapsed:.2f}s")

    worker_distribution = {}
    for r in success:
        worker_id = r.get("worker_id", -1)
        worker_distribution[worker_id] = worker_distribution.get(worker_id, 0) + 1

    print(f"Successful worker distribution: {worker_distribution}")

    if failed:
        print("\nSample failed errors:")
        for item in failed[:10]:
            print(f"Request {item.get('id')}: {item.get('error')}")

    print("============================================\n")


# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------

def main():
    if FAIL_WORKER_ID < 0 or FAIL_WORKER_ID >= len(OLLAMA_ENDPOINTS):
        raise ValueError(
            f"FAIL_WORKER_ID={FAIL_WORKER_ID} is invalid. "
            f"You only have {len(OLLAMA_ENDPOINTS)} workers."
        )

    print("=" * 70)
    print("FAULT-TOLERANCE STRESS TEST")
    print(f"Concurrent requests: {NUM_USERS}")
    print(f"Remote workers: {len(OLLAMA_ENDPOINTS)}")
    print(f"Worker capacity: {WORKER_CAPACITY}")
    print(f"Load-balancing strategy: {STRATEGY}")
    print(f"Worker to fail: GPU-{FAIL_WORKER_ID}")
    print(f"Failure delay: {FAIL_AFTER_SECONDS}s")
    print("=" * 70)

    print("\nConfigured endpoints:")
    for i, endpoint in enumerate(OLLAMA_ENDPOINTS):
        print(f"GPU-{i}: {endpoint}")

    workers = [
        GPUWorker(
            i,
            max_capacity=WORKER_CAPACITY,
            enable_batching=False,
            request_timeout=REQUEST_TIMEOUT,
            ollama_url=OLLAMA_ENDPOINTS[i],
        )
        for i in range(len(OLLAMA_ENDPOINTS))
    ]

    lb = LoadBalancer(workers)

    scheduler = Scheduler(
        task_timeout=REQUEST_TIMEOUT,
        capacity_wait_timeout=REQUEST_TIMEOUT,
        capacity_retry_delay=0.5,
    )

    lb.scheduler = scheduler
    scheduler.lb = lb

    try:
        simulate_worker_failure_later(
            workers,
            worker_id=FAIL_WORKER_ID,
            delay_seconds=FAIL_AFTER_SECONDS,
        )

        print("\n[FaultTest] Starting concurrent load test...\n")

        start = time.time()
        results = run_load_test(
            lb,
            num_users=NUM_USERS,
            strategy=STRATEGY,
        )
        elapsed = time.time() - start

        print_fault_test_summary(results, elapsed)
        print_worker_status(workers, lb, results)

        print("[FaultTest] Expected behavior:")
        print("- One worker should become OFFLINE.")
        print("- Requests should continue on the remaining active worker.")
        print("- Some requests may retry during the failure moment.")
        print("- The system should not completely crash unless all workers fail.")

    finally:
        print("\n[FaultTest] Shutting down...")

        for worker in workers:
            worker.stop()

        if hasattr(lb, "stop"):
            lb.stop()

        scheduler.stop()


if __name__ == "__main__":
    main()