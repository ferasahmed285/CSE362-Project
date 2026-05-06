# main.py - Final version with REAL LLM (Ollama)
import threading
import time

from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


def simulate_failure(workers, worker_id=1, delay=5.0):
    def fail_later():
        time.sleep(delay)
        workers[worker_id].simulate_failure()
        print(f"\n[Test] Worker {worker_id} has been KILLED mid-test!\n")
    threading.Thread(target=fail_later, daemon=True).start()


def recover_all_workers(workers):
    for worker in workers:
        worker.recover()
        worker.is_alive = True


def print_worker_metrics(workers, lb):
    print("\n========== GPU WORKER METRICS ==========")
    for worker in workers:
        summary = worker.get_metrics_summary()
        with lb.lock:
            util = lb.worker_stats[worker.id].gpu_utilization
        print(
            f"  GPU-{worker.id} | "
            f"Processed: {summary['total_processed']:>4} | "
            f"Failed: {summary['total_failed']:>2} | "
            f"Avg Latency: {summary['avg_latency_s']:.3f}s | "
            f"P95: {summary['p95_latency_s']:.3f}s | "
            f"GPU Util: {util}%"
        )
    print("========================================\n")


def main():
    # Real LLM (tinyllama via Ollama) takes 3-10s per request
    # 8 workers x capacity 3 = 24 concurrent real LLM slots
    # This is realistic - real GPU servers handle limited concurrent inferences
    OLLAMA_ENDPOINTS = [
        "https://arrived-fully-character-teach.trycloudflare.com",
        "https://tablets-adaptor-livestock-drop.trycloudflare.com",
    ]

    workers = [
        GPUWorker(
            i,
            max_capacity=20,
            enable_batching=False,
            request_timeout=600,
            ollama_url=OLLAMA_ENDPOINTS[i])
        for i in range(len(OLLAMA_ENDPOINTS))]

    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb

    try:
        print("\n===== TEST 1: 10 USERS (Real LLM) =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=10, strategy="least_connections")
        print_worker_metrics(workers, lb)

        print("\n===== TEST 2: 20 USERS (Real LLM) =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=20, strategy="least_connections")
        print_worker_metrics(workers, lb)

        print("\n===== TEST 3: 24 USERS + WORKER FAILURE (Real LLM) =====")
        recover_all_workers(workers)
        simulate_failure(workers, worker_id=1, delay=5.0)
        run_load_test(lb, num_users=24, strategy="least_connections")
        print_worker_metrics(workers, lb)

        print("\n===== STRATEGY TEST 1: ROUND ROBIN =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=24, strategy="round_robin")
        print_worker_metrics(workers, lb)

        print("\n===== STRATEGY TEST 2: LEAST CONNECTIONS =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=24, strategy="least_connections")
        print_worker_metrics(workers, lb)

        print("\n===== STRATEGY TEST 3: LOAD AWARE =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=24, strategy="load_aware")
        print_worker_metrics(workers, lb)

    finally:
        print("\nShutting down...")
        for worker in workers:
            worker.stop()
        lb.stop()
        scheduler.stop()


if __name__ == "__main__":
    main()