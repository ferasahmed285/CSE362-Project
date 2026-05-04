# main.py - Final combined version
import threading
import time

from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


def simulate_failure(workers, worker_id=1, delay=0.5):
    def fail_later():
        time.sleep(delay)
        workers[worker_id].simulate_failure()
        print(f"\n[Test] Worker {worker_id} has been KILLED mid-test!\n")
    threading.Thread(target=fail_later, daemon=True).start()


def recover_all_workers(workers):
    for worker in workers:
        worker.recover()
        worker.is_alive = True


def main():
    # 8 workers x capacity 200 = 1600 concurrent slots
    # This handles 1000 truly concurrent threads comfortably
    workers = [GPUWorker(i, max_capacity=200) for i in range(8)]

    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb

    try:
        print("\n===== TEST 1: 100 USERS =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=100, strategy="least_connections")

        print("\n===== TEST 2: 500 USERS =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=500, strategy="least_connections")

        print("\n===== TEST 3: 1000 USERS + WORKER FAILURE =====")
        recover_all_workers(workers)
        simulate_failure(workers, worker_id=1, delay=0.5)
        run_load_test(lb, num_users=1000, strategy="least_connections")

        print("\n===== STRATEGY TEST 1: ROUND ROBIN =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=1000, strategy="round_robin")

        print("\n===== STRATEGY TEST 2: LEAST CONNECTIONS =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=1000, strategy="least_connections")

        print("\n===== STRATEGY TEST 3: LOAD AWARE =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=1000, strategy="load_aware")

    finally:
        print("\nShutting down...")
        for worker in workers:
            worker.stop()
        lb.stop()
        scheduler.stop()


if __name__ == "__main__":
    main()
