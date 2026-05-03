import threading
import time

from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test


def simulate_failure(workers, worker_id=1, delay=0.2):
    """Simulate a worker failure after a delay."""
    def fail_later():
        time.sleep(delay)
        workers[worker_id].fail()

    thread = threading.Thread(target=fail_later, daemon=True)
    thread.start()


def recover_all_workers(workers):
    """Recover all workers before a new clean test."""
    for worker in workers:
        worker.recover()


def main():
    workers = [GPUWorker(i) for i in range(4)]

    lb = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler

    try:
        print("\n===== TEST 1: 100 USERS =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=100, strategy="least_connections")

        print("\n===== TEST 2: 500 USERS =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=500, strategy="least_connections")

        print("\n===== TEST 3: 1000 USERS + FAILURE =====")
        recover_all_workers(workers)
        simulate_failure(workers, worker_id=1, delay=0.2)
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

        print("\n===== HEAVY STRESS TEST: 1500 USERS =====")
        recover_all_workers(workers)
        run_load_test(lb, num_users=1500, strategy="least_connections")

    finally:
        print("\nShutting down system...")

        for worker in workers:
            worker.stop()

        lb.stop()
        scheduler.stop()


if __name__ == "__main__":
    main()