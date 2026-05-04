from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test

def main():
    # 8 workers x 50 capacity = 400 concurrent slots
    # More than enough for 1000 users cycling through
    workers = [GPUWorker(i, max_capacity=50) for i in range(8)]

    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler

    try:
        run_load_test(lb, num_users=1000)
    finally:
        for worker in workers:
            if hasattr(worker, 'stop'):
                worker.stop()
        if hasattr(lb, 'stop'):
            lb.stop()
        if hasattr(scheduler, 'stop'):
            scheduler.stop()

if __name__ == "__main__":
    main()