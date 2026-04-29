from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test

def main():
    # Create GPU workers
    workers = [GPUWorker(i) for i in range(4)] # simulate 4 GPUs
    
    # Load Balancer
    lb = LoadBalancer(workers)
    
    # Scheduler
    scheduler = Scheduler()
    
    # Wire them up properly
    lb.scheduler = scheduler
    
    try:
        # Run simulation against the Load Balancer exactly as standard arch dictates
        run_load_test(lb, num_users=1000)
    finally:
        # Gracefully shutdown resources
        for worker in workers:
            if hasattr(worker, 'stop'):
                worker.stop()
        if hasattr(lb, 'stop'):
            lb.stop()
        if hasattr(scheduler, 'stop'):
            scheduler.stop()

if __name__ == "__main__":
    main()