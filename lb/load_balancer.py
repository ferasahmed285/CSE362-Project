# lb/load_balancer.py
import threading
from common.models import WorkerStats

class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.rr_index = 0
        self.scheduler = None # Set externally to prevent circular imports
        
        # Initialize tracking stats for each worker
        self.worker_stats = {
            worker.id: WorkerStats(worker_id=worker.id) 
            for worker in workers
        }
        
        # Give each worker its respective stats object
        for worker in workers:
            if hasattr(worker, 'set_stats'):
                worker.set_stats(self.worker_stats[worker.id])
        
        # Lock to prevent race conditions during concurrent requests
        self.lock = threading.Lock()

    def increment_connection(self, worker_id):
        """Called by the Scheduler exactly when a task starts."""
        with self.lock:
            self.worker_stats[worker_id].active_connections += 1

    def decrement_connection(self, worker_id):
        """Called by the Scheduler when a task finishes, crashes, or times out."""
        with self.lock:
            # max(0) prevents the counter from dipping into negatives during edge cases
            self.worker_stats[worker_id].active_connections = max(
                0, self.worker_stats[worker_id].active_connections - 1
            )

    def get_worker_round_robin(self):
        """Baseline strategy: cycles through alive workers sequentially."""
        with self.lock:
            alive_workers = [w for w in self.workers if self.worker_stats[w.id].is_alive]
            if not alive_workers:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            self.rr_index = self.rr_index % len(alive_workers)
            worker = alive_workers[self.rr_index]
            self.rr_index = (self.rr_index + 1) % len(alive_workers)
            return worker

    def get_worker_least_connections(self):
        """Advanced strategy: routes to the worker with the lowest active load."""
        with self.lock:
            alive_stats = [s for s in self.worker_stats.values() if s.is_alive]
            if not alive_stats:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            best_stat = min(alive_stats, key=lambda s: s.active_connections)
            return next(w for w in self.workers if w.id == best_stat.worker_id)

    def get_worker_load_aware(self):
        """Advanced strategy: Composite scoring using Queue Length and GPU Utils."""
        with self.lock:
            alive_stats = [s for s in self.worker_stats.values() if s.is_alive]
            if not alive_stats:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            W_QUEUE = 1.0
            W_GPU = 0.05 
            
            def calculate_load_score(stat):
                # Fallback to 0 if gpu_utilization isn't added to WorkerStats yet
                gpu_util = getattr(stat, 'gpu_utilization', 0)
                return (stat.active_connections * W_QUEUE) + (gpu_util * W_GPU)
            
            best_stat = min(alive_stats, key=calculate_load_score)
            return next(w for w in self.workers if w.id == best_stat.worker_id)

    def dispatch(self, request, strategy="least_connections"):
        """Hands the request to the Master Node for fault-tolerant execution."""
        if not self.scheduler:
            raise Exception("LoadBalancer has no attached Master/Scheduler!")
            
        return self.scheduler.submit_task(request, strategy=strategy)