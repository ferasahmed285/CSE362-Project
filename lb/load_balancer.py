# lb/load_balancer.py
import threading
from common.models import WorkerStats

class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.rr_index = 0
        
        # Initialize tracking stats for each worker
        self.worker_stats = {
            worker.id: WorkerStats(worker_id=worker.id) 
            for worker in workers
        }
        
        # Lock to prevent race conditions during concurrent requests
        self.lock = threading.Lock()

    def get_worker_round_robin(self):
        """Baseline strategy: cycles through workers sequentially."""
        with self.lock:
            worker = self.workers[self.rr_index]
            self.rr_index = (self.rr_index + 1) % len(self.workers)
            return worker

    def get_worker_least_connections(self):
        """Advanced strategy: routes to the worker with the lowest active load."""
        with self.lock:
            alive_stats = [s for s in self.worker_stats.values() if s.is_alive]
            if not alive_stats:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            # Find the worker stat with the absolute minimum active connections
            best_stat = min(alive_stats, key=lambda s: s.active_connections)
            return next(w for w in self.workers if w.id == best_stat.worker_id)

    def get_worker_load_aware(self):
        """Advanced strategy: routes based on active connections (can be expanded with latency history)."""
        with self.lock:
            alive_stats = [s for s in self.worker_stats.values() if s.is_alive]
            if not alive_stats:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            # For this sprint, load-aware heavily weights active connections
            best_stat = min(alive_stats, key=lambda s: s.active_connections)
            return next(w for w in self.workers if w.id == best_stat.worker_id)

    def dispatch(self, request, strategy="least_connections"):
        """Routes the request based on the selected strategy."""
        # 1. Select the worker
        if strategy == "least_connections":
            worker = self.get_worker_least_connections()
        elif strategy == "load_aware":
            worker = self.get_worker_load_aware()
        else:
            worker = self.get_worker_round_robin()
            
        # 2. Increment tracker safely
        with self.lock:
            self.worker_stats[worker.id].active_connections += 1
            
        try:
            # 3. Process the request
            response = worker.process(request)
            return response
        finally:
            # 4. Decrement tracker safely, even if the worker crashes
            with self.lock:
                self.worker_stats[worker.id].active_connections -= 1