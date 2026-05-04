# lb/load_balancer.py
import threading
from common.models import WorkerStats

class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.rr_index = 0
        self.scheduler = None # Will be set externally to decouple circular dependencies
        
        # Initialize tracking stats for each worker
        self.worker_stats = {
            worker.id: WorkerStats(worker_id=worker.id) 
            for worker in workers
        }
        
        # Give each worker its respective stats object so it can update its simulated metrics
        for worker in workers:
            if hasattr(worker, 'set_stats'):
                worker.set_stats(self.worker_stats[worker.id])
        
        # Lock to prevent race conditions during concurrent requests
        self.lock = threading.Lock()
        
        # NOTE: The health check loop has been moved to the Master Node (Scheduler)
        # to properly separate Routing from Orchestration & Resilience.

    def stop(self):
        # Stop signal removed as LB no longer runs threads natively.
        pass

    def get_worker_round_robin(self):
        """Baseline strategy: cycles through alive workers sequentially."""
        with self.lock:
            alive_workers = [w for w in self.workers if self.worker_stats[w.id].is_alive]
            if not alive_workers:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            # Ensure index doesn't go out of bounds if workers drop
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
            
            # Find the worker stat with the absolute minimum active connections
            best_stat = min(alive_stats, key=lambda s: s.active_connections)
            return next(w for w in self.workers if w.id == best_stat.worker_id)

    def get_worker_load_aware(self):
        """
        Advanced strategy (100% Goal): Adaptive Weighted Load-Aware Routing.
        
        Instead of basic Round Robin or just Least Connections (which acts as Join-the-Shortest-Queue),
        this algorithm calculates a composite 'load score'. It evaluates BOTH the current queue length 
        (active_connections) AND the physical hardware load (gpu_utilization).
        
        Wider Reading / Theory Application: 
        While "Join-the-Shortest-Queue" (JSQ) is effective for homogeneous tasks, AI/LLM inference 
        tasks have high variance in execution time. A node with a short queue might still be 
        bottlenecked by high thermal GPU usage from a previous heavy query. By combining queue length 
        with physical GPU utilization metrics, this algorithm avoids the "herd behavior" routing problem.
        """
        with self.lock:
            alive_stats = [s for s in self.worker_stats.values() if s.is_alive]
            if not alive_stats:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            # Algorithm Weights:
            # Active connections (Queue length) are heavily weighted because LLM inference is highly sequential.
            # GPU Utilization is factored in to break ties and prevent routing to throttling nodes.
            W_QUEUE = 1.0
            W_GPU = 0.05  # Scale down percentage (0-100) to match queue length scale
            
            def calculate_load_score(stat):
                return (stat.active_connections * W_QUEUE) + (stat.gpu_utilization * W_GPU)
            
            # Select the node with the lowest composite load score
            best_stat = min(alive_stats, key=calculate_load_score)
            return next(w for w in self.workers if w.id == best_stat.worker_id)

    def dispatch(self, request, strategy="least_connections"):
        """Entry point for requests. Defer execution and retries to the Master Node."""
        if not self.scheduler:
            raise Exception("LoadBalancer has no attached Master/Scheduler!")
            
        # The Master Node now handles fault detection and task reassignment.
        # It will call get_worker_* methods to fetch a node based on the given strategy.
        return self.scheduler.submit_task(request, strategy=strategy)