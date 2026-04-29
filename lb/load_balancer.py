# lb/load_balancer.py
import threading
import time
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
        
        # Start the health check loop thread
        self._stop_event = threading.Event()
        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_thread.start()

    def _health_check_loop(self):
        """Pings every worker periodically (every 2.5 seconds) to update their alive status."""
        while not self._stop_event.is_set():
            for worker in self.workers:
                try:
                    # In a real system, this would be a network request
                    is_alive = hasattr(worker, 'ping') and worker.ping()
                    with self.lock:
                        was_alive = self.worker_stats[worker.id].is_alive
                        self.worker_stats[worker.id].is_alive = bool(is_alive)
                        
                        if not was_alive and is_alive:
                            print(f"[LB] Worker {worker.id} has recovered and is back online!")
                        elif was_alive and not is_alive:
                            print(f"[LB] Worker {worker.id} failed ping. Marked offline.")
                except Exception:
                    # If ping fails or times out, mark as dead
                    with self.lock:
                        if self.worker_stats[worker.id].is_alive:
                            print(f"[LB] Worker {worker.id} timed out. Marked offline.")
                        self.worker_stats[worker.id].is_alive = False
            time.sleep(2.5)

    def stop(self):
        self._stop_event.set()

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
        """Advanced strategy: routes based on reported gpu_utilization."""
        with self.lock:
            alive_stats = [s for s in self.worker_stats.values() if s.is_alive]
            if not alive_stats:
                raise Exception("CRITICAL: All GPU worker nodes are offline!")
            
            # Load-aware routes to the worker with the lowest GPU utilization
            best_stat = min(alive_stats, key=lambda s: s.gpu_utilization)
            return next(w for w in self.workers if w.id == best_stat.worker_id)

    def dispatch(self, request, strategy="least_connections"):
        """Routes the request based on the selected strategy with fault tolerance."""
        max_retries = 3 
        retries = 0
        
        while retries < max_retries:
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
                # 3. Non-blocking hand-off to the Master/Scheduler
                if not self.scheduler:
                    raise Exception("LoadBalancer has no attached Master/Scheduler!")
                    
                # LB immediately schedules the task asynchronously
                event = self.scheduler.enqueue_task(request, worker)
                
                # Client Thread blocks cleanly awaiting the Master, but LB Thread doesn't stall!
                event.wait() 
                response = self.scheduler.get_result(request.id)
                
                if "error" in response:
                    raise Exception(response["error"]) # Trip the retry logic on failure
                
                return response
            except Exception as e:
                # 4. Fault Detection & Task Reassignment
                print(f"[LB] Error on Worker {worker.id}: {e}. Marking as offline and reassigning...")
                with self.lock:
                    self.worker_stats[worker.id].is_alive = False
                retries += 1
            finally:
                # 5. Decrement tracker safely
                with self.lock:
                    self.worker_stats[worker.id].active_connections -= 1
                    
        raise Exception(f"[LB] CRITICAL: Failed to process request {request.id} after {max_retries} attempts.")