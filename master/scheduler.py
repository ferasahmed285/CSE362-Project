import queue
import threading
import time
from common.models import RequestStatus, InternalTaskMessage

class Scheduler:
    def __init__(self, lb=None):
        self.task_queue = queue.Queue()
        self.results = {}
        self.lb = lb
        
        self._stop_event = threading.Event()
        
        # Start health check loop
        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_thread.start()
        
        # Start task dispatcher
        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatcher_thread.start()

    def set_lb(self, lb):
        self.lb = lb

    def _health_check_loop(self):
        """Pings every worker periodically (every 2.5 seconds) to update their alive status."""
        while not self._stop_event.is_set():
            if not self.lb:
                time.sleep(1.0)
                continue
                
            for worker in self.lb.workers:
                try:
                    is_alive = hasattr(worker, 'ping') and worker.ping()
                    with self.lb.lock:
                        was_alive = self.lb.worker_stats[worker.id].is_alive
                        self.lb.worker_stats[worker.id].is_alive = bool(is_alive)
                        
                        if not was_alive and is_alive:
                            print(f"[Master] Worker {worker.id} has recovered and is back online!")
                        elif was_alive and not is_alive:
                            print(f"[Master] Worker {worker.id} failed ping. Marked offline.")
                except Exception:
                    # If ping fails or times out, mark as dead
                    with self.lb.lock:
                        if self.lb.worker_stats[worker.id].is_alive:
                            print(f"[Master] Worker {worker.id} timed out. Marked offline.")
                        self.lb.worker_stats[worker.id].is_alive = False
            time.sleep(2.5)

    def submit_task(self, request, strategy="least_connections"):
        """Called by the LB to hand off the task and block until completion."""
        request.status = RequestStatus.PENDING
        event = threading.Event()
        self.results[request.id] = {
            "event": event, 
            "response": None, 
            "request": request, 
            "strategy": strategy
        }
        self.task_queue.put(request.id)
        
        # Wait for the Master to complete the task (including any retries)
        event.wait()
        return self.results[request.id]["response"]

    def _dispatch_loop(self):
        """Pulls from the global queue and hands it directly to the assignment logic."""
        while not self._stop_event.is_set():
            try:
                req_id = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            task_info = self.results[req_id]
            request = task_info["request"]
            strategy = task_info["strategy"]
            
            # Spin up a thread to process the worker assignment and retries without blocking the master queue!
            t = threading.Thread(target=self._run_worker_task_with_retries, args=(req_id, request, strategy), daemon=True)
            t.start()

    def _run_worker_task_with_retries(self, req_id, request, strategy):
        """Handles task assignment, execution, and fault-tolerant reassignment."""
        retries = request.retries
        max_retries = request.max_retries
        response = None
        
        while retries < max_retries:
            # 1. Ask LB for a worker based on strategy
            try:
                if strategy == "least_connections":
                    worker = self.lb.get_worker_least_connections()
                elif strategy == "load_aware":
                    worker = self.lb.get_worker_load_aware()
                else:
                    worker = self.lb.get_worker_round_robin()
            except Exception as e:
                response = {"id": req_id, "error": str(e), "latency": 0}
                break

            # 2. Increment active connections
            with self.lb.lock:
                self.lb.worker_stats[worker.id].active_connections += 1

            try:
                request.status = RequestStatus.PROCESSING
                
                # Construct internal message (for logging/debugging or future worker changes)
                task_msg = InternalTaskMessage(
                    request=request,
                    worker_id=worker.id,
                    failure_flags=["retried"] if retries > 0 else [],
                    retry_count=retries
                )
                
                # Execute on worker
                worker_response = worker.process(request)
                
                request.status = RequestStatus.COMPLETED
                response = worker_response
                break  # Success, exit retry loop
                
            except Exception as e:
                # 3. Fault Detection & Task Reassignment
                print(f"[Master] Error on Worker {worker.id}: {e}. Marking as offline and reassigning...")
                with self.lb.lock:
                    self.lb.worker_stats[worker.id].is_alive = False
                retries += 1
                request.retries = retries
            finally:
                # 4. Decrement active connections
                with self.lb.lock:
                    self.lb.worker_stats[worker.id].active_connections -= 1

        # Final failure check
        if not response:
            request.status = RequestStatus.FAILED
            response = {"id": req_id, "error": f"Failed to process request {request.id} after {max_retries} attempts.", "latency": 0}

        # Save result and signal completion
        self.results[req_id]["response"] = response
        self.results[req_id]["event"].set()

    def stop(self):
        self._stop_event.set()