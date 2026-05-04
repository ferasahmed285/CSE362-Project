# master/scheduler.py
import queue
import threading
import time
from common.models import RequestStatus, InternalTaskMessage

class Scheduler:
    def __init__(self, lb=None):
        self.task_queue = queue.Queue()
        self.results    = {}
        self.lb         = lb
        self._stop_event = threading.Event()

        threading.Thread(target=self._health_check_loop, daemon=True).start()
        threading.Thread(target=self._dispatch_loop,     daemon=True).start()

    def set_lb(self, lb):
        self.lb = lb

    def _health_check_loop(self):
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
                            print(f"[Master] Worker {worker.id} recovered and is back ONLINE")
                        elif was_alive and not is_alive:
                            print(f"[Master] Worker {worker.id} failed ping. Marked OFFLINE")
                except Exception:
                    with self.lb.lock:
                        if self.lb.worker_stats[worker.id].is_alive:
                            print(f"[Master] Worker {worker.id} timed out. Marked OFFLINE")
                        self.lb.worker_stats[worker.id].is_alive = False
            time.sleep(2.5)

    def submit_task(self, request, strategy="least_connections"):
        request.status = RequestStatus.PENDING
        event = threading.Event()
        self.results[request.id] = {
            "event":    event,
            "response": None,
            "request":  request,
            "strategy": strategy
        }
        self.task_queue.put(request.id)
        event.wait()
        return self.results[request.id]["response"]

    def _dispatch_loop(self):
        while not self._stop_event.is_set():
            try:
                req_id = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            task_info = self.results[req_id]
            threading.Thread(
                target=self._run_worker_task_with_retries,
                args=(req_id, task_info["request"], task_info["strategy"]),
                daemon=True
            ).start()

    def _run_worker_task_with_retries(self, req_id, request, strategy):
        retries     = request.retries
        max_retries = request.max_retries
        response    = None

        while retries < max_retries:
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

            # FIX 3: Scheduler does NOT touch active_connections.
            # The worker manages its own active_jobs and updates stats directly.
            # This removes the double-counting bug.

            try:
                request.status = RequestStatus.PROCESSING
                task_msg = InternalTaskMessage(
                    request       = request,
                    worker_id     = worker.id,
                    failure_flags = ["retried"] if retries > 0 else [],
                    retry_count   = retries
                )
                worker_response = worker.process(request)
                request.status  = RequestStatus.COMPLETED
                response        = worker_response
                break

            except Exception as e:
                err = str(e)
                if "at capacity" in err or "capacity" in err.lower():
                    print(f"[Master] Worker {worker.id} full, retrying on another...")
                else:
                    print(f"[Master] Worker {worker.id} error: {e}. Marking OFFLINE.")
                    with self.lb.lock:
                        self.lb.worker_stats[worker.id].is_alive = False
                retries += 1
                request.retries = retries

        if not response:
            request.status = RequestStatus.FAILED
            response = {
                "id":    req_id,
                "error": f"Failed after {max_retries} attempts.",
                "latency": 0
            }

        self.results[req_id]["response"] = response
        self.results[req_id]["event"].set()

    def stop(self):
        self._stop_event.set()
