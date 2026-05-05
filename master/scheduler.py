# master/scheduler.py
import queue
import threading
import time
import random
from common.models import RequestStatus, InternalTaskMessage

class Scheduler:
    def __init__(self, lb=None):
        self.task_queue = queue.Queue()
        self.results    = {}
        self.results_lock = threading.Lock()  # FIX 5: protect results dict
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
            time.sleep(15.0)  # Real LLM takes 5-10s per request, check less frequently

    def submit_task(self, request, strategy="least_connections"):
        request.status = RequestStatus.PENDING
        event = threading.Event()
        with self.results_lock:  # FIX 5: lock around results access
            self.results[request.id] = {
                "event":    event,
                "response": None,
                "request":  request,
                "strategy": strategy
            }
        self.task_queue.put(request.id)

        # FIX 3: increased timeout to 300s for 1000-user stress tests
        # Real LLM at ~5s/req, 1000 reqs / 80 slots = ~62s minimum
        # Add generous headroom for retries and backoff
        if not event.wait(timeout=300):
            with self.results_lock:
                self.results.pop(request.id, None)  # cleanup memory
            return {"id": request.id, "error": "Scheduler timeout (300s)", "latency": 0}

        with self.results_lock:  # FIX 5: lock around results access
            entry = self.results.pop(request.id, None)
        response = entry["response"] if entry else None

        if response is None:
            return {"id": request.id, "error": "No response from worker", "latency": 0}
        return response

    def _dispatch_loop(self):
        while not self._stop_event.is_set():
            try:
                req_id = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # FIX 5: guard against KeyError if request was already timed out
            with self.results_lock:
                task_info = self.results.get(req_id)
            if task_info is None:
                continue  # request already timed out and was cleaned up

            threading.Thread(
                target=self._run_worker_task_with_retries,
                args=(req_id, task_info["request"], task_info["strategy"]),
                daemon=True
            ).start()

    def _run_worker_task_with_retries(self, req_id, request, strategy):
        retries  = 0
        response = None

        # FIX 1+2: Use a deadline instead of fixed retry count.
        # Keep retrying with exponential backoff until the deadline expires.
        # This ensures requests queue properly instead of failing instantly.
        deadline = time.time() + 240  # 4-minute absolute deadline per task

        while time.time() < deadline:
            # FIX 5: check if the request was already timed out by submit_task
            with self.results_lock:
                if req_id not in self.results:
                    return  # request was cleaned up, no need to continue

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
                    # FIX 1: Exponential backoff with jitter when workers are full.
                    # Instead of burning through retries instantly, we wait for a
                    # slot to free up. Backoff caps at 3s to stay responsive.
                    backoff = min(3.0, 0.1 * (2 ** min(retries, 5))) + random.uniform(0, 0.3)
                    time.sleep(backoff)
                else:
                    print(f"[Master] Worker {worker.id} error: {e}. Marking OFFLINE.")
                    with self.lb.lock:
                        self.lb.worker_stats[worker.id].is_alive = False
                    time.sleep(0.5)  # brief pause before retrying on different worker
                retries += 1
                request.retries = retries

        if not response:
            request.status = RequestStatus.FAILED
            response = {
                "id":    req_id,
                "error": f"Failed after {retries} retries (deadline expired).",
                "latency": 0
            }

        # FIX 5: safely update results
        with self.results_lock:
            entry = self.results.get(req_id)
            if entry:
                entry["response"] = response
                entry["event"].set()

    def stop(self):
        self._stop_event.set()