# master/scheduler.py
import queue
import threading
import time
import random
from common.models import RequestStatus, InternalTaskMessage


class Scheduler:
    def __init__(
        self,
        lb=None,
        task_timeout=300,
        capacity_wait_timeout=300,
        capacity_retry_delay=0.5,
    ):
        self.task_queue = queue.Queue()
        self.results = {}
        self.results_lock = threading.Lock()
        self.lb = lb

        self.task_timeout = task_timeout
        self.capacity_wait_timeout = capacity_wait_timeout
        self.capacity_retry_delay = capacity_retry_delay

        self._stop_event = threading.Event()

        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_thread.start()

        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatcher_thread.start()

    def set_lb(self, lb):
        self.lb = lb

    def _health_check_loop(self):
        """Pings every worker periodically to update their alive status."""
        while not self._stop_event.is_set():
            if not self.lb:
                time.sleep(1.0)
                continue

            for worker in self.lb.workers:
                try:
                    is_alive = hasattr(worker, "ping") and worker.ping()
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

            time.sleep(15.0)

    def submit_task(self, request, strategy="least_connections"):
        """Called by the LB to hand off the task and block until completion."""
        request.status = RequestStatus.PENDING
        event = threading.Event()

        with self.results_lock:
            self.results[request.id] = {
                "event": event,
                "response": None,
                "request": request,
                "strategy": strategy,
            }

        self.task_queue.put(request.id)

        if not event.wait(timeout=self.task_timeout):
            with self.results_lock:
                self.results.pop(request.id, None)

            return {
                "id": request.id,
                "error": f"Scheduler timeout ({self.task_timeout}s)",
                "latency": 0,
            }

        with self.results_lock:
            entry = self.results.pop(request.id, None)

        response = entry["response"] if entry else None

        if response is None:
            return {"id": request.id, "error": "No response from worker", "latency": 0}

        return response

    def _dispatch_loop(self):
        """Pulls from the global queue and hands it directly to the assignment logic."""
        while not self._stop_event.is_set():
            try:
                req_id = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self.results_lock:
                task_info = self.results.get(req_id)

            if task_info is None:
                self.task_queue.task_done()
                continue

            threading.Thread(
                target=self._run_worker_task_with_retries,
                args=(req_id, task_info["request"], task_info["strategy"]),
                daemon=True,
            ).start()

            self.task_queue.task_done()

    def _select_worker(self, strategy):
        if strategy == "least_connections":
            return self.lb.get_worker_least_connections()
        if strategy == "load_aware":
            return self.lb.get_worker_load_aware()
        return self.lb.get_worker_round_robin()

    @staticmethod
    def _is_capacity_error(error_message):
        msg = error_message.lower()
        return "capacity" in msg or "full" in msg or "busy" in msg or "timeout" in msg

    def _run_worker_task_with_retries(self, req_id, request, strategy):
        """Handles task assignment, execution, and fault-tolerant reassignment."""
        retries = 0
        response = None
        deadline = time.time() + self.capacity_wait_timeout

        while time.time() < deadline:
            with self.results_lock:
                if req_id not in self.results:
                    return

            try:
                worker = self._select_worker(strategy)

            except Exception as e:
                response = {"id": req_id, "error": str(e), "latency": 0}
                break

            try:
                request.status = RequestStatus.PROCESSING

                InternalTaskMessage(
                    request=request,
                    worker_id=worker.id,
                    failure_flags=["retried"] if retries > 0 else [],
                    retry_count=retries,
                )

                worker_response = worker.process(request)
                request.status = RequestStatus.COMPLETED
                response = worker_response
                break

            except Exception as e:
                err = str(e)

                if self._is_capacity_error(err):
                    backoff = min(
                        3.0,
                        self.capacity_retry_delay * (2 ** min(retries, 5)),
                    ) + random.uniform(0, 0.3)
                    time.sleep(backoff)

                else:
                    print(f"[Master] Worker {worker.id} error: {e}. Marking OFFLINE.")

                    with self.lb.lock:
                        self.lb.worker_stats[worker.id].is_alive = False

                    time.sleep(self.capacity_retry_delay)

                retries += 1
                request.retries = retries

        if not response:
            request.status = RequestStatus.FAILED
            response = {
                "id": req_id,
                "error": f"Failed after {retries} retries (deadline expired after {self.capacity_wait_timeout}s).",
                "latency": 0,
            }

        with self.results_lock:
            entry = self.results.get(req_id)
            if entry:
                entry["response"] = response
                entry["event"].set()

    def stop(self):
        self._stop_event.set()