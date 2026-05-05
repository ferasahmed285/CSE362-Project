# workers/gpu_worker.py
import time
import threading
import random
import queue
from collections import deque

from llm.inference import run_llm, run_llm_batch
from rag.retriever import retrieve_context
from common.models import WorkerStats

DEFAULT_MAX_CAPACITY = 10
BATCH_WINDOW_MS      = 50
MAX_BATCH_SIZE       = 8


class GPUWorker:
    def __init__(self, id: int, max_capacity: int = DEFAULT_MAX_CAPACITY, enable_batching: bool = True):
        self.id              = id
        self.max_capacity    = max_capacity
        self.enable_batching = enable_batching
        self.is_alive        = True
        self.active_jobs     = 0
        self.load_lock       = threading.Lock()
        self.stats           = None
        self.results         = {}
        self.results_lock    = threading.Lock()
        self.events          = {}
        self.events_lock     = threading.Lock()
        self.request_queue   = queue.Queue()
        self.total_processed = 0
        self.total_failed    = 0
        self.latency_history = deque(maxlen=500)
        self.metrics_lock    = threading.Lock()

        threading.Thread(target=self._processing_loop, daemon=True).start()
        threading.Thread(target=self._monitor_load,    daemon=True).start()
        print(f"[GPU-{self.id}] Started | capacity={max_capacity} | batching={enable_batching}")

    def set_stats(self, stats_obj):
        self.stats = stats_obj

    def ping(self) -> bool:
        return self.is_alive

    def process(self, request):
        if not self.is_alive:
            raise Exception(f"GPU-{self.id} is down")

        with self.load_lock:
            if self.active_jobs >= self.max_capacity:
                raise Exception(f"GPU-{self.id} at capacity ({self.active_jobs}/{self.max_capacity})")
            self.active_jobs += 1

        # Update stats for LB routing decisions
        if self.stats:
            self.stats.active_connections = self.active_jobs

        event = threading.Event()
        with self.events_lock:
            self.events[request.id] = event

        self.request_queue.put(request)

        if not event.wait(timeout=30.0):
            with self.load_lock:
                self.active_jobs -= 1
            raise Exception(f"GPU-{self.id} timeout on request {request.id}")

        with self.results_lock:
            response = self.results.pop(request.id, None)
        with self.events_lock:
            self.events.pop(request.id, None)

        if response and "error" in response:
            raise Exception(response["error"])

        return response

    def simulate_failure(self):
        self.is_alive = False
        if self.stats:
            self.stats.is_alive = False
        print(f"[GPU-{self.id}] FAILURE SIMULATED - node is DOWN")

    def recover(self):
        self.is_alive = True
        if self.stats:
            self.stats.is_alive = True
        print(f"[GPU-{self.id}] RECOVERED - node is back UP")

    def stop(self):
        self.is_alive = False

    def get_metrics_summary(self) -> dict:
        with self.metrics_lock:
            avg_lat = (sum(self.latency_history) / len(self.latency_history)
                       if self.latency_history else 0.0)
            sorted_lat = sorted(self.latency_history)
            idx     = int(len(sorted_lat) * 0.95)
            p95_lat = sorted_lat[idx] if sorted_lat else 0.0
        return {
            "worker_id":       self.id,
            "total_processed": self.total_processed,
            "total_failed":    self.total_failed,
            "avg_latency_s":   round(avg_lat, 4),
            "p95_latency_s":   round(p95_lat, 4),
        }

    def _processing_loop(self):
        while True:
            # Fix: check is_alive at top of every loop iteration
            if not self.is_alive:
                time.sleep(0.1)
                continue
            try:
                if self.enable_batching:
                    self._process_batch()
                else:
                    self._process_single()
            except Exception as e:
                print(f"[GPU-{self.id}] Loop error: {e}")

    def _process_single(self):
        try:
            request = self.request_queue.get(timeout=1.0)
        except queue.Empty:
            return

        # FIX 4: Check is_alive before processing each request
        if not self.is_alive:
            print(f"[GPU-{self.id}] Skipping req {request.id} - worker is DOWN")
            with self.load_lock:
                self.active_jobs -= 1
            if self.stats:
                self.stats.active_connections = self.active_jobs
            self._deliver(request.id, {"id": request.id, "error": "Worker went down", "latency": 0})
            return

        self._handle_one(request)

    def _process_batch(self):
        try:
            first = self.request_queue.get(timeout=1.0)
        except queue.Empty:
            return

        # FIX 4: Check is_alive before processing the batch
        if not self.is_alive:
            print(f"[GPU-{self.id}] Skipping req {first.id} - worker is DOWN")
            with self.load_lock:
                self.active_jobs -= 1
            if self.stats:
                self.stats.active_connections = self.active_jobs
            self._deliver(first.id, {"id": first.id, "error": "Worker went down", "latency": 0})
            return

        batch    = [first]
        deadline = time.time() + (BATCH_WINDOW_MS / 1000.0)
        while time.time() < deadline and len(batch) < MAX_BATCH_SIZE:
            try:
                batch.append(self.request_queue.get_nowait())
            except queue.Empty:
                break

        if len(batch) == 1:
            self._handle_one(batch[0])
        else:
            self._handle_batch(batch)

    def _handle_one(self, request):
        start = time.time()
        print(f"[GPU-{self.id}] Processing req {request.id}: '{request.query}'")
        try:
            context  = retrieve_context(request.query)
            result   = run_llm(request.query, context)
            latency  = time.time() - start
            response = {"id": request.id, "worker_id": self.id, "result": result, "latency": latency}
            with self.metrics_lock:
                self.total_processed += 1
                self.latency_history.append(latency)
            if self.stats:
                self.stats.total_processed = self.total_processed
                self.stats.current_latency = latency
            print(f"[GPU-{self.id}] Done req {request.id} in {latency:.3f}s")
        except Exception as e:
            latency  = time.time() - start
            response = {"id": request.id, "error": str(e), "latency": latency}
            with self.metrics_lock:
                self.total_failed += 1
        finally:
            with self.load_lock:
                self.active_jobs -= 1
            if self.stats:
                self.stats.active_connections = self.active_jobs
            self._deliver(request.id, response)

    def _handle_batch(self, requests):
        n     = len(requests)
        start = time.time()
        print(f"[GPU-{self.id}] Batch processing {n} requests: {[r.id for r in requests]}")
        try:
            batch_input = [(r.query, retrieve_context(r.query)) for r in requests]
            results     = run_llm_batch(batch_input)
            latency     = time.time() - start
            per_lat     = latency / n
            with self.metrics_lock:
                self.total_processed += n
                self.latency_history.extend([per_lat] * n)
            if self.stats:
                self.stats.total_processed = self.total_processed
                self.stats.current_latency = per_lat
            print(f"[GPU-{self.id}] Batch done: {n} reqs in {latency:.3f}s ({per_lat:.3f}s each)")
            for request, result_text in zip(requests, results):
                self._deliver(request.id, {"id": request.id, "worker_id": self.id, "result": result_text, "latency": per_lat})
        except Exception as e:
            latency = time.time() - start
            for request in requests:
                with self.metrics_lock:
                    self.total_failed += 1
                self._deliver(request.id, {"id": request.id, "error": str(e), "latency": latency})
        finally:
            with self.load_lock:
                self.active_jobs -= n
            if self.stats:
                self.stats.active_connections = self.active_jobs

    def _deliver(self, request_id, response):
        with self.results_lock:
            self.results[request_id] = response
        with self.events_lock:
            event = self.events.get(request_id)
            if event:
                event.set()

    def _monitor_load(self):
        while True:
            time.sleep(1.0)
            if self.stats and self.is_alive:
                with self.load_lock:
                    jobs = self.active_jobs
                # Fix: ensure utilization is always 0-100, never negative
                util = round(max(0.0, min(100.0, (jobs / self.max_capacity) * 100)), 1)
                self.stats.gpu_utilization = util
                self.stats.max_capacity    = self.max_capacity