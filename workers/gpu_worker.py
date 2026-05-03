import time
import threading
import random

from llm.inference import run_llm
from rag.retriever import retrieve_context


class GPUWorker:
    def __init__(self, id):
        self.id = id
        self.stats = None
        self.active_jobs = 0
        self.is_alive = True

        self._stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor_load,
            daemon=True
        )
        self.monitor_thread.start()

    def set_stats(self, stats_obj):
        """Link the worker to its stats tracking object."""
        self.stats = stats_obj

    def _monitor_load(self):
        """Periodically update GPU utilization and latency."""
        while not self._stop_event.is_set():
            if self.stats:
                simulated_utilization = min(
                    100.0,
                    (self.active_jobs * 25.0) + random.uniform(5.0, 15.0)
                )

                if self.active_jobs == 0:
                    simulated_utilization = random.uniform(1.0, 5.0)

                self.stats.gpu_utilization = simulated_utilization
                self.stats.current_latency = 0.1 + (
                    simulated_utilization / 100.0
                ) * 1.5

            time.sleep(1.0)

    def process(self, request):
        """Process one request using RAG + LLM."""
        if not self.is_alive:
            raise Exception(f"Worker {self.id} is offline")

        self.active_jobs += 1

        try:
            start = time.time()
            print(f"[Worker {self.id}] Processing request {request.id}")

            context = retrieve_context(request.query)
            result = run_llm(request.query, context)

            latency = time.time() - start

            return {
                "id": request.id,
                "result": result,
                "latency": latency,
                "worker_id": self.id
            }

        finally:
            self.active_jobs -= 1

    def ping(self):
        """Health check used by the load balancer."""
        return self.is_alive

    def fail(self):
        """Simulate worker failure."""
        print(f"[Worker {self.id}] FAILED")
        self.is_alive = False

    def recover(self):
        """Simulate worker recovery."""
        print(f"[Worker {self.id}] RECOVERED")
        self.is_alive = True

    def stop(self):
        """Stop background monitor thread."""
        self._stop_event.set()