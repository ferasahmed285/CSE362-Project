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
        self._stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_load, daemon=True)
        self.monitor_thread.start()

    def set_stats(self, stats_obj):
        """Link the worker to its stats tracking object."""
        self.stats = stats_obj

    def _monitor_load(self):
        """Periodically update gpu_utilization and current_latency based on simulated load."""
        while not self._stop_event.is_set():
            if self.stats:
                # Simulate metrics: base + random variance, scaled by active jobs
                simulated_utilization = min(100.0, (self.active_jobs * 25.0) + random.uniform(5.0, 15.0))
                if self.active_jobs == 0:
                    simulated_utilization = random.uniform(1.0, 5.0) # idle load
                    
                self.stats.gpu_utilization = simulated_utilization
                self.stats.current_latency = 0.1 + (simulated_utilization / 100.0) * 1.5
                
            time.sleep(1.0) # Update every 1 second

    def process(self, request):
        self.active_jobs += 1
        try:
            start = time.time()
            print(f"[Worker {self.id}] Processing request {request.id}")
            
            # RAG Step
            context = retrieve_context(request.query)
            
            # LLM Step
            result = run_llm(request.query, context)
            
            latency = time.time() - start
            return {
                "id": request.id,
                "result": result,
                "latency": latency
            }
        finally:
            self.active_jobs -= 1

    def stop(self):
        self._stop_event.set()

    def ping(self):
        """Simulate a health check ping."""
        return True