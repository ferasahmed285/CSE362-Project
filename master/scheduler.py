import queue
import threading
from common.models import RequestStatus

class Scheduler:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.results = {}
        
        self._stop_event = threading.Event()
        self.dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatcher_thread.start()

    def enqueue_task(self, request, worker_selected):
        """Called by the LB to hand off the task non-blocking."""
        request.status = RequestStatus.PENDING
        event = threading.Event()
        # Store full task info
        self.results[request.id] = {"event": event, "response": None, "worker": worker_selected, "request": request}
        self.task_queue.put(request.id)
        
        return event

    def get_result(self, request_id):
        return self.results[request_id]["response"]

    def _dispatch_loop(self):
        """Pulls from the global queue and hands it directly to the designated worker."""
        while not self._stop_event.is_set():
            try:
                req_id = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            task_info = self.results[req_id]
            worker = task_info["worker"]
            request = task_info["request"]
            
            # Spin up a thread to process the worker assignment without blocking the master queue!
            t = threading.Thread(target=self._run_worker_task, args=(req_id, worker, request), daemon=True)
            t.start()

    def _run_worker_task(self, req_id, worker, request):
        try:
            request.status = RequestStatus.PROCESSING
            response = worker.process(request)
            request.status = RequestStatus.COMPLETED
            self.results[req_id]["response"] = response
        except Exception as e:
            request.status = RequestStatus.FAILED
            self.results[req_id]["response"] = {"id": req_id, "error": str(e), "latency": 0}
        finally:
            # Signal that the response is ready
            self.results[req_id]["event"].set()

    def stop(self):
        self._stop_event.set()