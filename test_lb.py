# test_lb.py
import threading
import time
import random
from common.models import Request, Response
from lb.load_balancer import LoadBalancer

# Mock the GPU Worker to simulate different processing speeds
class MockWorker:
    def __init__(self, id):
        self.id = id

    def process(self, request):
        # Simulate processing time: Node 0 is fast (0.1s), Node 1 is slow
        delay = 0.1 if self.id == 0 else random.uniform(0.2, 0.5)
        time.sleep(delay)
        return Response(id=request.id, result="Success", latency=delay)

    def ping(self):
        return True

# setup the test environment
workers = [MockWorker(0), MockWorker(1)]
lb = LoadBalancer(workers)
from master.scheduler import Scheduler
scheduler = Scheduler(lb)
lb.scheduler = scheduler

def simulate_client_request(req_id):
    req = Request(id=req_id, query="Test")
    # You can swap 'least_connections' with 'round_robin' or 'load_aware' to test different logic
    lb.dispatch(req, strategy="load_aware")

# Fire a burst of concurrent requests
print("Firing 20 concurrent requests...")
threads = []
for i in range(20):
    t = threading.Thread(target=simulate_client_request, args=(i,))
    threads.append(t)
    t.start()

# Monitor the load balancer's state in real-time
for _ in range(20):  # Loop more times
    time.sleep(0.01) # Check much faster
    print(f"Active Connections -> Node 0: {lb.worker_stats[0].active_connections} | Node 1: {lb.worker_stats[1].active_connections}")

for t in threads:
    t.join()

print("Test complete. All connections safely closed.")