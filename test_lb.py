# test_lb.py
# Quick test to verify load balancer + scheduler are wired correctly
import time
import threading
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from common.models import Request

def test_basic_routing():
    print("\n=== Test 1: Basic Routing ===")
    workers   = [GPUWorker(i, max_capacity=10) for i in range(4)]
    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb
    
    try:
        time.sleep(0.2)
        req  = Request(id=1, query="test query")
        resp = lb.dispatch(req, strategy="least_connections")
        assert resp is not None, "Response should not be None"
        print(f"  Response: {resp}")
        print("  PASS")
    finally:
        scheduler.stop() # CRITICAL: Kill background threads

def test_all_strategies():
    print("\n=== Test 2: All 3 Strategies ===")
    workers   = [GPUWorker(i, max_capacity=10) for i in range(4)]
    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb
    
    try:
        time.sleep(0.2)
        for strategy in ["round_robin", "least_connections", "load_aware"]:
            req  = Request(id=hash(strategy) % 1000, query=f"test {strategy}")
            resp = lb.dispatch(req, strategy=strategy)
            assert resp is not None, f"Strategy {strategy} returned None"
            print(f"  {strategy}: PASS")
    finally:
        scheduler.stop()

def test_fault_tolerance():
    print("\n=== Test 3: Fault Tolerance ===")
    workers   = [GPUWorker(i, max_capacity=10) for i in range(4)]
    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb
    
    try:
        time.sleep(0.2)
        # Kill worker 0
        workers[0].simulate_failure()
        time.sleep(0.1)

        # System should still handle requests using other workers
        req  = Request(id=999, query="test after failure")
        resp = lb.dispatch(req, strategy="least_connections")
        assert resp is not None, "Should still work after worker failure"
        print(f"  Worker 0 killed. Request handled successfully. Response: {resp}")
        print("  PASS")
    finally:
        scheduler.stop()

def test_concurrent_requests():
    print("\n=== Test 4: Concurrent Requests ===")
    workers   = [GPUWorker(i, max_capacity=50) for i in range(4)]
    lb        = LoadBalancer(workers)
    scheduler = Scheduler()
    lb.scheduler = scheduler
    scheduler.lb = lb
    
    try:
        time.sleep(0.2)
        results = []
        lock    = threading.Lock()

        def send(req_id):
            req  = Request(id=req_id, query=f"concurrent query {req_id}")
            resp = lb.dispatch(req, strategy="load_aware")
            with lock:
                results.append(resp)

        threads = [threading.Thread(target=send, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        print(f"  20 concurrent requests all completed successfully")
        print("  PASS")
    finally:
        scheduler.stop()

if __name__ == "__main__":
    test_basic_routing()
    test_all_strategies()
    test_fault_tolerance()
    test_concurrent_requests()
    print("\n=== ALL TESTS PASSED ===")