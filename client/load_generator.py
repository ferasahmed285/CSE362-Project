import concurrent.futures
from common.models import Request

def simulate_user(lb, user_id):
    request = Request(id=user_id, query=f"Query {user_id}")
    # Target the Load Balancer via HTTP/gRPC abstraction exactly as a Client would!
    response = lb.dispatch(request)
    print(f"[Client] Response {response['id']} | Latency: {response['latency']:.3f}s")
    return response

def run_load_test(lb, num_users=1000):
    print(f"[Client] Starting Load Test: Sending {num_users} simultaneous requests...")
    # Use ThreadPoolExecutor to ensure dispatch() is hit concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        futures = [executor.submit(simulate_user, lb, i) for i in range(num_users)]
        
        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"[Client] Request generated an exception: {exc}")
    print("[Client] Load Test Complete.")