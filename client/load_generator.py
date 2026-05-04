# client/load_generator.py

import concurrent.futures
import time
import statistics
from common.models import Request


def simulate_user(lb, user_id, strategy="least_connections"):
    request = Request(
        id=user_id,
        query=f"Query {user_id}: explain load balancing and RAG"
    )

    start = time.time()

    try:
        response = lb.dispatch(request, strategy=strategy)
        end = time.time()

        return {
            "id": user_id,
            "latency": end - start,
            "success": True
        }

    except Exception as exc:
        end = time.time()

        return {
            "id": user_id,
            "latency": end - start,
            "success": False,
            "error": str(exc)
        }


def run_load_test(lb, num_users=1000, max_workers=200, strategy="least_connections"):
    print(f"\n[Client] Starting load test with {num_users} users...")

    start_time = time.time()
    responses = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_user, lb, i, strategy)
            for i in range(num_users)
        ]

        for future in concurrent.futures.as_completed(futures):
            responses.append(future.result())

    total_time = time.time() - start_time

    success = [r for r in responses if r["success"]]
    failed = [r for r in responses if not r["success"]]

    latencies = [r["latency"] for r in success]

    print("\n========== RESULTS ==========")
    print(f"Total Requests: {num_users}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")
    print(f"Total Time: {total_time:.2f}s")

    if total_time > 0:
        print(f"Throughput: {len(success)/total_time:.2f} req/sec")

    if latencies:
        print(f"Avg Latency: {statistics.mean(latencies):.3f}s")
        print(f"Min Latency: {min(latencies):.3f}s")
        print(f"Max Latency: {max(latencies):.3f}s")

        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(0.95 * len(sorted_lat)) - 1]
        print(f"P95 Latency: {p95:.3f}s")

    print("============================\n")

    return responses# client/load_generator.py

import concurrent.futures
import time
import statistics
from common.models import Request


def simulate_user(lb, user_id, strategy="least_connections"):
    request = Request(
        id=user_id,
        query=f"Query {user_id}: explain load balancing and RAG"
    )

    start = time.time()

    try:
        response = lb.dispatch(request, strategy=strategy)
        end = time.time()

        return {
            "id": user_id,
            "latency": end - start,
            "success": True
        }

    except Exception as exc:
        end = time.time()

        return {
            "id": user_id,
            "latency": end - start,
            "success": False,
            "error": str(exc)
        }


def run_load_test(lb, num_users=1000, max_workers=200, strategy="least_connections"):
    print(f"\n[Client] Starting load test with {num_users} users...")

    start_time = time.time()
    responses = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_user, lb, i, strategy)
            for i in range(num_users)
        ]

        for future in concurrent.futures.as_completed(futures):
            responses.append(future.result())

    total_time = time.time() - start_time

    success = [r for r in responses if r["success"]]
    failed = [r for r in responses if not r["success"]]

    latencies = [r["latency"] for r in success]

    print("\n========== RESULTS ==========")
    print(f"Total Requests: {num_users}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")
    print(f"Total Time: {total_time:.2f}s")

    if total_time > 0:
        print(f"Throughput: {len(success)/total_time:.2f} req/sec")

    if latencies:
        print(f"Avg Latency: {statistics.mean(latencies):.3f}s")
        print(f"Min Latency: {min(latencies):.3f}s")
        print(f"Max Latency: {max(latencies):.3f}s")

        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(0.95 * len(sorted_lat)) - 1]
        print(f"P95 Latency: {p95:.3f}s")

    print("============================\n")

    return responses