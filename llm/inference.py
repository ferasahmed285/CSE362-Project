# llm/inference.py
import time
import random

BASE_LATENCY   = 0.05
TIME_PER_TOKEN = 0.008
VARIANCE_STD   = 0.03
BATCH_SPEEDUP  = 0.65

def estimate_tokens(text: str) -> int:
    return len(text.split())

def calculate_latency(query: str, context: str = "") -> float:
    query_tokens   = estimate_tokens(query)
    context_tokens = estimate_tokens(context) if context else 0
    token_time = (query_tokens * TIME_PER_TOKEN) + (context_tokens * TIME_PER_TOKEN * 0.3)
    noise      = random.gauss(0, VARIANCE_STD)
    return max(BASE_LATENCY, BASE_LATENCY + token_time + noise)

def run_llm(query: str, context: str = "") -> str:
    """
    Simulates LLM inference with realistic variable latency.
    Longer queries take more time (token-based scaling).
    """
    latency = calculate_latency(query, context)
    time.sleep(latency)
    tokens = estimate_tokens(query)
    if context:
        return (
            f"Based on retrieved context, answer to '{query}': "
            f"[Simulated response | {tokens} input tokens | "
            f"{estimate_tokens(context)} context tokens]"
        )
    return f"Answer to '{query}': [Simulated LLM response | {tokens} input tokens]"

def run_llm_batch(requests: list) -> list:
    """
    Processes multiple queries together (batching).
    Faster per-request than individual processing.
    Real GPUs amortize startup cost across the whole batch.
    """
    if not requests:
        return []
    batch_size  = len(requests)
    max_latency = max(calculate_latency(q, c) for q, c in requests)
    size_bonus  = min(0.20, (batch_size - 1) * 0.03)
    batch_lat   = max(BASE_LATENCY, max_latency * (BATCH_SPEEDUP - size_bonus))
    time.sleep(batch_lat)
    results = []
    for query, context in requests:
        tokens = estimate_tokens(query)
        results.append(
            f"[BATCH-{batch_size}] Answer to '{query}': "
            f"[Simulated response | {tokens} tokens | batch_lat={batch_lat:.3f}s]"
        )
    return results
