# llm/inference.py
# Real LLM (Ollama) with automatic fallback to simulation

import time
import random

_USE_REAL_LLM = False
try:
    import ollama
    ollama.list()
    _USE_REAL_LLM = True
    print("[LLM] Ollama detected — using REAL LLM inference (tinyllama)")
except Exception:
    print("[LLM] Ollama not available — using SIMULATION mode")

MODEL_NAME  = "tinyllama"
MAX_TOKENS  = 150
TEMPERATURE = 0.7

_SIM_RESPONSES = [
    "Load balancing distributes workloads across multiple computing resources to optimize utilization.",
    "Distributed computing coordinates multiple computers to solve complex problems via parallel processing.",
    "GPU clusters enable massive parallel computation, ideal for AI inference workloads.",
    "Fault tolerance ensures availability by detecting failures and redistributing tasks to healthy nodes.",
    "RAG enhances LLM responses by incorporating relevant info from external knowledge bases.",
    "The master-worker architecture separates scheduling from execution for efficient resource management.",
    "Round-robin distributes requests evenly across all available servers in sequential order.",
    "Least-connections routing directs new requests to the server with fewest active connections.",
    "Vector databases store embeddings as high-dimensional vectors for fast similarity search.",
    "Batch processing groups multiple inference requests to maximize GPU throughput and efficiency.",
]


def run_llm(query: str, context: str = "") -> str:
    if _USE_REAL_LLM:
        return _run_real(query, context)
    return _run_sim(query, context)


def _run_real(query, context):
    if context and context != "No context available":
        prompt = (f"You are a helpful assistant. Use this context:\n\n"
                  f"Context: {context}\n\nQuestion: {query}\n\nAnswer concisely in 2-3 sentences:")
    else:
        prompt = f"You are a helpful assistant.\n\nQuestion: {query}\n\nAnswer concisely in 2-3 sentences:"
    try:
        resp = ollama.chat(model=MODEL_NAME,
                           messages=[{"role": "user", "content": prompt}],
                           options={"num_predict": MAX_TOKENS, "temperature": TEMPERATURE})
        return resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"LLM inference failed: {e}. Ensure Ollama is running and tinyllama is pulled.")


def _run_sim(query, context):
    time.sleep(random.uniform(0.3, 1.5))
    q = query.lower()
    for r in _SIM_RESPONSES:
        if len(set(r.lower().split()) & set(q.split())) >= 2:
            return f"[Simulated] {r}"
    return f"[Simulated] {random.choice(_SIM_RESPONSES)}"


def run_llm_batch(requests: list) -> list:
    return [run_llm(q, c) for q, c in requests]


if __name__ == "__main__":
    print(f"=== Testing LLM ({'Real' if _USE_REAL_LLM else 'Sim'}) ===\n")
    start = time.time()
    print(f"  Answer: {run_llm('What is machine learning?')}")
    print(f"  Time: {time.time()-start:.2f}s\n")
    print("=== Test complete ===")