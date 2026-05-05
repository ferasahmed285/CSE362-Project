# llm/inference.py
# ============================================================
# REAL LLM INFERENCE USING OLLAMA
#
# This replaces the simulation with a real AI model (tinyllama)
# running locally via Ollama.
#
# Requirements:
#   1. Install Ollama: https://ollama.com
#   2. Pull the model: ollama pull tinyllama
#   3. Install Python library: pip install ollama
#
# The rest of the distributed system (load balancer, scheduler,
# GPU workers) stays exactly the same. Only this file changes.
# ============================================================

import time
import ollama


# ── Model Configuration ────────────────────────────────────
MODEL_NAME   = "tinyllama"   # change to "llama2" or "mistral" if you have them
MAX_TOKENS   = 150           # limit response length for speed
TEMPERATURE  = 0.7           # 0=deterministic, 1=creative


def run_llm(query: str, context: str = "") -> str:
    """
    Runs a real LLM inference using Ollama (tinyllama model).
    
    This is called by GPUWorker for each request.
    Returns a real AI-generated answer.
    
    Args:
        query:   the user's question
        context: extra info retrieved from RAG module
    
    Returns:
        str: real AI-generated response
    """
    # Build the prompt — include RAG context if available
    if context and context != "No context available":
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely in 2-3 sentences:"
        )
    else:
        prompt = (
            f"You are a helpful assistant.\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely in 2-3 sentences:"
        )

    try:
        response = ollama.chat(
            model   = MODEL_NAME,
            messages = [{"role": "user", "content": prompt}],
            options  = {
                "num_predict": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }
        )
        return response["message"]["content"].strip()

    except Exception as e:
        raise RuntimeError(
            f"LLM inference failed: {str(e)}. "
            f"Make sure Ollama is running: 'ollama serve' and model is pulled: 'ollama pull tinyllama'"
        )


def run_llm_batch(requests: list) -> list:
    """
    Processes multiple queries through the real LLM.
    
    Note: Ollama does not natively support true batching like a GPU would.
    We process sequentially here, but the GPUWorker's threading handles
    the parallelism — multiple workers call run_llm() simultaneously.
    
    For the report: real GPU batching would use tensor parallelism
    at the hardware level. Ollama simulates this by handling one
    request at a time per model instance.
    
    Args:
        requests: list of (query, context) tuples
    
    Returns:
        list of answer strings
    """
    results = []
    for query, context in requests:
        result = run_llm(query, context)
        results.append(result)
    return results


# ── Test: run this file directly to verify Ollama is working ──
if __name__ == "__main__":
    print("=== Testing Real LLM with Ollama ===\n")

    # Test 1: simple question
    print("Test 1: Simple question")
    start  = time.time()
    result = run_llm("What is machine learning?")
    elapsed = time.time() - start
    print(f"  Time:   {elapsed:.2f}s")
    print(f"  Answer: {result}\n")

    # Test 2: with RAG context
    print("Test 2: With RAG context")
    context = "Load balancing distributes requests across multiple servers to prevent overload."
    start   = time.time()
    result  = run_llm("What is load balancing?", context)
    elapsed = time.time() - start
    print(f"  Time:   {elapsed:.2f}s")
    print(f"  Answer: {result}\n")

    # Test 3: batch
    print("Test 3: Batch of 3 queries")
    queries = [
        ("What is a GPU?", ""),
        ("What is distributed computing?", ""),
        ("What is fault tolerance?", ""),
    ]
    start   = time.time()
    results = run_llm_batch(queries)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s for {len(queries)} queries")
    for i, r in enumerate(results):
        print(f"  Q{i+1}: {r[:80]}...")

    print("\n=== Test complete ===")