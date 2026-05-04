# CSE354: Distributed Computing Project
## Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests

**Ain Shams University — Faculty of Engineering — 2nd Semester 2025/2026**

---

## Project Overview

This project implements a distributed system capable of handling 1000+ concurrent user requests involving Large Language Model (LLM) inference and Retrieval-Augmented Generation (RAG). The system focuses on efficient load balancing, GPU cluster task distribution, fault tolerance, and scalability.

---

## Folder Structure

```
CSE362-Project/
├── common/
│   └── models.py           # Shared data models (Request, Response, WorkerStats)
├── lb/
│   └── load_balancer.py    # Load balancer with 3 routing strategies
├── master/
│   └── scheduler.py        # Master node: task scheduling + fault detection
├── workers/
│   ├── gpu_worker.py       # GPU worker with batching, metrics, fault tolerance
│   └── metrics.py          # Performance metrics tracker
├── llm/
│   └── inference.py        # LLM inference simulation (variable latency)
├── rag/
│   └── retriever.py        # RAG module with keyword-based retrieval
├── client/
│   └── load_generator.py   # Load generator simulating 1000 concurrent users
├── data/
│   └── knowledge_base.txt  # Knowledge base for RAG retrieval
├── main.py                 # Main entry point - runs all 6 tests
├── test_lb.py              # Unit tests for load balancer + scheduler
└── README.md               # This file
```

---

## How to Run

### Requirements
- Python 3.9+

### Install dependencies
No external packages needed — uses Python standard library only.

### Run the full test suite
```bash
python main.py
```

This automatically runs 6 tests:
1. 100 concurrent users
2. 500 concurrent users
3. 1000 concurrent users + worker failure mid-run
4. 1000 users with Round Robin strategy
5. 1000 users with Least Connections strategy
6. 1000 users with Load Aware strategy

### Run unit tests
```bash
python test_lb.py
```

---

## Load Balancing Strategies

### 1. Round Robin
Distributes requests sequentially across all alive workers in circular order. Simple and fair but does not account for current load.

### 2. Least Connections
Routes each new request to the worker with the fewest active connections. More intelligent than Round Robin — avoids overloading busy workers.

### 3. Load Aware (Best)
Calculates a composite load score combining active connections and GPU utilization:
```
score = (active_connections × 1.0) + (gpu_utilization × 0.05)
```
Routes to the worker with the lowest score. Handles both queue depth and hardware load for optimal routing.

---

## System Architecture

```
Client Layer (1000 concurrent users)
        ↓
Load Balancer (routing strategy selection)
        ↓
Master Node / Scheduler (task queuing + fault detection)
        ↓
GPU Workers 0-7 (LLM inference + RAG + batching)
        ↑
RAG Module (knowledge base retrieval)
```

---

## Fault Tolerance

- **Health monitoring:** Master node pings all workers every 2.5 seconds
- **Failure detection:** Worker marked offline if ping fails
- **Task reassignment:** Failed tasks automatically retried on other workers (up to 5 attempts)
- **Recovery:** Workers can be brought back online via `worker.recover()`
- **Demo:** Test 3 kills Worker 1 mid-run and still achieves 1000/1000 success

---

## GPU Worker Features

- **Batching:** Groups up to 8 requests together for parallel processing (35% faster per-request)
- **Capacity limits:** Each worker has configurable max concurrent requests
- **Thread safety:** Lock-protected counters prevent race conditions
- **Metrics tracking:** Tracks total processed, failures, avg latency, p95 latency

---

## LLM Simulation

Real LLM replaced with realistic simulation:
- Variable latency based on query token count (`time_per_token = 0.008s`)
- Gaussian random noise (`std = 0.03s`) simulates GPU variance
- Batch processing speedup factor (`0.65x`) simulates GPU parallelism
- RAG context integrated into response

---

## RAG (Retrieval-Augmented Generation)

- Reads from `data/knowledge_base.txt`
- Splits text into sentences (chunks)
- Scores each chunk by keyword overlap with query
- Returns top 3 most relevant chunks as context
- Results cached to avoid re-processing identical queries

---

## Testing & Results

| Test | Users | Success | Failed | Throughput | Avg Latency | P95 Latency |
|------|-------|---------|--------|------------|-------------|-------------|
| Test 1 | 100 | 100 | 0 | 151 req/s | 0.377s | 0.452s |
| Test 2 | 500 | 500 | 0 | 293 req/s | 0.554s | 0.755s |
| Test 3 + failure | 1000 | 1000 | 0 | 294 req/s | 1.294s | 2.355s |
| Round Robin | 1000 | 1000 | 0 | 399 req/s | 1.198s | 1.981s |
| Least Connections | 1000 | 1000 | 0 | 394 req/s | 1.182s | 1.962s |
| Load Aware | 1000 | 1000 | 0 | 402 req/s | 1.165s | 1.944s |

---

## Limitations

- LLM inference is simulated, not a real model
- GPU workers are threads, not physical GPU servers
- Knowledge base is small (20 sentences) — real RAG would use a vector database
- No persistent storage — all state is in-memory
- No authentication or security layer

---

## Team Members

- Member 1: Load Balancer (`lb/`)
- Member 2: Master Node / Scheduler (`master/`)
- Member 3: GPU Workers + LLM Inference (`workers/`, `llm/`)
- Member 4: RAG + Client Load Generator (`rag/`, `client/`)
