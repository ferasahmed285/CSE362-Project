# CSE354: Distributed Computing Project
## Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests

**Ain Shams University — Faculty of Engineering — 2nd Semester 2025/2026**

---

## Project Overview

This project implements a distributed system that handles concurrent user requests using a real Large Language Model (LLM) inference engine and Retrieval-Augmented Generation (RAG). The system demonstrates efficient load balancing, GPU cluster task distribution, fault tolerance, and scalability using a real AI model running locally via Ollama.

---

## Requirements

- Python 3.9+
- Ollama (local LLM runtime)
- ollama Python library

---

## Setup Instructions

### Step 1 — Install Ollama
Download and install from: **https://ollama.com**

### Step 2 — Pull the AI model
```bash
ollama pull tinyllama
```
This downloads the tinyllama model (~637MB). It is a real neural network that generates actual AI responses.

### Step 3 — Start Ollama server
```bash
ollama serve
```
Keep this running in a separate terminal while using the project.

### Step 4 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Run the project
```bash
python main.py
```

### Step 6 — Run unit tests
```bash
python test_lb.py
```

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
│   └── gpu_worker.py       # GPU worker with capacity limits and fault tolerance
├── llm/
│   └── inference.py        # Real LLM inference using Ollama (tinyllama)
├── rag/
│   └── retriever.py        # RAG module with keyword-based retrieval
├── client/
│   └── load_generator.py   # Load generator simulating concurrent users
├── data/
│   └── knowledge_base.txt  # Knowledge base for RAG retrieval
├── main.py                 # Main entry point - runs all 6 tests
├── test_lb.py              # Unit tests for load balancer + scheduler
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## How It Works

### Real LLM Integration
Unlike simulated systems, this project uses **tinyllama** — a real neural network running locally via Ollama. Each request is processed by the actual model, producing genuine AI-generated responses. This means:
- Each request takes 1-10 seconds of real computation
- Responses are different each time (non-deterministic)
- The system must handle real-world latency variance

### System Architecture

```
Client Layer (concurrent users)
        ↓
Load Balancer (routing strategy selection)
        ↓
Master Node / Scheduler (task queuing + fault detection)
        ↓
GPU Workers 0-7 (real LLM inference via Ollama + RAG)
        ↑
RAG Module (knowledge base retrieval)
```

---

## Load Balancing Strategies

### 1. Round Robin
Distributes requests sequentially across all alive workers in circular order.

### 2. Least Connections
Routes each new request to the worker with the fewest active connections.

### 3. Load Aware
Calculates a composite load score using active connections and GPU utilization, routing to the lowest-scored worker.

---

## Fault Tolerance

- **Health monitoring:** Master node pings all workers every 15 seconds
- **Failure detection:** Worker marked offline if ping fails
- **Task reassignment:** Failed tasks automatically retried on other workers (up to 5 attempts)
- **Recovery:** Workers automatically come back online after recovery
- **Demo:** Test 3 kills Worker 1 mid-run — system recovers with 0 failed requests

---

## Testing & Results

Tests use real LLM inference (tinyllama via Ollama). User counts in the default demo are lower than simulation because real AI inference takes 1-10 seconds per request — this is the realistic behavior of actual GPU-based LLM serving systems.

### Default Demo (main.py)

| Test | Users | Success | Failed | Throughput | Avg Latency |
|------|-------|---------|--------|------------|-------------|
| Test 1 | 10 | 10 | 0 | 1.17 req/s | 5.17s |
| Test 2 | 20 | 20 | 0 | 1.26 req/s | 8.43s |
| Test 3 + failure | 24 | 24 | 0 | 0.77 req/s | 10.74s |
| Round Robin | 24 | 24 | 0 | 1.22 req/s | 10.56s |
| Least Connections | 24 | 24 | 0 | 1.07 req/s | 11.81s |
| Load Aware | 24 | 24 | 0 | 1.09 req/s | 11.16s |

### 1000-User Stress Test (stress_test.py)

The system architecture fully supports 1000+ concurrent requests. The worker pool queues and processes all requests using real LLM inference. Run with:

```bash
python stress_test.py
```

**Note:** With real LLM inference taking 1-10 seconds per request and 8 workers × 10 capacity = 80 concurrent slots, 1000 requests will take approximately 30-60 minutes to complete. This is expected behavior for real neural network inference on consumer hardware without a dedicated GPU.

---

## Limitations

- Ollama processes one request at a time per worker (no true GPU batching without hardware)
- Knowledge base is small (20 sentences) — production RAG uses vector databases
- No persistent storage — all state is in-memory
- tinyllama is a small model — larger models like Llama2-7B would give better responses but require more RAM

---

## Team Members

- Member 1: Load Balancer (`lb/`)
- Member 2: Master Node / Scheduler (`master/`)
- Member 3: GPU Workers + LLM Inference (`workers/`, `llm/`)
- Member 4: RAG + Client Load Generator (`rag/`, `client/`)
