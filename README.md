# CSE354: Distributed Computing Project
## Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests

**Ain Shams University — Faculty of Engineering — 2nd Semester 2025/2026**

---

## Project Overview

This project implements a distributed load balancing system for handling concurrent user requests involving Large Language Model (LLM) inference and Retrieval-Augmented Generation (RAG). The system demonstrates efficient task distribution, fault tolerance, and scalability by utilizing multiple physical laptop GPU workers exposed through **Cloudflare Tunnels**.

### Key Achievements
- **Real Distributed Environment:** GPU workers run on separate physical machines (laptops) connected via Cloudflare tunnels, making it a true distributed computing setup.
- **Real LLM Integration:** Uses the **tinyllama** model via Ollama to generate genuine AI responses.
- **High Concurrency:** Built to handle stress tests of 1000+ concurrent requests.

---

## System Architecture

```text
Client Layer (Concurrent Users)
        ↓
Load Balancer (Routing strategy selection)
        ↓
Master Node / Scheduler (Task queuing + fault detection)
        ↓ (via Cloudflare Tunnels)
Distributed GPU Workers (Multiple Laptops running Ollama + RAG)
        ↑
RAG Module (Knowledge base retrieval)
```

### Load Balancing Strategies Implemented
1. **Round Robin:** Distributes requests sequentially across all alive workers.
2. **Least Connections:** Routes each new request to the worker with the fewest active connections.
3. **Load Aware:** Calculates a composite load score based on active connections and GPU utilization, routing to the optimal worker.

### Fault Tolerance Mechanisms
- **Health monitoring:** The Master node periodically pings all remote laptop workers.
- **Failure detection:** A worker is automatically marked offline if a Cloudflare tunnel drops or the node becomes unresponsive.
- **Task reassignment:** Failed tasks are automatically retried on other available remote workers, ensuring 0 dropped requests during a worker crash.

---

## Requirements

- Python 3.9+
- Ollama installed on each worker laptop
- `tinyllama` model pulled in Ollama
- Cloudflared tunnel installed for each worker
- Python libraries: `ollama`, `requests` (see `requirements.txt`)

---

## Setup & Execution Instructions

### 1. Worker Setup (Run on each laptop GPU worker)
1. Start Ollama on the worker laptop.
2. Pull the model (if not already downloaded):
   ```bash
   ollama pull tinyllama
   ```
3. Expose the worker using a Cloudflare tunnel:
   ```bash
   cloudflared tunnel --url http://localhost:11434 --http-host-header="localhost:11434"
   ```

### 2. Master Setup (Run on the central controller)
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Update the configuration:
   Copy the generated Cloudflare URLs from your workers and paste them into the worker configurations in the code (e.g., in `main.py` or the stress tests).
3. Run the main system:
   ```bash
   python main.py
   ```

---

## Testing & Evaluation

Run the following scripts to evaluate system performance, load distribution, and fault tolerance:

- `python main.py` - Runs the default demo and testing suite.
- `python test_lb.py` - Unit tests for the load balancer and scheduler.
- `python stress_test.py` - Stress testing for up to 1000 concurrent users.
- `python stress_test_2.py` - Advanced stress testing and failure simulation scenarios.

---

## Limitations & Future Work
- Ollama processes one request at a time per worker instance (no true hardware batching).
- The knowledge base for RAG is currently a text stub; a full production system would upgrade to a vector database.
- `tinyllama` is a small model for performance purposes; larger models would yield better AI responses but require more RAM.