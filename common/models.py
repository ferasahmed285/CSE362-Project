# common/models.py
from dataclasses import dataclass, field
from enum import Enum
import time

class RequestStatus(Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"

@dataclass
class Request:
    id: int
    query: str
    retries: int = 0
    status: RequestStatus = RequestStatus.PENDING
    timestamp: float = field(default_factory=time.time)  # FIXED: was time.time() which gave same timestamp to every request

@dataclass
class Response:
    id: int
    result: str
    latency: float
    worker_id: int = -1       # ADDED: which GPU handled this
    success: bool = True
    error_message: str = ""

@dataclass
class WorkerStats:
    worker_id: int
    active_connections: int = 0
    max_capacity: int = 10    # ADDED: needed for utilization calculation
    total_processed: int = 0  # ADDED: needed for throughput reporting
    gpu_utilization: float = 0.0
    current_latency: float = 0.0
    is_alive: bool = True
