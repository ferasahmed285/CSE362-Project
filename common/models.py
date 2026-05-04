# common/models.py
# Final merged version - combines track1 InternalTaskMessage + member4 fixes
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
    max_retries: int = 3
    status: RequestStatus = RequestStatus.PENDING
    timestamp: float = field(default_factory=time.time)  # fixed bug from original

@dataclass
class Response:
    id: int
    result: str
    latency: float
    worker_id: int = -1
    success: bool = True
    error_message: str = ""

@dataclass
class WorkerStats:
    worker_id: int
    active_connections: int = 0
    max_capacity: int = 10
    total_processed: int = 0
    gpu_utilization: float = 0.0
    current_latency: float = 0.0
    is_alive: bool = True

@dataclass
class InternalTaskMessage:
    """Used by Master Scheduler to track task routing decisions."""
    request: Request
    worker_id: int
    failure_flags: list
    retry_count: int
