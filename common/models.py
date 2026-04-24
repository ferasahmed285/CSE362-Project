# common/models.py
from dataclasses import dataclass
from enum import Enum
import time

class RequestStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class Request:
    id: int
    query: str
    retries: int = 0  # Tracks how many times this has been reassigned
    status: RequestStatus = RequestStatus.PENDING
    timestamp: float = time.time() # Useful for load-aware routing

@dataclass
class Response:
    id: int
    result: str
    latency: float
    success: bool = True  # Allows the worker to signal a failure gracefully
    error_message: str = ""

@dataclass
class WorkerStats:
    worker_id: int
    active_connections: int = 0
    is_alive: bool = True