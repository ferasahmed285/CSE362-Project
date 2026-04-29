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
    retries: int = 0  
    status: RequestStatus = RequestStatus.PENDING
    timestamp: float = time.time() 

@dataclass
class Response:
    id: int
    result: str
    latency: float
    success: bool = True  
    error_message: str = ""

@dataclass
class WorkerStats:
    worker_id: int
    active_connections: int = 0
    gpu_utilization: float = 0.0
    current_latency: float = 0.0
    is_alive: bool = True