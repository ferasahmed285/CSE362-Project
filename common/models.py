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
    max_retries: int = 5
    status: RequestStatus = RequestStatus.PENDING
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self):
        return {
            "id": self.id,
            "query": self.query,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            query=data["query"],
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 5),
            status=RequestStatus(data.get("status", "PENDING")),
            timestamp=data.get("timestamp", time.time())
        )

@dataclass
class Response:
    id: int
    result: str
    latency: float
    worker_id: int = -1
    success: bool = True
    error_message: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "result": self.result,
            "latency": self.latency,
            "success": self.success,
            "error_message": self.error_message
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            result=data["result"],
            latency=data["latency"],
            success=data.get("success", True),
            error_message=data.get("error_message", "")
        )

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
    def to_dict(self):
        return {
            "request": self.request.to_dict(),
            "worker_id": self.worker_id,
            "failure_flags": self.failure_flags,
            "retry_count": self.retry_count
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            request=Request.from_dict(data["request"]),
            worker_id=data["worker_id"],
            failure_flags=data.get("failure_flags", []),
            retry_count=data.get("retry_count", 0)
        )
