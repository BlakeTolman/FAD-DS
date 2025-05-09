# Cache entry class for storing key-value pairs and metadata
from dataclasses import dataclass
from time import time

@dataclass
class CacheEntry:
    key: str
    value: any
    access_count: int = 0
    last_accessed: float = None
    decay_score: float = 0.0
    creation_time: float = None
    promotion_count: int = 0
    last_promotion_time: float = None
    access_history: list = None  # Track recent access times

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = time()
        if self.creation_time is None:
            self.creation_time = time()
        if self.last_promotion_time is None:
            self.last_promotion_time = time()
        if self.access_history is None:
            self.access_history = []

    def access(self):
        self.access_count += 1
        self.last_accessed = time()
        self.access_history.append(time())
        # Keep only last 5 accesses
        if len(self.access_history) > 5:
            self.access_history.pop(0)

    def promote(self):
        self.promotion_count += 1
        self.last_promotion_time = time()
        self.last_accessed = time()

    def get_access_frequency(self) -> float:
        """Calculate access frequency based on recent history."""
        if len(self.access_history) < 2:
            return 0.0
        # Calculate average time between accesses
        intervals = [self.access_history[i] - self.access_history[i-1] 
                    for i in range(1, len(self.access_history))]
        if not intervals:
            return 0.0
        avg_interval = sum(intervals) / len(intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0