from typing import Any, Dict
from metrics.monitor import MetricsMonitor

class FIFOCache:
    def __init__(self, max_size: int):
        """Initialize FIFO cache with a maximum size."""
        self.cache: Dict[str, Any] = {}  # Store key-value pairs
        self.order: list[str] = []  # Track insertion order
        self.max_size = max_size
        self.monitor = MetricsMonitor()

    def get(self, key: str) -> Any:
        """Get a value from the cache."""
        try:
            value = self.cache[key]
            self.monitor.record_operation("get", key, True)
            self.monitor.record_memory_usage()
            self.monitor.record_cpu_usage()
            return value
        except KeyError:
            self.monitor.record_operation("get", key, False)
            self.monitor.record_memory_usage()
            self.monitor.record_cpu_usage()
            return None

    def put(self, key: str, value: Any) -> None:
        """Put a value into the cache."""
        # If key doesn't exist and cache is full, remove oldest item
        if key not in self.cache and len(self.cache) >= self.max_size:
            oldest_key = self.order.pop(0)  # Remove first item (oldest)
            del self.cache[oldest_key]
        
        # Update or add the value
        self.cache[key] = value
        
        # Only add to order list if it's a new key
        if key not in self.order:
            self.order.append(key)
        
        # Only record memory and CPU usage, not operation
        self.monitor.record_memory_usage()
        self.monitor.record_cpu_usage()

    def summary(self) -> dict:
        """Return cache performance summary."""
        return self.monitor.summary() 