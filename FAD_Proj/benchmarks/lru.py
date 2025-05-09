from cachetools import LRUCache as CacheToolsLRUCache  
from typing import Any
from metrics.monitor import MetricsMonitor

class LRUCache:
    def __init__(self, max_size: int):
        """Initialize LRU cache with a maximum size."""
        self.cache = CacheToolsLRUCache(maxsize=max_size)  
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
        self.cache[key] = value
        # Only record memory and CPU usage, not operation
        self.monitor.record_memory_usage()
        self.monitor.record_cpu_usage()

    def summary(self) -> dict:
        """Return cache performance summary."""
        return self.monitor.summary()