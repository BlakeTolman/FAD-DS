from collections import OrderedDict
from typing import Any, Optional
from metrics.monitor import MetricsMonitor

class ARCCache:
    def __init__(self, max_size: int):
        """Initialize ARC cache with a maximum size."""
        self.max_size = max_size
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.monitor = MetricsMonitor()

    def reset(self) -> None:
        """Reset the cache to its initial state."""
        self.p = 0
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()
        self.monitor.reset()

    def _replace(self, key: str) -> None:
        """Replace an item when cache is full, adapting p."""
        # If t1 is empty or p is 0, we must replace from t2
        if not self.t1 or self.p == 0:
            if self.t2:
                old_key, _ = self.t2.popitem(last=False)
                self.b2[old_key] = None
            else:
                # If both t1 and t2 are empty, we can't replace anything
                return
        # If t2 is empty or p is max_size, we must replace from t1
        elif not self.t2 or self.p == self.max_size:
            if self.t1:
                old_key, _ = self.t1.popitem(last=False)
                self.b1[old_key] = None
            else:
                # If both t1 and t2 are empty, we can't replace anything
                return
        # Otherwise, use the ARC algorithm to decide which to replace
        elif (key in self.b2 and len(self.t1) == self.p) or (len(self.t1) > self.p):
            if self.t1:
                old_key, _ = self.t1.popitem(last=False)
                self.b1[old_key] = None
            else:
                # Fallback to t2 if t1 is empty
                if self.t2:
                    old_key, _ = self.t2.popitem(last=False)
                    self.b2[old_key] = None
        else:
            if self.t2:
                old_key, _ = self.t2.popitem(last=False)
                self.b2[old_key] = None
            else:
                # Fallback to t1 if t2 is empty
                if self.t1:
                    old_key, _ = self.t1.popitem(last=False)
                    self.b1[old_key] = None

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if key in self.t1:
            value = self.t1.pop(key)
            self.t2[key] = value
            self.monitor.record_operation("get", key, True)
            self.monitor.record_memory_usage()
            self.monitor.record_cpu_usage()
            return value
        elif key in self.t2:
            self.t2.move_to_end(key)
            self.monitor.record_operation("get", key, True)
            self.monitor.record_memory_usage()
            self.monitor.record_cpu_usage()
            return self.t2[key]

        if key in self.b1:
            self.p = min(self.max_size, self.p + max(len(self.b2) // len(self.b1), 1))
            self._replace(key)
            self.b1.pop(key)
            self.t2[key] = None
            self.monitor.record_operation("get", key, False)
            self.monitor.record_memory_usage()
            self.monitor.record_cpu_usage()
            return None

        if key in self.b2:
            self.p = max(0, self.p - max(len(self.b1) // len(self.b2), 1))
            self._replace(key)
            self.b2.pop(key)
            self.t2[key] = None
            self.monitor.record_operation("get", key, False)
            self.monitor.record_memory_usage()
            self.monitor.record_cpu_usage()
            return None

        self.monitor.record_operation("get", key, False)
        self.monitor.record_memory_usage()
        self.monitor.record_cpu_usage()
        return None

    def put(self, key: str, value: Any) -> None:
        """Put a value into the cache."""
        if key in self.t1:
            self.t1[key] = value
        elif key in self.t2:
            self.t2[key] = value
        elif key in self.b1:
            self.p = min(self.max_size, self.p + max(len(self.b2) // len(self.b1), 1))
            self._replace(key)
            self.b1.pop(key)
            self.t2[key] = value
        elif key in self.b2:
            self.p = max(0, self.p - max(len(self.b1) // len(self.b2), 1))
            self._replace(key)
            self.b2.pop(key)
            self.t2[key] = value
        else:
            total_size = len(self.t1) + len(self.t2)
            if total_size < self.max_size:
                if len(self.t1) + len(self.b1) >= self.max_size:
                    if len(self.t1) < self.max_size:
                        self.b1.popitem(last=False)
                    else:
                        self.t1.popitem(last=False)
                self.t1[key] = value
            else:
                self._replace(key)
                self.t1[key] = value

        # Only record memory and CPU usage, not operation
        self.monitor.record_memory_usage()
        self.monitor.record_cpu_usage()

    def summary(self) -> dict:
        """Return cache performance summary."""
        return self.monitor.summary()