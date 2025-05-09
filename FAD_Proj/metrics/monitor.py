from typing import Dict, List
from time import time
import psutil
import os

class MetricsMonitor:
    def __init__(self):
        """Initialize the metrics monitor."""
        self.hits: int = 0
        self.misses: int = 0
        self.operations: List[Dict] = []
        self.decay_rates: Dict[str, List[Dict]] = {
            "cold": [],
            "warm": [],
            "hot": []
        }
        self.memory_usage: List[Dict] = []
        self.cpu_usage: List[Dict] = []  # CPU usage tracking
        self.process = psutil.Process(os.getpid())  # Process for CPU/memory
        self.promotions: List[Dict] = []  # Track promotions between segments
        self.demotions: List[Dict] = []  # Track demotions between segments
        self.evictions: List[Dict] = []  # Track evictions from segments
        self.promotion_count = 0
        self.demotion_count = 0
        self.eviction_count = 0
        self.operation_count = 0
        self.cpu_window_size = 50  # Measure CPU over 50 operations
        self.cpu_window_start = time()
        self.cpu_window_ops = 0

    def record_operation(self, op_type: str, key: str, hit: bool) -> None:
        """Record a cache operation (get/put)."""
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        self.operations.append({
            "time": time(),
            "type": op_type,
            "key": key,
            "hit": hit,
            "total_ops": self.hits + self.misses
        })


    def record_memory_usage(self) -> None:
        """Record current process memory usage."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024  # Convert to MB
        self.memory_usage.append({
            "time": time(),
            "memory_mb": memory_mb
        })

    def record_cpu_usage(self) -> None:
        """Record current process CPU usage over a window of operations."""
        self.operation_count += 1
        self.cpu_window_ops += 1
        
        # Only measure CPU after a window of operations
        if self.cpu_window_ops >= self.cpu_window_size:
            window_end = time()
            window_duration = window_end - self.cpu_window_start
            
            # Get CPU usage for the entire window
            cpu_percent = self.process.cpu_percent(interval=None)
            # Calculate CPU usage per operation in the window
            cpu_per_op = cpu_percent / self.cpu_window_ops if self.cpu_window_ops > 0 else 0
            
            # Record the average CPU usage for this window
            self.cpu_usage.append({
                "time": window_end,
                "cpu_percent": cpu_per_op
            })
            
            # Reset window counters
            self.cpu_window_start = window_end
            self.cpu_window_ops = 0
        else:
            # Use last known CPU value for intermediate operations
            if self.cpu_usage:
                self.cpu_usage.append({
                    "time": time(),
                    "cpu_percent": self.cpu_usage[-1]["cpu_percent"]
                })
            else:
                self.cpu_usage.append({
                    "time": time(),
                    "cpu_percent": 0.0
                })

    
    def record_decay_change(self, segment: str, decay_rate: float) -> None:
        """Record a change in decay rate for a segment."""
        self.decay_rates[segment].append({
            "time": time(),
            "decay_rate": decay_rate
        })
        
    def record_promotion(self, from_segment: str, to_segment: str) -> None:
        """Record a promotion between segments."""
        self.promotions.append({
            "time": time(),
            "from": from_segment,
            "to": to_segment
        })
        self.promotion_count += 1

    def record_demotion(self, from_segment: str, to_segment: str) -> None:
        """Record a demotion between segments."""
        self.demotions.append({
            "time": time(),
            "from": from_segment,
            "to": to_segment
        })
        self.demotion_count += 1

    def record_eviction(self, segment: str, key: str) -> None:
        """Record an eviction from a segment."""
        self.evictions.append({
            "time": time(),
            "segment": segment,
            "key": key
        })
        self.eviction_count += 1

    def get_hit_ratio(self) -> float:
        """Calculate the current hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_promotion_count(self) -> int:
        return self.promotion_count

    def get_demotion_count(self) -> int:
        return self.demotion_count

    def get_eviction_count(self) -> int:
        return self.eviction_count

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.operations = []
        for segment in self.decay_rates:
            self.decay_rates[segment] = []
        self.memory_usage = []
        self.cpu_usage = []
        self.promotions = []
        self.demotions = []
        self.evictions = []
        self.promotion_count = 0
        self.demotion_count = 0
        self.eviction_count = 0

    def summary(self) -> Dict:
        """Return a summary of collected metrics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.get_hit_ratio(),
            "total_operations": len(self.operations),
            "decay_changes": {seg: len(changes) for seg, changes in self.decay_rates.items()},
            "memory_samples": len(self.memory_usage),
            "cpu_samples": len(self.cpu_usage),
            "avg_memory_mb": sum(m["memory_mb"] for m in self.memory_usage) / (len(self.memory_usage) or 1),
            "avg_cpu_percent": sum(c["cpu_percent"] for c in self.cpu_usage) / (len(self.cpu_usage) or 1),
            "promotions": len(self.promotions),
            "demotions": len(self.demotions),
            "evictions": len(self.evictions)
        }