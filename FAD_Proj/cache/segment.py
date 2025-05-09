from time import time
from collections import OrderedDict, deque
from .entry import CacheEntry
import numpy as np

# Segment logic for cache segments (cold, warm, hot)

class CacheSegment:
    def __init__(self, size: int, decay_rate: float, freq_weight: float = 1.0, recency_weight: float = 1.0, target_ratio: float = 0.1):
        self.max_size = size
        self.decay_rate = decay_rate
        self.freq_weight = freq_weight
        self.recency_weight = recency_weight
        self.target_ratio = target_ratio
        self.entries = OrderedDict()
        self.total_accesses = 0
        self.hits = 0
        self.misses = 0
        self.last_decay_time = time()
        self.hit_ratio_window = deque(maxlen=100)  # Track last 100 operations

    def add(self, key: str, value: any) -> None:
        # Always evict if we're at max size, even for existing keys
        if len(self.entries) >= self.max_size:
            self.evict()
        entry = CacheEntry(key, value)
        self.entries[key] = entry

    def get(self, key: str) -> CacheEntry:
        self.total_accesses += 1
        if key in self.entries:
            entry = self.entries[key]
            entry.access_count += 1
            self.hits += 1
            self.hit_ratio_window.append(1)  # 1 for hit
            entry.last_accessed = time()
            self.entries.move_to_end(key)
            return entry
        self.misses += 1
        self.hit_ratio_window.append(0)  # 0 for miss
        return None

    def check_exists(self, key: str) -> CacheEntry:
        """Check if key exists without recording metrics or updating access patterns."""
        return self.entries.get(key)

    def evict(self) -> None:
        if self.entries:
            # Calculate current ratio of entries
            current_ratio = len(self.entries) / self.max_size
            
            # If we're above target ratio, be more aggressive in eviction
            if current_ratio > self.target_ratio:
                # Use more aggressive eviction criteria
                evict_key = min(
                    self.entries.items(),
                    key=lambda kv: (kv[1].decay_score * 0.7, kv[1].access_count * 0.3)
                )[0]
            else:
                # Use normal eviction criteria
                evict_key = min(
                    self.entries.items(),
                    key=lambda kv: (kv[1].decay_score, kv[1].access_count)
                )[0]
                self.entries.pop(evict_key)

    def update_decay(self, decay_rate: float) -> None:
        self.decay_rate = decay_rate

    def apply_decay(self) -> None:
        current_time = time()
        time_since_last_decay = current_time - self.last_decay_time
        self.last_decay_time = current_time

        # More aggressive decay
        for entry in self.entries.values():
            time_diff = current_time - entry.last_accessed
            # Exponential decay with higher rate
            decay_factor = max(0.2, np.exp(-self.decay_rate * time_diff * 2.0))
            entry.access_count *= decay_factor
            
            # Calculate decay score with more weight on recency
            age = time_diff
            entry.decay_score = (entry.access_count ** (self.freq_weight * 0.7)) / ((age + 0.1) ** (self.recency_weight * 1.5))
            entry.decay_score = max(0, min(entry.decay_score, 1))

    def get_statistics(self) -> dict:
        """Get statistics about the segment."""
        if not self.entries:
            return {
                "size": 0,
                "avg_access_count": 0,
                "avg_age": 0,
                "avg_decay_score": 0,
                "total_accesses": 0,
                "hit_ratio": 0.0
            }

        current_time = time()
        total_access_count = sum(entry.access_count for entry in self.entries.values())
        total_age = sum(current_time - entry.last_accessed for entry in self.entries.values())
        total_decay_score = sum(entry.decay_score for entry in self.entries.values())
        
        return {
            "size": len(self.entries),
            "avg_access_count": total_access_count / len(self.entries),
            "avg_age": total_age / len(self.entries),
            "avg_decay_score": total_decay_score / len(self.entries),
            "total_accesses": self.total_accesses,
            "hit_ratio": self.get_hit_ratio()
        }

    def get_hit_ratio(self) -> float:
        """Calculate the hit ratio based on recent operations."""
        if not self.hit_ratio_window:
            return 0.0  
            
        # Calculate hit ratio over the window
        window_hits = sum(self.hit_ratio_window)
        window_size = len(self.hit_ratio_window)
        
        # Also consider overall hit ratio with some weight
        if self.total_accesses > 0:
            overall_hit_ratio = self.hits / self.total_accesses
            # Weight recent window more heavily (70% recent, 30% overall)
            return (0.7 * (window_hits / window_size)) + (0.3 * overall_hit_ratio)
        else:
            return window_hits / window_size  
