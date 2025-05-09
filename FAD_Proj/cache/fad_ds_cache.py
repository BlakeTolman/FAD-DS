# Main implementation of the FAD-DS adaptive segmented cache
from .segment import CacheSegment
from .feedback_controller import FeedbackController
from metrics.monitor import MetricsMonitor
from .entry import CacheEntry
from collections import deque
import time
import math

class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.01):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()
        self.error_history = deque(maxlen=5)  # Shorter history for faster response
        self.max_integral = 10.0  # Prevent integral windup

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time

        # Calculate integral term with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, self.max_integral), -self.max_integral)
        
        # Calculate derivative term with smoothing
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # Compute PID output with more aggressive response
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # Update state
        self.last_error = error
        self.last_time = current_time
        self.error_history.append(error)
        
        return output

class FAD_DSCache:
    def __init__(self, total_size: int, config: dict):
        self.total_size = total_size
        self.config = config
        self.initial_decay_rates = config["initial_decay_rates"].copy()

        # Initialize segments with cold_params
        cold_params = config.get("cold_params", {"freq_weight": 1.2, "target_ratio": 0.1})
        self.cold = CacheSegment(total_size, config["initial_decay_rates"]["cold"], 
                               freq_weight=cold_params["freq_weight"],
                               target_ratio=cold_params["target_ratio"])
        self.warm = CacheSegment(total_size, config["initial_decay_rates"]["warm"])
        self.hot = CacheSegment(total_size, config["initial_decay_rates"]["hot"])
        
        # Track current size
        self._current_size = 0

        self.feedback = FeedbackController(self)
        self.monitor = MetricsMonitor()
        
        # Initialize PID controllers with config parameters
        self.pid_controllers = {
            "cold": PIDController(**config["pid_params"]["cold"]),
            "warm": PIDController(**config["pid_params"]["warm"]),
            "hot": PIDController(**config["pid_params"]["hot"])
        }
        
        self.target_hit_ratios = config["target_hit_ratios"]

        # Initialize eviction parameters from config
        self.eviction_params = config["eviction_params"]
        
        # Track historical performance
        self.performance_history = {
            "hit_ratios": [],
            "eviction_counts": [],
            "promotion_success": [],
            "best_hit_ratio": 0.0,
            "best_params": None,
            "stable_period": 0,  # Track periods of stable performance
            "window_hit_ratios": []  # Track hit ratios in current window
        }
        
        # Performance tracking parameters from config
        self.window_size = config["performance"]["window_size"]
        self._update_counter = 0
        self._last_param_update = time.time()

        # Initialize eviction queue
        self.eviction_queue = []
        self.last_queue_update = time.time()
        self.queue_update_interval = config["performance"]["queue_update_interval"]

        # Burst detection parameters from config
        self._last_access_time = 0
        self._burst_threshold = config["performance"]["burst_threshold"]
        self._is_burst_mode = False

    def reset(self) -> None:
        """Reset the cache to its initial state."""
        # Clear all segments
        self.cold.entries.clear()
        self.warm.entries.clear()
        self.hot.entries.clear()
        
        # Reset decay rates to initial values
        self.cold.update_decay(self.initial_decay_rates["cold"])
        self.warm.update_decay(self.initial_decay_rates["warm"])
        self.hot.update_decay(self.initial_decay_rates["hot"])
        
        # Reset PID controllers
        for controller in self.pid_controllers.values():
            controller.integral = 0
            controller.last_error = 0
            controller.last_time = time.time()
            controller.error_history.clear()
        
        # Reset monitor
        self.monitor.reset()
        
        # Reset feedback controller
        self.feedback = FeedbackController(self)

    def get_total_entries(self) -> int:
        """Get total number of entries across all segments."""
        self._current_size = len(self.cold.entries) + len(self.warm.entries) + len(self.hot.entries)
        return self._current_size

    def get(self, key: str) -> any:
        self.apply_decay()

        # Try segments in order of priority
        for segment in [self.hot, self.warm, self.cold]:
            entry = segment.get(key)
            if entry:
                self.monitor.record_operation("get", key, True)
                self._promote_or_demote(entry)
                self.monitor.record_memory_usage()
                self.monitor.record_cpu_usage()
                self.feedback.update()
                return entry.value

        self.monitor.record_operation("get", key, False)
        self.monitor.record_memory_usage()
        self.monitor.record_cpu_usage()
        self.feedback.update()
        return None

    def _is_in_burst(self) -> bool:
        """Simple burst detection."""
        current_time = time.time()
        interval = current_time - self._last_access_time
        self._last_access_time = current_time
        
        # Simple threshold check
        self._is_burst_mode = interval < self._burst_threshold
        return self._is_burst_mode

    def _promote_or_demote(self, entry: CacheEntry) -> None:
        """Promotion/demotion logic with fluid segments."""
        # Check burst state
        self._is_in_burst()
        
        key = entry.key
        current_time = time.time()
        
        # Get segment statistics
        cold_stats = self.cold.get_statistics()
        warm_stats = self.warm.get_statistics()
        hot_stats = self.hot.get_statistics()
        
        # Calculate relative metrics
        total_size = cold_stats["size"] + warm_stats["size"] + hot_stats["size"]
        total_accesses = cold_stats["total_accesses"] + warm_stats["total_accesses"] + hot_stats["total_accesses"]
        
        # Check which segment contains the entry using the key
        if key in self.cold.entries:
            # Cold to Warm promotion based on access patterns
            if (self._is_burst_mode or 
                entry.access_count >= self.config["promotion_thresholds"]["cold_to_warm"] or 
                entry.decay_score > 0.6):
                value = self.cold.entries.pop(key).value
                self.warm.add(key, value)
                self.monitor.record_promotion("cold", "warm")
                
        elif key in self.warm.entries:
            # Warm to Hot promotion
            if (self._is_burst_mode or 
                entry.access_count >= self.config["promotion_thresholds"]["warm_to_hot"] or 
                entry.decay_score > 0.8):
                value = self.warm.entries.pop(key).value
                self.hot.add(key, value)
                self.monitor.record_promotion("warm", "hot")
            # Warm to Cold demotion
            elif entry.access_count < self.config["promotion_thresholds"]["warm_to_cold"]:
                value = self.warm.entries.pop(key).value
                self.cold.add(key, value)
                self.monitor.record_demotion("warm", "cold")
                
        elif key in self.hot.entries:
            # Hot to Warm demotion
            if entry.access_count < self.config["promotion_thresholds"]["hot_to_warm"]:
                value = self.hot.entries.pop(key).value
                self.warm.add(key, value)
                self.monitor.record_demotion("hot", "warm")

    def put(self, key: str, value: any) -> None:
        """Put a value into the cache with fluid segment sizes."""
        self.apply_decay()

        # Check if key exists in any segment
        for segment in [self.hot, self.warm, self.cold]:
            entry = segment.check_exists(key)
            if entry:
                entry.value = value
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.monitor.record_memory_usage()
                self.monitor.record_cpu_usage()
                self.feedback.update()
                return

        # Ensure we have space for the new item by evicting if necessary
        while self.get_total_entries() >= self.total_size:
            self._evict()

        # Add new items to cold segment by default
        self.cold.add(key, value)
        self._current_size += 1
            
        self.monitor.record_memory_usage()
        self.monitor.record_cpu_usage()
        self.feedback.update()

    def _detect_performance_drop(self):
        """Detect if there's a significant performance drop."""
        if len(self.performance_history["window_hit_ratios"]) < self.window_size:
            return False, 0.0

        current_window = self.performance_history["window_hit_ratios"][-self.window_size:]
        window_avg = sum(current_window) / self.window_size
        
        # Compare with best performance
        if self.performance_history["best_hit_ratio"] > 0:
            drop = self.performance_history["best_hit_ratio"] - window_avg
            if drop > 0.05:  # Performance dropped by more than 5%
                return True, drop
        
        return False, 0.0

    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on performance feedback."""
        if len(self.performance_history["hit_ratios"]) < self.window_size:
            return

        # Calculate current performance metrics
        current_hit_ratio = sum(self.performance_history["hit_ratios"][-self.window_size:]) / self.window_size
        performance_trend = (current_hit_ratio - 
                           sum(self.performance_history["hit_ratios"][-2*self.window_size:-self.window_size]) / 
                           self.window_size)

        # Update best known parameters if we have a new best
        if current_hit_ratio > self.performance_history["best_hit_ratio"]:
            self.performance_history["best_hit_ratio"] = current_hit_ratio
            self.performance_history["best_params"] = {
                "position_costs": self.eviction_params["position_costs"].copy(),
                "weights": self.eviction_params["weights"].copy(),
                "promotion_multiplier": self.eviction_params["promotion_multiplier"]
            }
            self.performance_history["stable_period"] += 1

        # If performance is dropping significantly, revert to best known parameters
        if performance_trend < -0.1 and self.performance_history["stable_period"] > 0:
            if self.performance_history["best_params"]:
                self.eviction_params["position_costs"] = self.performance_history["best_params"]["position_costs"].copy()
                self.eviction_params["weights"] = self.performance_history["best_params"]["weights"].copy()
                self.eviction_params["promotion_multiplier"] = self.performance_history["best_params"]["promotion_multiplier"]
                self.performance_history["stable_period"] = 0

    def _calculate_eviction_cost(self, entry, segment_name, segment_stats):
        """Calculate the cost of evicting an item based on multiple factors."""
        current_time = time.time()
        
        # Update adaptive parameters
        self._update_adaptive_parameters()
        
        # 1. Reloading Cost (based on access frequency)
        creation_time = getattr(entry, 'creation_time', entry.last_accessed)
        time_in_cache = current_time - creation_time
        access_rate = entry.access_count / max(1.0, time_in_cache)
        reload_cost = min(8.0, access_rate * 3.0)
        
        # 2. Position Cost (using adaptive costs)
        position_cost = self.eviction_params["position_costs"][segment_name]
        
        # 3. Temporal Locality Cost (using segment's decay rate)
        time_since_access = current_time - entry.last_accessed
        segment = getattr(self, segment_name)
        recency_factor = math.exp(-segment.decay_rate * time_since_access)
        
        # 4. Promotion Investment Cost (using adaptive multiplier)
        promotion_count = getattr(entry, 'promotion_count', 0)
        promotion_cost = self.eviction_params["promotion_multiplier"] * promotion_count
        
        # 5. Decay Score Consideration
        decay_penalty = 1.0 - entry.decay_score
        
        # Combine costs using adaptive weights
        weights = self.eviction_params["weights"]
        total_cost = (
            weights["reload"] * reload_cost +
            weights["position"] * position_cost +
            weights["recency"] * recency_factor +
            weights["promotion"] * promotion_cost +
            weights["decay"] * decay_penalty
        )
        
        # Record promotion success if this is a promoted item
        if promotion_count > 0:
            success = entry.access_count > segment_stats[segment_name]["avg_access_count"]
            self.performance_history["promotion_success"].append(success)
        
        return total_cost

    def _update_eviction_queue(self) -> None:
        """Update the eviction queue with current candidates."""
        current_time = time.time()
        
        # Increase update interval for better stability
        if current_time - self.last_queue_update < self.queue_update_interval:
            return

        self.last_queue_update = current_time
        self.eviction_queue = []

        # Calculate segment statistics with safety checks
        segment_stats = {}
        for segment_name, segment in [("hot", self.hot), ("warm", self.warm), ("cold", self.cold)]:
            entries = segment.entries
            num_entries = len(entries)
            
            if num_entries == 0:
                segment_stats[segment_name] = {
                    "size": 0,
                    "avg_access_count": 1.0,
                    "avg_age": 1.0,
                    "hit_ratio": 0.0
                }
                continue
                
            total_access_count = sum(entry.access_count for entry in entries.values())
            total_age = sum(current_time - entry.last_accessed for entry in entries.values())
            
            # Add hit ratio to segment stats
            hit_ratio = segment.get_hit_ratio() if hasattr(segment, 'get_hit_ratio') else 0.0
            
            segment_stats[segment_name] = {
                "size": num_entries,
                "avg_access_count": max(1.0, total_access_count / num_entries),
                "avg_age": max(1.0, total_age / num_entries),
                "hit_ratio": hit_ratio
            }

        # Add candidates from each segment to the queue with cost consideration
        for segment_name, segment in [("hot", self.hot), ("warm", self.warm), ("cold", self.cold)]:
            if not segment.entries:
                continue
                
            for key, entry in segment.entries.items():
                # Calculate basic scores
                current_age = current_time - entry.last_accessed
                age_score = current_age / segment_stats[segment_name]["avg_age"]
                access_score = 1 - (entry.access_count / segment_stats[segment_name]["avg_access_count"])
                decay_score = 1 - entry.decay_score
                
                # Calculate eviction cost
                eviction_cost = self._calculate_eviction_cost(entry, segment_name, segment_stats)
                
                # Adjust cost based on segment hit ratio
                hit_ratio = segment_stats[segment_name]["hit_ratio"]
                if hit_ratio > 0.8:  # Protect high-performing segments
                    eviction_cost *= 1.5
                
                # Combine scores with cost consideration
                total_score = (
                    0.4 * min(5.0, age_score) +
                    0.3 * max(0.0, min(1.0, access_score)) +
                    0.3 * max(0.0, min(1.0, decay_score))
                )
                
                # Adjust final score based on eviction cost
                final_score = total_score * (1.0 + eviction_cost)

                self.eviction_queue.append((final_score, segment_name, key))

        # Sort queue by score (lowest score = most likely to evict)
        self.eviction_queue.sort(key=lambda x: x[0])

    def _evict(self) -> None:
        """Evict items based on the eviction queue."""
        # Update the eviction queue if needed
        self._update_eviction_queue()
        
        if not self.eviction_queue:
            # Fallback to basic LRU if queue is empty
            segments = {
                "hot": self.hot,
                "warm": self.warm,
                "cold": self.cold
            }
            largest_segment = max(segments.items(), key=lambda x: len(x[1].entries))
            segment_name, segment = largest_segment
            
            if len(segment.entries) > 0:
                lru_key = min(segment.entries.items(), 
                             key=lambda x: (x[1].access_count / max(1, time.time() - x[1].last_accessed)))[0]
                segment.entries.pop(lru_key)
                self._current_size -= 1
                self.monitor.record_eviction(segment_name, lru_key)
            return
            
        # Get the entry with lowest score (most likely to evict)
        _, segment_name, key = self.eviction_queue[0]
        
        # Remove from the appropriate segment
        segment = getattr(self, segment_name)
        if key in segment.entries:
            segment.entries.pop(key)
            self._current_size -= 1
            self.monitor.record_eviction(segment_name, key)
            
        # Remove from queue
        self.eviction_queue.pop(0)

    def apply_decay(self) -> None:
        for segment in [self.cold, self.warm, self.hot]:
            segment.apply_decay()

    def summary(self) -> dict:
        return self.monitor.summary()
