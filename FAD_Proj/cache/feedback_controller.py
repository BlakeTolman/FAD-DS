# Feedback controller for adaptive cache segment tuning
from time import time
import math
import random

class FeedbackController:
    def __init__(self, cache):
        self.cache = cache
        self.last_update_time = 0
        self.cooldown_period = 0.01  # More responsive
        self.learning_rate = 0.2
        self.history_size = 3
        self.hit_history = {"cold": [], "warm": [], "hot": []}
        # More balanced segment weights with emphasis on warm
        self.segment_weights = {"cold": 0.8, "warm": 1.5, "hot": 1.2}
        self.min_decay = 0.1
        self.max_decay = 0.9
        # Track stagnation and utilization
        self.stagnation_counters = {"cold": 0, "warm": 0, "hot": 0}
        self.last_decay_rates = {"cold": 0, "warm": 0, "hot": 0}
        self.stagnation_threshold = 1  # More frequent changes
        self.utilization_history = {"cold": [], "warm": [], "hot": []}
        self.utilization_window = 5  # Reduced from 10 for faster response

    def update(self) -> None:
        now = time()
        if now - self.last_update_time < self.cooldown_period:
            return
        self.last_update_time = now

        # Calculate segment utilizations
        total_size = self.cache.total_size
        for segment_name, segment in zip(["cold", "warm", "hot"],
                                       [self.cache.cold, self.cache.warm, self.cache.hot]):
            utilization = len(segment.entries) / total_size
            self.utilization_history[segment_name].append(utilization)
            if len(self.utilization_history[segment_name]) > self.utilization_window:
                self.utilization_history[segment_name].pop(0)

        for segment_name, segment in zip(["cold", "warm", "hot"],
                                       [self.cache.cold, self.cache.warm, self.cache.hot]):
            old_decay = segment.decay_rate
            hit_ratio = segment.get_hit_ratio()
            self.hit_history[segment_name].append(hit_ratio)
            if len(self.hit_history[segment_name]) > self.history_size:
                self.hit_history[segment_name].pop(0)
            weights = [0.8, 0.15, 0.05]
            smoothed_ratio = sum(w * r for w, r in zip(weights, self.hit_history[segment_name][-3:]))
            avg_utilization = sum(self.utilization_history[segment_name]) / len(self.utilization_history[segment_name])
            self.adjust_learning_rate(hit_ratio, avg_utilization)
            if abs(old_decay - self.last_decay_rates[segment_name]) < 0.001:
                self.stagnation_counters[segment_name] += 1
            else:
                self.stagnation_counters[segment_name] = 0
            self.last_decay_rates[segment_name] = old_decay
            target_ratio = self.cache.target_hit_ratios[segment_name]
            error = target_ratio - smoothed_ratio
            
            # --- Aggressive Utilization-based Adjustments ---
            if segment_name == "cold" and avg_utilization > 0.8:
                new_decay = min(old_decay * 1.25, self.cache.warm.decay_rate - 0.2, self.max_decay)
            elif segment_name == "warm" and avg_utilization < 0.1:
                new_decay = max(old_decay * 0.75, self.min_decay)
            elif segment_name == "hot" and avg_utilization < 0.05:
                # Hot is way underutilized, force decay down aggressively
                new_decay = max(old_decay * 0.5, self.min_decay)
                # Optionally, lower the promotion threshold from warm to hot
                if hasattr(self.cache, 'config') and 'promotion_thresholds' in self.cache.config:
                    self.cache.config['promotion_thresholds']['warm_to_hot'] = max(0.1, self.cache.config['promotion_thresholds']['warm_to_hot'] * 0.7)
            # --- End Aggressive Utilization-based Adjustments ---
            elif avg_utilization < 0.2:
                new_decay = old_decay * 0.9
            elif avg_utilization > 0.8:
                new_decay = old_decay * 1.1
            else:
                if abs(error) > 0.005:
                    if segment_name == "warm":
                        adjustment = 1.0 + (error * self.learning_rate * self.segment_weights[segment_name] * 1.5)
                    else:
                        adjustment = 1.0 + (error * self.learning_rate * self.segment_weights[segment_name])
                    # Add a small random exploration term
                    exploration = random.uniform(-0.01, 0.01)
                    adjustment += exploration
                    new_decay = old_decay * adjustment
                else:
                    new_decay = old_decay
            # Replace clamping logic: all segments use absolute min/max
            new_decay = max(self.min_decay, min(new_decay, self.max_decay))
            # Enforce hot < warm < cold
            epsilon = 0.01
            if segment_name == "cold":
                new_decay = max(new_decay, self.cache.warm.decay_rate + epsilon)
                new_decay = min(new_decay, self.max_decay)
            elif segment_name == "warm":
                new_decay = max(new_decay, self.cache.hot.decay_rate + epsilon)
                new_decay = min(new_decay, self.cache.cold.decay_rate - epsilon)
            elif segment_name == "hot":
                new_decay = min(new_decay, self.cache.warm.decay_rate - epsilon)
                new_decay = max(new_decay, self.min_decay)
            segment.update_decay(new_decay)
            self.cache.monitor.record_decay_change(segment_name, new_decay)

    def adjust_learning_rate(self, hit_ratio, utilization):
        # More dynamic learning rate adjustment based on both hit ratio and utilization
        if hit_ratio < 0.1 or utilization < 0.2:
            self.learning_rate = 0.3  # Very aggressive for poor performance or underutilization
        elif hit_ratio < 0.3 or utilization < 0.4:
            self.learning_rate = 0.2  # Aggressive for medium performance or low utilization
        else:
            self.learning_rate = 0.15  # Still significant for good performance

    def dynamic_promotion_thresholds(self, name, hit_ratio) -> None:
        # More responsive threshold adjustments
        if name == "cold":
            if len(self.hit_history["cold"]) >= 3:
                smoothed = sum(self.hit_history["cold"][-3:]) / 3
                threshold = self.cache.config["promotion_thresholds"]["cold_to_warm"]
                if smoothed < 0.2:
                    self.cache.config["promotion_thresholds"]["cold_to_warm"] = max(0.3, threshold - 0.4)
                elif smoothed > 0.4:
                    self.cache.config["promotion_thresholds"]["cold_to_warm"] = min(1.8, threshold + 0.4)

        if name == "warm":
            if len(self.hit_history["warm"]) >= 3:
                smoothed = sum(self.hit_history["warm"][-3:]) / 3
                threshold = self.cache.config["promotion_thresholds"]["warm_to_hot"]
                if smoothed < 0.2:
                    self.cache.config["promotion_thresholds"]["warm_to_hot"] = max(0.2, threshold - 0.4)
                elif smoothed > 0.4:
                    self.cache.config["promotion_thresholds"]["warm_to_hot"] = min(1.5, threshold + 0.4)
