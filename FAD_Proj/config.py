CONFIG = {
    # Total number of items the cache can hold
    "cache_size": 100,

    # Initial decay rates per segment
    "initial_decay_rates": {
        "cold": 0.1,
        "warm": 0.1,
        "hot": 0.05
    },

    # Decay rate bounds for feedback control
    "decay_bounds": {
        "min": 0.01,
        "max": 0.2
    },

    # Promotion thresholds (number of effective accesses before promotion)
    "promotion_thresholds": {
        "cold_to_warm": 3.0,
        "warm_to_hot": 6.0,
        "hot_to_warm": 2.0,  # Demotion thresholds
        "warm_to_cold": 1.0
    },

    # PID controller parameters for each segment
    "pid_params": {
        "cold": {"kp": 1.2, "ki": 0.15, "kd": 0.03},
        "warm": {"kp": 1.5, "ki": 0.2, "kd": 0.05},
        "hot": {"kp": 2.5, "ki": 0.4, "kd": 0.15}
    },

    # Target hit ratios for each segment
    "target_hit_ratios": {
        "cold": 0.2,
        "warm": 0.3,
        "hot": 0.5
    },

    # Eviction cost model parameters
    "eviction_params": {
        "position_costs": {
            "hot": 5.0,
            "warm": 2.5,
            "cold": 1.0
        },
        "weights": {
            "reload": 0.35,
            "position": 0.25,
            "recency": 0.25,
            "promotion": 0.10,
            "decay": 0.05
        },
        "promotion_multiplier": 0.8
    },

    # Performance tracking parameters
    "performance": {
        "window_size": 20,  # Track performance over 20 operations
        "queue_update_interval": 0.1,  # Update queue every 100ms
        "burst_threshold": 0.05  # 50ms threshold for burst detection
    }
}
