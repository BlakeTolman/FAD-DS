from cache.fad_ds_cache import FAD_DSCache
from config import CONFIG
from workload.synthetic_generator import WorkloadGenerator
from metrics.visualizer import MetricsVisualizer
from benchmarks.lru import LRUCache
from benchmarks.lfu import LFUCacheWrapper
from benchmarks.arc import ARCCache
from benchmarks.fifo import FIFOCache
from metrics.excel_logger import ExcelLogger
import random
import psutil
import os
import time
import gc
import sys
import json

# Global Excel logger for all cache+workload results
global_logger = ExcelLogger(filename="all_cache_metrics.xlsx")

def test_cache(cache, workload: list[tuple[str, str]], cache_name: str, workload_name: str) -> None:
    step = 0
    process = psutil.Process(os.getpid())
    # Create a per-run Excel file for this cache/workload
    excel_filename = f"subprocess_{cache_name}_{workload_name}.xlsx"
    logger = ExcelLogger(filename=excel_filename)
    
    # Force garbage collection and clear memory
    gc.collect()
    process.memory_info().rss  # Force memory update
    
    # Get initial memory usage and start time with high precision
    initial_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    start_time = time.perf_counter()  # Use high-precision timer
    cpu_start = process.cpu_percent(interval=None)

    # For per-window CPU percent (delta method)
    prev_cpu_time = process.cpu_times().user + process.cpu_times().system
    prev_wall_time = time.perf_counter()
    cpu_percent_window = []
    window_size = 50
    last_cpu_percent = 0.0

    # Reset cache and monitor before each workload test
    if hasattr(cache, 'monitor'):
        cache.monitor.reset()
    if isinstance(cache, FAD_DSCache):
        cache.reset()

    # Clear any existing memory samples
    if hasattr(cache, 'monitor'):
        cache.monitor.memory_usage = []
        cache.monitor.cpu_usage = []

    # Process workload in order to maintain access patterns
    for key, value in workload:
        # First try to get the key (should miss if not in cache)
        result = cache.get(key)
        
        # Log metrics after get operation
        if hasattr(cache, 'monitor'):
            # Get current memory usage and current time with high precision
            current_memory = process.memory_info().rss / (1024 * 1024)
            current_time = time.perf_counter() - start_time  # This will always be positive

            # Per-step CPU percent (delta method, but log every 50 steps)
            cpu_time_now = process.cpu_times().user + process.cpu_times().system
            wall_time_now = time.perf_counter()
            delta_cpu = cpu_time_now - prev_cpu_time
            delta_wall = wall_time_now - prev_wall_time
            cpu_percent = (delta_cpu / delta_wall) * 100 if delta_wall > 0 else 0.0
            cpu_percent_window.append(cpu_percent)
            prev_cpu_time = cpu_time_now
            prev_wall_time = wall_time_now

            # Log every 50 steps, otherwise log last known value
            if (step + 1) % window_size == 0 or (step + 1) == len(workload):
                avg_cpu_percent = sum(cpu_percent_window) / len(cpu_percent_window)
                last_cpu_percent = avg_cpu_percent
                cpu_percent_window = []
            # For other steps, use last known value
            cpu_percent_to_log = last_cpu_percent

            # For FAD-DS cache
            if hasattr(cache, 'cold'):
                logger.log(
                    step=step,
                    hit_rate=cache.monitor.get_hit_ratio(),
                    hits=cache.monitor.hits,
                    misses=cache.monitor.misses,
                    memory_mb=current_memory,
                    cpu_time_delta=cpu_percent_to_log,
                    timestamp=current_time,
                    cold_size=len(cache.cold.entries),
                    warm_size=len(cache.warm.entries),
                    hot_size=len(cache.hot.entries),
                    cold_decay=cache.cold.decay_rate,
                    warm_decay=cache.warm.decay_rate,
                    hot_decay=cache.hot.decay_rate,
                    cache_name=cache_name,
                    workload_name=workload_name,
                    promotions=cache.monitor.get_promotion_count(),
                    demotions=cache.monitor.get_demotion_count(),
                    evictions=cache.monitor.get_eviction_count()
                )
            # For benchmark caches
            else:
                logger.log(
                    step=step,
                    hit_rate=cache.monitor.get_hit_ratio(),
                    hits=cache.monitor.hits,
                    misses=cache.monitor.misses,
                    memory_mb=current_memory,
                    cpu_time_delta=cpu_percent_to_log,
                    timestamp=current_time,
                    cold_size=0,
                    warm_size=0,
                    hot_size=0,
                    cold_decay=0,
                    warm_decay=0,
                    hot_decay=0,
                    cache_name=cache_name,
                    workload_name=workload_name,
                    promotions=0,
                    demotions=0,
                    evictions=0
                )
        
        # Then put the key-value pair in the cache
        cache.put(key, value)
        step += 1

    cpu_end = process.cpu_percent(interval=0.1)
    avg_cpu = (cpu_end - cpu_start) if cpu_end > cpu_start else 0.0

    logger.export()
    # Print summary for terminal only
    summary = cache.summary()
    summary["avg_cpu_percent"] = avg_cpu
    if hasattr(cache, 'cold'):
        summary["cold_size"] = len(cache.cold.entries)
        summary["warm_size"] = len(cache.warm.entries)
        summary["hot_size"] = len(cache.hot.entries)
        summary["cold_decay"] = cache.cold.decay_rate
        summary["warm_decay"] = cache.warm.decay_rate
        summary["hot_decay"] = cache.hot.decay_rate
        # Add decay change counts for each segment
        summary["cold_decay_changes"] = len(cache.monitor.decay_rates["cold"])
        summary["warm_decay_changes"] = len(cache.monitor.decay_rates["warm"])
        summary["hot_decay_changes"] = len(cache.monitor.decay_rates["hot"])
    else:
        summary["cold_size"] = 0
        summary["warm_size"] = 0
        summary["hot_size"] = 0
        summary["cold_decay"] = 0
        summary["warm_decay"] = 0
        summary["hot_decay"] = 0
        summary["cold_decay_changes"] = 0
        summary["warm_decay_changes"] = 0
        summary["hot_decay_changes"] = 0
    print(f"{cache_name} with {workload_name} - {summary}")

def main():
    CONFIG["cache_size"] = 500

    caches = [
        ("FAD-DS", FAD_DSCache(total_size=CONFIG["cache_size"], config=CONFIG)),
        ("LRU", LRUCache(max_size=CONFIG["cache_size"])),
        ("LFU", LFUCacheWrapper(max_size=CONFIG["cache_size"])),
        ("ARC", ARCCache(max_size=CONFIG["cache_size"])),
        ("FIFO", FIFOCache(max_size=CONFIG["cache_size"]))
    ]

    gen = WorkloadGenerator(key_space_size=5000, num_requests=50000)
    workloads = [
        ("Uniform", lambda: gen.generate_uniform_workload()),
        ("Zipf", lambda: gen.generate_zipf_workload(alpha=1.2)),
        ("Bursty", lambda: gen.generate_bursty_workload(burst_size=5, burst_freq=0.2)),
        ("Phase", lambda: gen.generate_phase_workload(phase_length=100, num_phases=10)),
        ("Mixed", lambda: gen.generate_mixed_workload(zipf_alpha=1.2, burst_freq=0.1))
    ]

    for cache_name, cache in caches:
        for workload_name, generate_workload in workloads:
            workload = generate_workload()
            test_cache(cache, workload, cache_name, workload_name)
        print("----------------------------------------------------------------------------------")
    global_logger.export()

def main_single(cache_name, workload_name):
    CONFIG["cache_size"] = 500

    caches = {
        "FAD-DS": FAD_DSCache(total_size=CONFIG["cache_size"], config=CONFIG),
        "LRU": LRUCache(max_size=CONFIG["cache_size"]),
        "LFU": LFUCacheWrapper(max_size=CONFIG["cache_size"]),
        "ARC": ARCCache(max_size=CONFIG["cache_size"]),
        "FIFO": FIFOCache(max_size=CONFIG["cache_size"])
    }

    gen = WorkloadGenerator(key_space_size=5000, num_requests=50000)
    workloads = {
        "Uniform": lambda: gen.generate_uniform_workload(),
        "Zipf": lambda: gen.generate_zipf_workload(alpha=1.2),
        "Bursty": lambda: gen.generate_bursty_workload(burst_size=5, burst_freq=0.2),
        "Phase": lambda: gen.generate_phase_workload(phase_length=100, num_phases=10),
        "Mixed": lambda: gen.generate_mixed_workload(zipf_alpha=1.2, burst_freq=0.1)
    }

    cache = caches[cache_name]
    workload = workloads[workload_name]()
    # Always use the same logger filename for all runs
    global global_logger
    global_logger = ExcelLogger(filename="all_cache_metrics.xlsx")
    test_cache(cache, workload, cache_name, workload_name)
    global_logger.export()

def run_single_test(config_path: str):
    """Run a single test with the given configuration."""
    # Load test configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set up cache
    cache_size = config["cache_size"]
    cache_name = config["cache_name"]
    
    if cache_name == "FAD-DS":
        cache = FAD_DSCache(total_size=cache_size, config=CONFIG)
    elif cache_name == "LRU":
        cache = LRUCache(max_size=cache_size)
    elif cache_name == "LFU":
        cache = LFUCacheWrapper(max_size=cache_size)
    elif cache_name == "ARC":
        cache = ARCCache(max_size=cache_size)
    elif cache_name == "FIFO":
        cache = FIFOCache(max_size=cache_size)
    else:
        raise ValueError(f"Unknown cache type: {cache_name}")
    
    # Use the provided workload data
    workload = config["workload_data"]
    workload_name = config["workload_name"]
    
    # Run the test
    test_cache(cache, workload, cache_name, workload_name)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Run single test with configuration file
        run_single_test(sys.argv[1])
    else:
        print("Usage: python main.py <config_file>")
        sys.exit(1)
