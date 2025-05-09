# FAD-DS (Feedback Adaptive Decay with Dynamic Segmentation) Cache

A high-performance, adaptive caching system that uses dynamic segmentation and feedback control to optimize cache performance across various workloads.

## Overview

FAD-DS is an advanced caching system that implements a three-tiered (hot/warm/cold) segmentation strategy with adaptive decay rates and feedback control. The system dynamically adjusts its behavior based on workload patterns and performance metrics to optimize hit rates and resource utilization.

## Key Features

- **Dynamic Segmentation**: Three-tiered cache structure (hot/warm/cold) with fluid segment boundaries
- **Adaptive Decay**: PID-controlled decay rates that adjust based on performance feedback
- **Burst Detection**: Intelligent detection and handling of bursty access patterns
- **Performance Monitoring**: Comprehensive metrics tracking including:
  - Hit/miss ratios
  - CPU utilization
  - Memory usage
  - Segment statistics
  - Promotion/demotion events
- **Benchmark Suite**: Extensive testing framework with multiple workload patterns:
  - Uniform distribution
  - Zipf distribution
  - Bursty access patterns
  - Phase-based workloads
  - Mixed workloads

## Architecture

### Core Components

1. **Cache Segments**
   - Hot segment: Frequently accessed items
   - Warm segment: Moderately accessed items
   - Cold segment: Rarely accessed items

2. **Feedback Controller**
   - PID-based control system
   - Adaptive parameter tuning
   - Performance monitoring and adjustment

3. **Metrics System**
   - Real-time performance tracking
   - CPU and memory monitoring
   - Excel-based logging and analysis

### Testing Matrix

The system includes a comprehensive testing matrix with various scenarios:
- Extreme pressure (T1)
- Low resource (T2)
- Default scenario (T3)
- Medium pressure (T4)
- Balanced load (T5)
- High throughput (T6)
- Maximal load test (T7)

## Usage

### Basic Usage

```python
from cache.fad_ds_cache import FAD_DSCache
from config import CONFIG

# Initialize cache
cache = FAD_DSCache(total_size=1000, config=CONFIG)

# Basic operations
cache.put("key", "value")
value = cache.get("key")
```

### Running Benchmarks

```python
python run_all_benchmarks.py
```

This will execute the full test matrix and generate Excel reports with detailed metrics.

## Performance Metrics

The system tracks various performance metrics:
- Hit rates and ratios
- CPU utilization (delta-based measurement)
- Memory usage
- Segment sizes and distributions
- Promotion/demotion statistics
- Eviction patterns

## Configuration

The system is highly configurable through the `config.py` file, including:
- Initial decay rates
- PID controller parameters
- Segment target ratios
- Performance thresholds
- Burst detection parameters

## Requirements

- Python 3.6+
- psutil
- pandas
- openpyxl
- numpy

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
