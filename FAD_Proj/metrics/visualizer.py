import matplotlib.pyplot as plt
from .monitor import MetricsMonitor
from typing import Optional

class MetricsVisualizer:
    def __init__(self, monitor: MetricsMonitor):
        """Initialize with a MetricsMonitor instance."""
        self.monitor = monitor

    def plot_hit_miss_ratio(self, output_file: Optional[str] = None) -> None:
        """Plot the cumulative hit/miss ratio over time."""
        times = [op["time"] - self.monitor.operations[0]["time"] for op in self.monitor.operations]
        hit_ratios = [sum(1 for op in self.monitor.operations[:i+1] if op["hit"]) / (i + 1)
                      for i in range(len(self.monitor.operations))]

        plt.figure(figsize=(10, 6))
        plt.plot(times, hit_ratios, label="Hit Ratio")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Hit Ratio")
        plt.title("Hit Ratio Over Time")
        plt.grid(True)
        plt.legend()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()

    def plot_decay_rates(self, output_file: Optional[str] = None) -> None:
        """Plot decay rate changes for each segment over time."""
        plt.figure(figsize=(10, 6))
        for segment, changes in self.monitor.decay_rates.items():
            if changes:
                times = [change["time"] - changes[0]["time"] for change in changes]
                rates = [change["decay_rate"] for change in changes]
                plt.plot(times, rates, label=f"{segment.capitalize()} Decay Rate")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Decay Rate")
        plt.title("Decay Rate Changes Over Time")
        plt.grid(True)
        plt.legend()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()

    def plot_memory_usage(self, output_file: Optional[str] = None) -> None:
        """Plot memory usage over time."""
        times = [m["time"] - self.monitor.memory_usage[0]["time"] for m in self.monitor.memory_usage]
        memory = [m["memory_mb"] for m in self.monitor.memory_usage]

        plt.figure(figsize=(10, 6))
        plt.plot(times, memory, label="Memory Usage (MB)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage Over Time")
        plt.grid(True)
        plt.legend()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close()