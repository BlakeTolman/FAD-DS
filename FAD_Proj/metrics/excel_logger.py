import pandas as pd
import numpy as np
import os

class ExcelLogger:
    _file_cleared_this_run = False  # Class variable to ensure file is only cleared once per run

    def __init__(self, filename="cache_metrics.xlsx"):
        self.filename = filename
        self.records = {}

    def log(self, step, hit_rate, hits, misses, memory_mb, cpu_time_delta, timestamp,
            cold_size, warm_size, hot_size,
            cold_decay, warm_decay, hot_decay,
            cache_name, workload_name,
            promotions=0, demotions=0, evictions=0):
        # Initialize cache records if not exists
        if cache_name not in self.records:
            self.records[cache_name] = []
        self.records[cache_name].append({
            "workload_name": workload_name,
            "step": step,
            "hit_rate": hit_rate,
            "hits": hits,
            "misses": misses,
            "memory_mb": memory_mb,
            "cpu_time_delta": cpu_time_delta,  # Windowed average CPU percent (now named cpu_time_delta)
            "seconds": timestamp,  # Store raw seconds
            "cold_size": cold_size,
            "warm_size": warm_size,
            "hot_size": hot_size,
            "cold_decay": cold_decay,
            "warm_decay": warm_decay,
            "hot_decay": hot_decay,
            "promotions": promotions,
            "demotions": demotions,
            "evictions": evictions
        })

    def export(self):
        """Append the logged records to an Excel file with separate tabs for each cache. Clears the file at the start of a run. Removes duplicates and sorts by workload_name and step in custom order."""
        from openpyxl import load_workbook
        workload_order = ['Uniform', 'Zipf', 'Bursty', 'Phase', 'Mixed']
        # Clear the file at the start of the first export in this run
        if not ExcelLogger._file_cleared_this_run and os.path.exists(self.filename):
            os.remove(self.filename)
            ExcelLogger._file_cleared_this_run = True
        if os.path.exists(self.filename):
            with pd.ExcelWriter(self.filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                for cache_name, records in self.records.items():
                    df_new = pd.DataFrame(records)
                    try:
                        df_existing = pd.read_excel(self.filename, sheet_name=cache_name)
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    except ValueError:
                        df_combined = df_new
                    # Remove duplicates
                    df_combined = df_combined.drop_duplicates(subset=['workload_name', 'step'], keep='first')
                    # Set custom workload order
                    df_combined['workload_name'] = pd.Categorical(df_combined['workload_name'], categories=workload_order, ordered=True)
                    # Sort by workload_name and step
                    df_combined = df_combined.sort_values(['workload_name', 'step'], kind='stable')
                    numeric_columns = ['hit_rate', 'memory_mb', 'cpu_time_delta', 'seconds', 
                                     'cold_decay', 'warm_decay', 'hot_decay', 'promotions', 'demotions', 'evictions']
                    for col in numeric_columns:
                        df_combined[col] = df_combined[col].astype(np.float64)
                    float_format = '%.15f'
                    df_combined.to_excel(writer, sheet_name=cache_name, index=False, float_format=float_format)
        else:
            with pd.ExcelWriter(self.filename, engine='openpyxl') as writer:
                for cache_name, records in self.records.items():
                    df = pd.DataFrame(records)
                    # Set custom workload order
                    df['workload_name'] = pd.Categorical(df['workload_name'], categories=workload_order, ordered=True)
                    # Sort by workload_name and step
                    df = df.sort_values(['workload_name', 'step'], kind='stable')
                    numeric_columns = ['hit_rate', 'memory_mb', 'cpu_time_delta', 'seconds', 
                                     'cold_decay', 'warm_decay', 'hot_decay', 'promotions', 'demotions', 'evictions']
                    for col in numeric_columns:
                        df[col] = df[col].astype(np.float64)
                    float_format = '%.15f'
                    df.to_excel(writer, sheet_name=cache_name, index=False, float_format=float_format)
