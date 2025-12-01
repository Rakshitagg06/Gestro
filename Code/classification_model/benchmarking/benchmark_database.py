"""Benchmark Database Manager
Maintains a persistent CSV database of all benchmark runs with device information.
"""

from pathlib import Path
from datetime import datetime
import platform
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = CURRENT_DIR / "benchmark_complete_data.csv"


class BenchmarkDatabase:
    """Manage persistent benchmark results database."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.ensure_database_exists()

    def ensure_database_exists(self):
        """Create database file with headers if it doesn't exist."""
        if not self.db_path.exists():
            headers = [
                'timestamp',
                'date',
                'time',
                'device_type',
                'cpu_model',
                'gpu_model',
                'npu_model',
                'ram_gb',
                'platform',
                'model_file',
                'model_type',
                'dataset_samples',
                'mean_inference_ms',
                'median_inference_ms',
                'std_inference_ms',
                'min_inference_ms',
                'max_inference_ms',
                'p95_inference_ms',
                'p99_inference_ms',
                'fps',
                'accuracy_percent',
                'total_correct',
                'total_samples'
            ]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.db_path, index=False)
            print(f"Created new benchmark database at {self.db_path}")

    def add_benchmark_result(self, result_data):
        """Add a new benchmark result to the database."""
        self.ensure_database_exists()
        df = pd.read_csv(self.db_path)

        now = datetime.now()
        new_row = {
            'timestamp': result_data.get('timestamp', now.isoformat()),
            'date': result_data.get('date', now.strftime('%Y-%m-%d')),
            'time': result_data.get('time', now.strftime('%H:%M:%S')),
            'device_type': result_data.get('device_type', 'CPU').upper(),
            'cpu_model': result_data.get('cpu_model', 'Unknown'),
            'gpu_model': result_data.get('gpu_model', 'None'),
            'npu_model': result_data.get('npu_model', 'None'),
            'ram_gb': result_data.get('ram_gb', 'Unknown'),
            'platform': result_data.get('platform', platform.platform()),
            'model_file': result_data.get('model_file', 'Unknown'),
            'model_type': result_data.get('model_type', 'Unknown'),
            'dataset_samples': result_data.get('dataset_samples', 0),
            'mean_inference_ms': result_data.get('mean_inference_ms', 0.0),
            'median_inference_ms': result_data.get('median_inference_ms', 0.0),
            'std_inference_ms': result_data.get('std_inference_ms', 0.0),
            'min_inference_ms': result_data.get('min_inference_ms', 0.0),
            'max_inference_ms': result_data.get('max_inference_ms', 0.0),
            'p95_inference_ms': result_data.get('p95_inference_ms', 0.0),
            'p99_inference_ms': result_data.get('p99_inference_ms', 0.0),
            'fps': result_data.get('fps', 0.0),
            'accuracy_percent': result_data.get('accuracy_percent', 0.0),
            'total_correct': result_data.get('total_correct', 0),
            'total_samples': result_data.get('total_samples', 0)
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.db_path, index=False)
        print(f"✓ Benchmark result appended to {self.db_path}")

    def get_all_results(self):
        """Return the full database as a DataFrame."""
        self.ensure_database_exists()
        return pd.read_csv(self.db_path)

    def get_latest_results(self, n=5):
        """Return the n most recent results."""
        df = self.get_all_results()
        return df.tail(n)

    def get_results_by_device(self, device_type):
        """Return all results for a specific device type."""
        df = self.get_all_results()
        return df[df['device_type'] == device_type.upper()]

    def print_summary(self):
        """Print a concise summary of the database contents."""
        df = self.get_all_results()
        if df.empty:
            print("Benchmark database is empty.")
            return

        print("\n" + "=" * 120)
        print(" " * 40 + "BENCHMARK DATABASE SUMMARY")
        print("=" * 120)
        print(f"Total runs: {len(df)}")
        print(f"Location : {self.db_path}")

        device_counts = df['device_type'].value_counts()
        print("\nRuns per device type:")
        for device, count in device_counts.items():
            print(f"  {device}: {count} runs")
        print("=" * 120)

    def print_recent_results(self, n=10):
        """Print the last n results in a table format."""
        df = self.get_all_results()
        if df.empty:
            print("No results in database yet.")
            return

        recent = df.tail(n)
        print("\n" + "=" * 120)
        print(" " * 35 + f"RECENT BENCHMARK RESULTS (Last {len(recent)})")
        print("=" * 120)
        header = f"{'#':<4} | {'Device':<6} | {'Processor/GPU Model':<40} | {'Mean(ms)':<10} | {'FPS':<8} | {'Accuracy':<8}"
        print(header)
        print("-" * 120)

        for idx, (_, row) in enumerate(recent.iterrows(), start=1):
            if row['device_type'] == 'GPU' and row['gpu_model'] != 'None':
                model = str(row['gpu_model'])
            elif row['device_type'] == 'NPU' and row['npu_model'] != 'None':
                model = str(row['npu_model'])
            else:
                model = str(row['cpu_model'])
            if len(model) > 40:
                model = model[:37] + '...'

            row_str = (
                f"{idx:<4} | {row['device_type']:<6} | {model:<40} | "
                f"{row['mean_inference_ms']:<10.4f} | {row['fps']:<8.2f} | {row['accuracy_percent']:<7.2f}%"
            )
            print(row_str)
        print("=" * 120)


if __name__ == "__main__":
    db = BenchmarkDatabase()
    db.print_summary()