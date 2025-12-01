"""
Comprehensive Inference Time Benchmarking Script
This script evaluates the model on the dataset, measures inference times,
and generates detailed reports with visualizations.

Supports: CPU, GPU, and NPU (Neural Processing Unit) acceleration
"""

import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from pathlib import Path
import warnings
import sys
import platform

# Ensure local imports work no matter where the script is executed from
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

ROOT_DIR = CURRENT_DIR.parent
DEFAULT_MODEL_PATH = ROOT_DIR / 'your_model.tflite'
DEFAULT_LABEL_MAP = ROOT_DIR / 'label_map.npy'
DEFAULT_DATASET_PATH = ROOT_DIR / 'gestures_train.csv'
DEFAULT_OUTPUT_DIR = CURRENT_DIR / 'benchmark_results'
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_DATABASE_PATH = CURRENT_DIR / 'benchmark_complete_data.csv'

from benchmark_database import BenchmarkDatabase
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DeviceDetector:
    """Detect and configure available compute devices"""
    
    @staticmethod
    def detect_devices():
        """Detect available compute devices"""
        devices = {'cpu': True}  # CPU always available
        
        # Check for GPU
        try:
            gpus = tf.config.list_physical_devices('GPU')
            devices['gpu'] = len(gpus) > 0
            if devices['gpu']:
                print(f"✓ GPU detected: {len(gpus)} device(s)")
                for gpu in gpus:
                    print(f"  - {gpu.name}")
        except:
            devices['gpu'] = False
        
        # Check for NPU/TPU (various implementations)
        devices['npu'] = False
        try:
            # Check for Edge TPU
            import tflite_runtime.interpreter as tflite
            delegates = []
            try:
                delegates.append(tflite.load_delegate('libedgetpu.so.1'))
                devices['npu'] = True
                print(f"✓ Edge TPU (NPU) detected")
            except:
                pass
        except:
            pass
        
        # Check for other accelerators
        try:
            other_devices = tf.config.list_physical_devices()
            for device in other_devices:
                if 'XLA' in device.device_type or 'TPU' in device.device_type:
                    devices['npu'] = True
                    print(f"✓ Accelerator detected: {device.device_type}")
        except:
            pass
        
        print(f"✓ CPU available")
        if not devices['gpu']:
            print("✗ GPU not detected")
        if not devices['npu']:
            print("✗ NPU/TPU not detected")
        
        return devices
    
    @staticmethod
    def configure_device(device_type):
        """Configure TensorFlow to use specific device"""
        if device_type == 'cpu':
            # Force CPU only
            tf.config.set_visible_devices([], 'GPU')
            print("Configured to use: CPU only")
            return True
        
        elif device_type == 'gpu':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to avoid OOM
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Configured to use: GPU ({len(gpus)} device(s))")
                    return True
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
                    return False
            else:
                print("GPU requested but not available")
                return False
        
        elif device_type == 'npu':
            print("NPU configuration requires TFLite with delegate")
            return True
        
        return False

class InferenceBenchmark:
    """Comprehensive benchmarking for gesture recognition model"""
    
    def __init__(self, model_path, label_map_path, dataset_path, output_dir="benchmark_results", device='cpu'):
        """
        Initialize the benchmark
        
        Args:
            model_path: Path to the model file (.h5 or .tflite)
            label_map_path: Path to label_map.npy
            dataset_path: Path to gestures.csv
            output_dir: Directory to save results
            device: Device to use ('cpu', 'gpu', 'npu', or 'all' for comparison)
        """
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        self.device_info = self._get_device_info()
        
        # Load label map
        self.label_map = np.load(label_map_path, allow_pickle=True).item()
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = pd.read_csv(dataset_path, header=None)
        self.gestures = self.dataset.iloc[:, 0].values  # First column is label
        self.features = self.dataset.iloc[:, 1:].values  # Rest are features
        
        print(f"Dataset loaded: {len(self.dataset)} samples")
        print(f"Gestures: {sorted(set(self.gestures))}")
        
        # Load model
        self.model_type = self._detect_model_type()
        self._load_model()
        
        # Results storage
        self.results = {
            'per_frame_times': [],
            'per_gesture_times': {},
            'per_gesture_stats': {},
            'predictions': [],
            'ground_truth': [],
            'correct_predictions': [],
            'device_used': device
        }
    
    def _get_device_info(self):
        """Get detailed device information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': sys.version.split()[0],
            'tensorflow_version': tf.__version__,
            'cpu_model': 'Unknown',
            'gpu_model': 'None',
            'npu_model': 'None',
            'ram_total': 'Unknown'
        }
        
        # Get CPU model
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            info['cpu_model'] = line.split(':')[1].strip()
                            break
            elif platform.system() == 'Darwin':  # macOS
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                info['cpu_model'] = result.stdout.strip()
            elif platform.system() == 'Windows':
                import subprocess
                result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                      capture_output=True, text=True)
                info['cpu_model'] = result.stdout.split('\n')[1].strip()
        except:
            pass
        
        # Get RAM info
        try:
            if platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            ram_kb = int(line.split()[1])
                            ram_gb = ram_kb / (1024 ** 2)
                            info['ram_total'] = f"{ram_gb:.1f} GB"
                            break
        except:
            pass
        
        # GPU info
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info['gpu_count'] = len(gpus)
                info['gpu_names'] = [gpu.name for gpu in gpus]
                
                # Try to get GPU model using nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_names = result.stdout.strip().split('\n')
                        info['gpu_model'] = gpu_names[0] if gpu_names else 'Unknown GPU'
                except:
                    info['gpu_model'] = 'GPU Available (model unknown)'
        except:
            pass
        
        # NPU info
        if self.device == 'npu':
            try:
                # Check for Edge TPU
                import subprocess
                result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
                if 'Edge TPU' in result.stdout or 'Google Inc' in result.stdout:
                    info['npu_model'] = 'Google Coral Edge TPU'
                else:
                    info['npu_model'] = 'NPU Available (model unknown)'
            except:
                info['npu_model'] = 'NPU Delegate Enabled'
        
        return info
        
    def _detect_model_type(self):
        """Detect if model is Keras (.h5) or TFLite (.tflite)"""
        if self.model_path.endswith('.h5'):
            return 'keras'
        elif self.model_path.endswith('.tflite'):
            return 'tflite'
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
    
    def _load_model(self):
        """Load the model based on its type and device"""
        print(f"Loading {self.model_type.upper()} model from {self.model_path}...")
        print(f"Target device: {self.device.upper()}")
        
        if self.model_type == 'keras':
            self.model = tf.keras.models.load_model(self.model_path)
            print("Keras model loaded successfully")
        else:  # tflite
            # Try to use delegates for NPU
            if self.device == 'npu':
                try:
                    import tflite_runtime.interpreter as tflite
                    delegates = []
                    
                    # Try Edge TPU
                    try:
                        delegates.append(tflite.load_delegate('libedgetpu.so.1'))
                        print("Using Edge TPU delegate")
                    except:
                        pass
                    
                    # Try NNAPI (Android NPU)
                    if not delegates:
                        try:
                            from tensorflow.lite.python.interpreter import load_delegate
                            delegates.append(load_delegate('libnnapi.so'))
                            print("Using NNAPI delegate")
                        except:
                            pass
                    
                    if delegates:
                        self.interpreter = tflite.Interpreter(
                            model_path=self.model_path,
                            experimental_delegates=delegates
                        )
                    else:
                        print("Warning: No NPU delegate found, falling back to CPU")
                        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                except ImportError:
                    print("Warning: tflite_runtime not available, using standard TFLite")
                    self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            else:
                # Standard TFLite for CPU/GPU
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("TFLite model loaded successfully")
    
    def _predict_single(self, features):
        """
        Predict a single sample and measure inference time
        
        Returns:
            prediction: Predicted class index
            inference_time: Time taken for inference in milliseconds
        """
        features = features.reshape(1, -1).astype(np.float32)
        
        if self.model_type == 'keras':
            start_time = time.perf_counter()
            pred = self.model.predict(features, verbose=0)
            end_time = time.perf_counter()
            prediction = int(np.argmax(pred))
        else:  # tflite
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            start_time = time.perf_counter()
            self.interpreter.invoke()
            end_time = time.perf_counter()
            pred = self.interpreter.get_tensor(self.output_details[0]['index'])
            prediction = int(np.argmax(pred))
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return prediction, inference_time
    
    def run_benchmark(self, warmup_runs=10):
        """
        Run comprehensive benchmark on the entire dataset
        
        Args:
            warmup_runs: Number of warmup predictions before actual benchmarking
        """
        print(f"\nStarting benchmark with {warmup_runs} warmup runs...")
        
        # Warmup
        print("Running warmup predictions...")
        for i in range(min(warmup_runs, len(self.features))):
            self._predict_single(self.features[i])
        
        print("\nRunning benchmark on full dataset...")
        total_samples = len(self.features)
        
        for idx, (features, gesture_label) in enumerate(zip(self.features, self.gestures)):
            # Predict
            prediction, inference_time = self._predict_single(features)
            predicted_gesture = self.inv_label_map[prediction]
            
            # Store results
            self.results['per_frame_times'].append(inference_time)
            self.results['predictions'].append(predicted_gesture)
            self.results['ground_truth'].append(gesture_label)
            self.results['correct_predictions'].append(predicted_gesture == gesture_label)
            
            # Store per-gesture times
            if gesture_label not in self.results['per_gesture_times']:
                self.results['per_gesture_times'][gesture_label] = []
            self.results['per_gesture_times'][gesture_label].append(inference_time)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_samples} samples...")
        
        print(f"\nBenchmark complete! Processed {total_samples} samples.")
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate comprehensive statistics"""
        print("\nCalculating statistics...")
        
        # Overall statistics
        all_times = np.array(self.results['per_frame_times'])
        self.results['overall_stats'] = {
            'mean': float(np.mean(all_times)),
            'median': float(np.median(all_times)),
            'std': float(np.std(all_times)),
            'min': float(np.min(all_times)),
            'max': float(np.max(all_times)),
            'p95': float(np.percentile(all_times, 95)),
            'p99': float(np.percentile(all_times, 99)),
            'total_samples': len(all_times)
        }
        
        # Per-gesture statistics
        for gesture, times in self.results['per_gesture_times'].items():
            times_array = np.array(times)
            self.results['per_gesture_stats'][gesture] = {
                'mean': float(np.mean(times_array)),
                'median': float(np.median(times_array)),
                'std': float(np.std(times_array)),
                'min': float(np.min(times_array)),
                'max': float(np.max(times_array)),
                'p95': float(np.percentile(times_array, 95)),
                'p99': float(np.percentile(times_array, 99)),
                'count': len(times_array)
            }
        
        # Accuracy
        self.results['accuracy'] = {
            'overall': float(np.mean(self.results['correct_predictions'])),
            'total_correct': int(np.sum(self.results['correct_predictions'])),
            'total_samples': len(self.results['correct_predictions'])
        }
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        print("\nGenerating report...")
        
        device_suffix = f"_{self.device}" if self.device != 'cpu' else ""
        report_path = self.output_dir / f"inference_benchmark_report{device_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Inference Time Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model:** {self.model_path}\n")
            f.write(f"**Model Type:** {self.model_type.upper()}\n")
            f.write(f"**Device:** {self.device.upper()}\n")
            f.write(f"**Dataset:** {self.dataset_path}\n")
            f.write(f"**Total Samples:** {self.results['overall_stats']['total_samples']}\n\n")
            
            # Device Information
            f.write("## System Information\n\n")
            f.write(f"- **Platform:** {self.device_info.get('platform', 'Unknown')}\n")
            f.write(f"- **Processor:** {self.device_info.get('processor', 'Unknown')}\n")
            f.write(f"- **TensorFlow Version:** {self.device_info.get('tensorflow_version', 'Unknown')}\n")
            if 'gpu_count' in self.device_info:
                f.write(f"- **GPU Count:** {self.device_info['gpu_count']}\n")
            f.write("\n")
            
            # Overall Statistics
            f.write("## Overall Inference Time Statistics\n\n")
            f.write("| Metric | Value (ms) |\n")
            f.write("|--------|------------|\n")
            for metric, value in self.results['overall_stats'].items():
                if metric != 'total_samples':
                    f.write(f"| {metric.upper()} | {value:.4f} |\n")
            
            # FPS calculation
            mean_time = self.results['overall_stats']['mean']
            fps = 1000.0 / mean_time if mean_time > 0 else 0
            f.write(f"\n**Estimated FPS:** {fps:.2f} frames/second\n\n")
            
            # Accuracy
            f.write("## Model Accuracy\n\n")
            f.write(f"- **Overall Accuracy:** {self.results['accuracy']['overall']*100:.2f}%\n")
            f.write(f"- **Correct Predictions:** {self.results['accuracy']['total_correct']}/{self.results['accuracy']['total_samples']}\n\n")
            
            # Per-Gesture Statistics
            f.write("## Per-Gesture Inference Time Statistics\n\n")
            f.write("| Gesture | Count | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | P95 (ms) | P99 (ms) |\n")
            f.write("|---------|-------|-----------|-------------|----------|----------|----------|----------|----------|\n")
            
            for gesture in sorted(self.results['per_gesture_stats'].keys()):
                stats = self.results['per_gesture_stats'][gesture]
                f.write(f"| {gesture} | {stats['count']} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                       f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | "
                       f"{stats['p95']:.4f} | {stats['p99']:.4f} |\n")
            
            # Summary
            f.write("\n## Summary\n\n")
            f.write(f"- The model processes each frame in an average of **{mean_time:.4f} ms**\n")
            f.write(f"- This allows for real-time processing at approximately **{fps:.2f} FPS**\n")
            f.write(f"- 95% of predictions complete within **{self.results['overall_stats']['p95']:.4f} ms**\n")
            f.write(f"- 99% of predictions complete within **{self.results['overall_stats']['p99']:.4f} ms**\n")
            
        print(f"Report saved to: {report_path}")
        return report_path
    
    def generate_plots(self):
        """Generate comprehensive visualization plots"""
        print("\nGenerating plots...")
        
        # 1. Distribution of inference times
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1a. Histogram
        axes[0, 0].hist(self.results['per_frame_times'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of Inference Times', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Inference Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.results['overall_stats']['mean'], color='r', 
                          linestyle='--', label=f"Mean: {self.results['overall_stats']['mean']:.4f} ms")
        axes[0, 0].axvline(self.results['overall_stats']['median'], color='g', 
                          linestyle='--', label=f"Median: {self.results['overall_stats']['median']:.4f} ms")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 1b. Box plot
        axes[0, 1].boxplot(self.results['per_frame_times'], vert=True)
        axes[0, 1].set_title('Inference Time Box Plot', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Inference Time (ms)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 1c. Time series
        axes[1, 0].plot(self.results['per_frame_times'], alpha=0.6, linewidth=0.5)
        axes[1, 0].set_title('Inference Time Over Samples', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Inference Time (ms)')
        axes[1, 0].axhline(self.results['overall_stats']['mean'], color='r', 
                          linestyle='--', label='Mean', alpha=0.7)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 1d. CDF
        sorted_times = np.sort(self.results['per_frame_times'])
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        axes[1, 1].plot(sorted_times, cdf * 100, linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Inference Time (ms)')
        axes[1, 1].set_ylabel('Percentile (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(95, color='r', linestyle='--', alpha=0.5, label='95th percentile')
        axes[1, 1].axvline(self.results['overall_stats']['p95'], color='r', linestyle='--', alpha=0.5)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plot1_path = self.output_dir / 'inference_time_distribution.png'
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot1_path}")
        plt.close()
        
        # 2. Per-Gesture Analysis
        gestures = sorted(self.results['per_gesture_stats'].keys())
        means = [self.results['per_gesture_stats'][g]['mean'] for g in gestures]
        medians = [self.results['per_gesture_stats'][g]['median'] for g in gestures]
        stds = [self.results['per_gesture_stats'][g]['std'] for g in gestures]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 2a. Bar plot with error bars
        x = np.arange(len(gestures))
        width = 0.35
        axes[0].bar(x - width/2, means, width, label='Mean', alpha=0.8, yerr=stds, capsize=5)
        axes[0].bar(x + width/2, medians, width, label='Median', alpha=0.8)
        axes[0].set_xlabel('Gesture')
        axes[0].set_ylabel('Inference Time (ms)')
        axes[0].set_title('Mean and Median Inference Time per Gesture', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(gestures, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2b. Violin plot
        data_for_violin = [self.results['per_gesture_times'][g] for g in gestures]
        parts = axes[1].violinplot(data_for_violin, positions=range(len(gestures)), 
                                   showmeans=True, showmedians=True)
        axes[1].set_xlabel('Gesture')
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('Inference Time Distribution per Gesture', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(gestures)))
        axes[1].set_xticklabels(gestures, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot2_path = self.output_dir / 'per_gesture_inference_times.png'
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot2_path}")
        plt.close()
        
        # 3. Heatmap of inference time by gesture
        gesture_times_matrix = []
        max_samples = max(len(times) for times in self.results['per_gesture_times'].values())
        
        for gesture in gestures:
            times = self.results['per_gesture_times'][gesture]
            # Pad with NaN to make equal length
            padded = times + [np.nan] * (max_samples - len(times))
            gesture_times_matrix.append(padded[:100])  # Limit to first 100 for visibility
        
        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(gesture_times_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks(range(len(gestures)))
        ax.set_yticklabels(gestures)
        ax.set_xlabel('Sample Index (first 100)')
        ax.set_ylabel('Gesture')
        ax.set_title('Inference Time Heatmap by Gesture', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Inference Time (ms)', rotation=270, labelpad=20)
        plt.tight_layout()
        plot3_path = self.output_dir / 'inference_time_heatmap.png'
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot3_path}")
        plt.close()
        
        # 4. Statistics comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        metrics = ['mean', 'median', 'min', 'max', 'p95', 'p99']
        x = np.arange(len(gestures))
        width = 0.12
        
        for i, metric in enumerate(metrics):
            values = [self.results['per_gesture_stats'][g][metric] for g in gestures]
            ax.bar(x + i * width, values, width, label=metric.upper(), alpha=0.8)
        
        ax.set_xlabel('Gesture')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Time Statistics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(gestures, rotation=45, ha='right')
        ax.legend(loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plot4_path = self.output_dir / 'statistics_comparison.png'
        plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot4_path}")
        plt.close()
        
        print("All plots generated successfully!")
    
    def save_json_results(self):
        """Save detailed results to JSON"""
        device_suffix = f"_{self.device}" if self.device != 'cpu' else ""
        json_path = self.output_dir / f"benchmark_results{device_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data for JSON (convert numpy types)
        json_data = {
            'model_info': {
                'model_path': self.model_path,
                'model_type': self.model_type,
                'dataset_path': self.dataset_path,
                'device': self.device
            },
            'device_info': self.device_info,
            'overall_stats': self.results['overall_stats'],
            'per_gesture_stats': self.results['per_gesture_stats'],
            'accuracy': self.results['accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"JSON results saved to: {json_path}")
        return json_path
    
    def save_csv_results(self):
        """Save detailed per-sample results to CSV"""
        device_suffix = f"_{self.device}" if self.device != 'cpu' else ""
        csv_path = self.output_dir / f"detailed_results{device_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame({
            'sample_index': range(len(self.results['per_frame_times'])),
            'ground_truth': self.results['ground_truth'],
            'prediction': self.results['predictions'],
            'correct': self.results['correct_predictions'],
            'inference_time_ms': self.results['per_frame_times'],
            'device': self.device
        })
        
        df.to_csv(csv_path, index=False)
        print(f"Detailed CSV results saved to: {csv_path}")
        return csv_path


def run_single_device_benchmark(model_path, label_map_path, dataset_path, output_dir, device, db_manager=None):
    """Run benchmark on a single device"""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING ON: {device.upper()}")
    print(f"{'='*80}\n")
    
    # Configure device
    if not DeviceDetector.configure_device(device):
        print(f"Failed to configure {device.upper()}, skipping...")
        return None
    
    try:
        # Initialize benchmark
        benchmark = InferenceBenchmark(
            model_path=model_path,
            label_map_path=label_map_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            device=device
        )
        
        # Run benchmark
        benchmark.run_benchmark(warmup_runs=10)
        
        # Generate outputs
        print(f"\n{'='*80}")
        print(f"GENERATING OUTPUTS FOR {device.upper()}")
        print(f"{'='*80}")
        
        report_path = benchmark.generate_report()
        benchmark.generate_plots()
        json_path = benchmark.save_json_results()
        csv_path = benchmark.save_csv_results()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETE FOR {device.upper()}!")
        print(f"{'='*80}")
        stats = benchmark.results['overall_stats']
        print(f"Mean Inference Time: {stats['mean']:.4f} ms")
        print(f"Median Inference Time: {stats['median']:.4f} ms")
        print(f"Std Deviation: {stats['std']:.4f} ms")
        print(f"95th Percentile: {stats['p95']:.4f} ms")
        print(f"Estimated FPS: {1000.0/stats['mean']:.2f}")
        print(f"Accuracy: {benchmark.results['accuracy']['overall']*100:.2f}%")
        print(f"{'='*80}\n")
        
        # Save to persistent database
        if db_manager:
            now = datetime.now()
            db_entry = {
                'timestamp': now.isoformat(),
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'device_type': device.upper(),
                'cpu_model': benchmark.device_info.get('cpu_model', 'Unknown'),
                'gpu_model': benchmark.device_info.get('gpu_model', 'None'),
                'npu_model': benchmark.device_info.get('npu_model', 'None'),
                'ram_gb': benchmark.device_info.get('ram_total', 'Unknown'),
                'platform': benchmark.device_info.get('platform', 'Unknown'),
                'tensorflow_version': benchmark.device_info.get('tensorflow_version', 'Unknown'),
                'python_version': benchmark.device_info.get('python_version', 'Unknown'),
                'model_file': model_path,
                'model_type': benchmark.model_type,
                'dataset_samples': stats['total_samples'],
                'mean_inference_ms': stats['mean'],
                'median_inference_ms': stats['median'],
                'std_inference_ms': stats['std'],
                'min_inference_ms': stats['min'],
                'max_inference_ms': stats['max'],
                'p95_inference_ms': stats['p95'],
                'p99_inference_ms': stats['p99'],
                'fps': 1000.0 / stats['mean'],
                'accuracy_percent': benchmark.results['accuracy']['overall'] * 100,
                'total_correct': benchmark.results['accuracy']['total_correct'],
                'total_samples': benchmark.results['accuracy']['total_samples']
            }
            db_manager.add_benchmark_result(db_entry)
        
        return {
            'device': device,
            'stats': stats,
            'accuracy': benchmark.results['accuracy']['overall'],
            'device_info': benchmark.device_info
        }
        
    except Exception as e:
        print(f"\nERROR on {device.upper()}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_device_comparison_report(results, output_dir):
    """Generate comparison report across devices"""
    if len(results) < 2:
        return
    
    output_dir = Path(output_dir)
    report_path = output_dir / f"device_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    print(f"\n{'='*80}")
    print("GENERATING DEVICE COMPARISON REPORT")
    print(f"{'='*80}")
    
    with open(report_path, 'w') as f:
        f.write("# Device Performance Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Hardware Information Table
        f.write("## Hardware Configuration\n\n")
        f.write("| Component | Specification |\n")
        f.write("|-----------|---------------|\n")
        
        # Get system info from first result
        if results and 'device_info' in results[0]:
            info = results[0]['device_info']
            f.write(f"| **CPU Model** | {info.get('cpu_model', 'Unknown')} |\n")
            f.write(f"| **RAM** | {info.get('ram_total', 'Unknown')} |\n")
            
            # Check if any result has GPU
            gpu_result = next((r for r in results if r['device'] == 'gpu'), None)
            if gpu_result and 'device_info' in gpu_result:
                gpu_info = gpu_result['device_info']
                f.write(f"| **GPU Model** | {gpu_info.get('gpu_model', 'None')} |\n")
            
            # Check if any result has NPU
            npu_result = next((r for r in results if r['device'] == 'npu'), None)
            if npu_result and 'device_info' in npu_result:
                npu_info = npu_result['device_info']
                f.write(f"| **NPU Model** | {npu_info.get('npu_model', 'None')} |\n")
            
            f.write(f"| **Platform** | {info.get('platform', 'Unknown')} |\n")
            f.write(f"| **TensorFlow** | {info.get('tensorflow_version', 'Unknown')} |\n")
            f.write(f"| **Python** | {info.get('python_version', 'Unknown')} |\n")
        
        f.write("\n")
        
        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Device | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | P95 (ms) | P99 (ms) | FPS | Accuracy |\n")
        f.write("|--------|-----------|-------------|----------|----------|----------|----------|----------|-----|----------|\n")
        
        for result in results:
            stats = result['stats']
            device = result['device'].upper()
            fps = 1000.0 / stats['mean']
            acc = result['accuracy'] * 100
            f.write(f"| {device} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                   f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | "
                   f"{stats['p95']:.4f} | {stats['p99']:.4f} | {fps:.2f} | {acc:.2f}% |\n")
        
        # Speedup analysis
        f.write("\n## Speedup Analysis\n\n")
        cpu_result = next((r for r in results if r['device'] == 'cpu'), None)
        if cpu_result:
            cpu_mean = cpu_result['stats']['mean']
            f.write("Speedup relative to CPU:\n\n")
            for result in results:
                if result['device'] != 'cpu':
                    speedup = cpu_mean / result['stats']['mean']
                    f.write(f"- **{result['device'].upper()}**: {speedup:.2f}x faster\n")
        
        f.write("\n## Recommendation\n\n")
        fastest = min(results, key=lambda x: x['stats']['mean'])
        f.write(f"**Best Performance:** {fastest['device'].upper()} "
               f"({fastest['stats']['mean']:.4f} ms mean inference time)\n")
    
    print(f"Device comparison report saved to: {report_path}")
    
    # Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    devices = [r['device'].upper() for r in results]
    means = [r['stats']['mean'] for r in results]
    medians = [r['stats']['median'] for r in results]
    stds = [r['stats']['std'] for r in results]
    fps_list = [1000.0/r['stats']['mean'] for r in results]
    
    # Mean comparison
    axes[0, 0].bar(devices, means, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(devices)])
    axes[0, 0].set_ylabel('Inference Time (ms)')
    axes[0, 0].set_title('Mean Inference Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(means):
        axes[0, 0].text(i, v + max(means)*0.02, f'{v:.4f}ms', ha='center', fontweight='bold')
    
    # FPS comparison
    axes[0, 1].bar(devices, fps_list, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(devices)])
    axes[0, 1].set_ylabel('Frames Per Second')
    axes[0, 1].set_title('FPS Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(fps_list):
        axes[0, 1].text(i, v + max(fps_list)*0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Speedup comparison (if CPU exists)
    if cpu_result:
        cpu_mean = cpu_result['stats']['mean']
        speedups = [cpu_mean / r['stats']['mean'] for r in results]
        colors = ['gray' if r['device'] == 'cpu' else '#2ca02c' for r in results]
        axes[1, 0].bar(devices, speedups, color=colors)
        axes[1, 0].set_ylabel('Speedup (x)')
        axes[1, 0].set_title('Speedup Relative to CPU', fontsize=14, fontweight='bold')
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='CPU baseline')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].legend()
        for i, v in enumerate(speedups):
            axes[1, 0].text(i, v + max(speedups)*0.02, f'{v:.2f}x', ha='center', fontweight='bold')
    
    # P95 comparison
    p95s = [r['stats']['p95'] for r in results]
    axes[1, 1].bar(devices, p95s, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(devices)])
    axes[1, 1].set_ylabel('Inference Time (ms)')
    axes[1, 1].set_title('95th Percentile Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(p95s):
        axes[1, 1].text(i, v + max(p95s)*0.02, f'{v:.4f}ms', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / 'device_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark gesture recognition model inference time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_inference.py --device cpu
  python benchmark_inference.py --device gpu
  python benchmark_inference.py --device all
  python benchmark_inference.py --model gesture_model.h5 --device cpu
        """
    )
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH),
                       help='Path to model file (.h5 or .tflite)')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'npu', 'all'], default='cpu',
                       help='Device to use for inference (default: cpu)')
    parser.add_argument('--label-map', default=str(DEFAULT_LABEL_MAP),
                       help='Path to label_map.npy')
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET_PATH),
                       help='Path to dataset CSV file')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GESTURE RECOGNITION MODEL - INFERENCE TIME BENCHMARK")
    print("=" * 80)

    model_path = Path(args.model).expanduser()
    label_map_path = Path(args.label_map).expanduser()
    dataset_path = Path(args.dataset).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    
    # Check if files exist
    for path_obj, name in [(model_path, "Model"), (label_map_path, "Label map"), (dataset_path, "Dataset")]:
        if not path_obj.exists():
            print(f"ERROR: {name} file not found: {path_obj}")
            print(f"Please ensure the file exists in the current directory.")
            return
    
    # Detect available devices
    print("\nDetecting available devices...")
    print("=" * 80)
    available_devices = DeviceDetector.detect_devices()
    print("=" * 80)
    
    # Determine which devices to benchmark
    if args.device == 'all':
        devices_to_test = [d for d, available in available_devices.items() if available]
        print(f"\nWill benchmark on all available devices: {', '.join(d.upper() for d in devices_to_test)}")
    else:
        if not available_devices.get(args.device, False):
            print(f"\nWARNING: {args.device.upper()} not available, falling back to CPU")
            devices_to_test = ['cpu']
        else:
            devices_to_test = [args.device]
    
    try:
        # Initialize persistent database
        db_manager = BenchmarkDatabase(DEFAULT_DATABASE_PATH)
        
        results = []
        for device in devices_to_test:
            result = run_single_device_benchmark(
                model_path=str(model_path),
                label_map_path=str(label_map_path),
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
                device=device,
                db_manager=db_manager
            )
            if result:
                results.append(result)
        
        # Generate comparison report if multiple devices
        if len(results) > 1:
            generate_device_comparison_report(results, output_dir)
        
        # Print final comprehensive summary table
        print_final_summary_table(results)
        
        # Show recent benchmark history from database
        print("\n" + "=" * 80)
        print("BENCHMARK HISTORY FROM DATABASE")
        print("=" * 80)
        db_manager.print_recent_results(n=10)
        
        print("\n" + "=" * 80)
        print("ALL BENCHMARKS COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}/")
        print(f"Complete data saved to: {DEFAULT_DATABASE_PATH}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def print_final_summary_table(results):
    """Print a comprehensive final summary table with hardware and performance data"""
    if not results:
        return
    
    print("\n" + "=" * 100)
    print(" " * 35 + "FINAL BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Hardware Configuration
    print("\n📊 HARDWARE CONFIGURATION")
    print("-" * 100)
    
    if results and 'device_info' in results[0]:
        info = results[0]['device_info']
        print(f"  CPU Model      : {info.get('cpu_model', 'Unknown')}")
        print(f"  System RAM     : {info.get('ram_total', 'Unknown')}")
        
        # GPU info
        gpu_result = next((r for r in results if r['device'] == 'gpu'), None)
        if gpu_result and 'device_info' in gpu_result:
            gpu_info = gpu_result['device_info']
            print(f"  GPU Model      : {gpu_info.get('gpu_model', 'N/A')}")
        else:
            print(f"  GPU Model      : Not Available")
        
        # NPU info
        npu_result = next((r for r in results if r['device'] == 'npu'), None)
        if npu_result and 'device_info' in npu_result:
            npu_info = npu_result['device_info']
            print(f"  NPU Model      : {npu_info.get('npu_model', 'N/A')}")
        else:
            print(f"  NPU Model      : Not Available")
        
        print(f"  Platform       : {info.get('platform', 'Unknown')}")
        print(f"  TensorFlow Ver : {info.get('tensorflow_version', 'Unknown')}")
        print(f"  Python Version : {info.get('python_version', 'Unknown')}")
    
    # Performance Summary Table
    print("\n⚡ PERFORMANCE SUMMARY")
    print("-" * 100)
    
    # Header
    header = f"{'Device':<10} | {'Model/Type':<25} | {'Mean (ms)':<12} | {'Median (ms)':<12} | {'P95 (ms)':<12} | {'FPS':<10} | {'Accuracy':<10}"
    print(header)
    print("-" * 100)
    
    for result in sorted(results, key=lambda x: x['stats']['mean']):
        device = result['device'].upper()
        stats = result['stats']
        fps = 1000.0 / stats['mean']
        acc = result['accuracy'] * 100
        
        # Get device model name
        if 'device_info' in result:
            if device == 'CPU':
                model_name = result['device_info'].get('cpu_model', 'Unknown')
                if len(model_name) > 25:
                    model_name = model_name[:22] + "..."
            elif device == 'GPU':
                model_name = result['device_info'].get('gpu_model', 'Unknown')
                if len(model_name) > 25:
                    model_name = model_name[:22] + "..."
            elif device == 'NPU':
                model_name = result['device_info'].get('npu_model', 'Unknown')
                if len(model_name) > 25:
                    model_name = model_name[:22] + "..."
            else:
                model_name = 'Unknown'
        else:
            model_name = 'N/A'
        
        row = f"{device:<10} | {model_name:<25} | {stats['mean']:>10.4f}   | {stats['median']:>10.4f}   | {stats['p95']:>10.4f}   | {fps:>8.2f}   | {acc:>8.2f}%"
        print(row)
    
    print("-" * 100)
    
    # Speedup Analysis
    cpu_result = next((r for r in results if r['device'] == 'cpu'), None)
    if cpu_result and len(results) > 1:
        print("\n🚀 SPEEDUP ANALYSIS (Relative to CPU)")
        print("-" * 100)
        cpu_mean = cpu_result['stats']['mean']
        
        for result in results:
            if result['device'] != 'cpu':
                speedup = cpu_mean / result['stats']['mean']
                device = result['device'].upper()
                print(f"  {device:<10} : {speedup:.2f}x faster than CPU")
    
    # Recommendation
    print("\n💡 RECOMMENDATION")
    print("-" * 100)
    fastest = min(results, key=lambda x: x['stats']['mean'])
    device_name = fastest['device'].upper()
    mean_time = fastest['stats']['mean']
    fps = 1000.0 / mean_time
    
    print(f"  Best Performance Device : {device_name}")
    print(f"  Mean Inference Time     : {mean_time:.4f} ms")
    print(f"  Estimated FPS           : {fps:.2f}")
    print(f"  Accuracy                : {fastest['accuracy']*100:.2f}%")
    
    if device_name == 'CPU':
        print("\n  ℹ️  CPU is the fastest available. Consider GPU/NPU for better performance.")
    elif device_name == 'GPU':
        print("\n  ✅ GPU provides excellent performance for real-time inference.")
    elif device_name == 'NPU':
        print("\n  ✅ NPU provides optimal performance for edge deployment.")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
