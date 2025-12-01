import numpy as np
import pandas as pd
import os
import sys

def get_predictions_from_dir(output_dir, num_samples):
    """
    Dynamically finds all .raw prediction files in an output directory,
    loads them, and returns a list of the predicted class indices,
    while printing a progress bar.

    Args:
        output_dir (str): The path to the output directory (e.g., 'output_cpu_fp32').
        num_samples (int): The total number of test samples to look for.

    Returns:
        list: A list of predicted integer labels. Returns an empty list on error.
    """
    if not os.path.isdir(output_dir):
        print(f"\nWarning: Directory not found: {output_dir}", file=sys.stderr)
        return []

    predictions = []
    print("  - Processing samples: ", end="")
    for i in range(num_samples):
        # --- Progress Bar Logic ---
        # Print a progress update every 25 samples
        if i > 0 and i % 25 == 0:
            print(".", end="", flush=True)

        result_dir = os.path.join(output_dir, f'Result_{i}')
        found_raw_file = None
        
        if os.path.isdir(result_dir):
            for root, _, files in os.walk(result_dir):
                for file in files:
                    if file.endswith('.raw'):
                        found_raw_file = os.path.join(root, file)
                        break
                if found_raw_file:
                    break
        
        if not found_raw_file:
            predictions.append(-1) 
            continue

        pred_vector = np.fromfile(found_raw_file, dtype=np.float32)
        if pred_vector.size == 0:
            predictions.append(-1)
            continue
            
        predicted_index = np.argmax(pred_vector)
        predictions.append(predicted_index)
    
    print(" Done.") # End of progress bar
    return predictions

if __name__ == "__main__":
    # --- Configuration ---
    BASE_DIR = 'work_dir'
    TEST_CSV_PATH = 'gestures_test.csv'
    LABELS_NPY_PATH = 'mymodel_labels.npy'

    BENCHMARKS = [
        {"name": "CPU (fp32)", "dir": "output_cpu_fp32"},
        {"name": "GPU (fp32)", "dir": "output_gpu_fp32"},
        {"name": "GPU (fp16)", "dir": "output_gpu_fp16"},
        {"name": "NPU (int8)", "dir": "output_npu_int8"},
        {"name": "NPU (w8a16)", "dir": "output_npu_w8a16"},
    ]
    # -------------------

    # --- Load Ground Truth and Labels ---
    try:
        label_map = np.load(LABELS_NPY_PATH, allow_pickle=True).item()
        inv_label_map = {v: k for k, v in label_map.items()}
        ground_truth_df = pd.read_csv(TEST_CSV_PATH, header=None)
        ground_truth_gestures = ground_truth_df.iloc[:, 0].tolist()
        ground_truth_labels = [label_map[g] for g in ground_truth_gestures]
        num_samples = len(ground_truth_labels)
    except Exception as e:
        print(f"Error: Failed to load ground truth data. {e}", file=sys.stderr)
        sys.exit(1)

    # --- Collect All Predictions ---
    results_df = pd.DataFrame({
        'Sample_ID': range(num_samples),
        'Ground_Truth': ground_truth_gestures
    })

    print("\n--- Loading and Analyzing Predictions ---")
    for bench in BENCHMARKS:
        print(f"\nProcessing {bench['name']}...")
        full_dir_path = os.path.join(BASE_DIR, bench['dir'])
        predictions = get_predictions_from_dir(full_dir_path, num_samples)
        
        # --- Explicit File Count ---
        if predictions:
            print(f"  - Found and processed {len(predictions)} prediction files.")
            results_df[bench['name']] = [inv_label_map.get(p, 'ERROR') for p in predictions]
        else:
            print("  - No prediction files were found or processed.")

    # --- Identify Mismatches ---
    print("\n--- Searching for Prediction Mismatches ---")
    benchmark_cols = [b['name'] for b in BENCHMARKS if b['name'] in results_df.columns]
    
    if not benchmark_cols:
        print("No benchmark data to compare.")
    else:
        results_df['Unique_Predictions'] = results_df[benchmark_cols].nunique(axis=1)
        mismatches = results_df[results_df['Unique_Predictions'] > 1]

        if mismatches.empty:
            print("✅ No mismatches found. All models produced identical predictions.")
        else:
            print(f"🔥 Found {len(mismatches)} samples with differing predictions:")
            for index, row in mismatches.iterrows():
                print(f"\n- Sample #{row['Sample_ID']} (Ground Truth: '{row['Ground_Truth']}')")
                for col in benchmark_cols:
                    print(f"  - {col:<15}: {row[col]}")

    # --- Final Accuracy Report ---
    print("\n" + "="*50)
    print(" " * 15 + "Overall Accuracy Report")
    print("="*50)
    print(f"{'Test Case':<25} | {'Accuracy (%)'}")
    print("-" * 50)

    for bench in BENCHMARKS:
        if bench['name'] in results_df.columns:
            correct_count = (results_df['Ground_Truth'] == results_df[bench['name']]).sum()
            accuracy = (correct_count / num_samples) * 100
            print(f"{bench['name']:<25} | {accuracy:.2f}%")
        else:
            print(f"{bench['name']:<25} | FAILED TO LOAD")
        
    print("-" * 50)