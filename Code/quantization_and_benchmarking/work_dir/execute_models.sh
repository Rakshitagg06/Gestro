#!/bin/sh

# ==============================================================================
# SNPE Inference Execution Script for Gesture Recognition Model on Android
# ==============================================================================
# This script runs inference for a gesture recognition model using different
# hardware backends (CPU, GPU, NPU) to generate output and performance logs.

# --- INFERENCE 1: CPU (float32) ---
echo "--- [1/5] Executing Inference: CPU (float32) ---"
time ./snpe-net-run \
  --container ./mymodel_float32.dlc \
  --input_list ./gestures_test_raw/input_list.txt \
  --output_dir ./output_cpu_fp32 \
  --perf_profile high_performance \
  --profiling_level detailed
echo "CPU (float32) execution complete."
echo ""
echo "----------------------------------------------------"
echo ""

# --- INFERENCE 2: GPU (float32) ---
echo "--- [2/5] Executing Inference: GPU (float32) ---"
time ./snpe-net-run \
  --container ./mymodel_float32.dlc \
  --input_list ./gestures_test_raw/input_list.txt \
  --output_dir ./output_gpu_fp32 \
  --use_gpu \
  --perf_profile high_performance \
  --profiling_level detailed
echo "GPU (float32) execution complete."
echo ""
echo "----------------------------------------------------"
echo ""

# --- INFERENCE 3: GPU (float16) ---
echo "--- [3/5] Executing Inference: GPU (float16) ---"
time ./snpe-net-run \
  --container ./mymodel_float32.dlc \
  --input_list ./gestures_test_raw/input_list.txt \
  --output_dir ./output_gpu_fp16 \
  --use_gpu \
  --gpu_mode float16 \
  --perf_profile high_performance \
  --profiling_level detailed
echo "GPU (float16) execution complete."
echo ""
echo "----------------------------------------------------"
echo ""

# --- INFERENCE 4: NPU (int8) ---
echo "--- [4/5] Executing Inference: NPU (int8) ---"
time ./snpe-net-run \
  --container ./mymodel_quantized_int8.dlc \
  --input_list ./gestures_test_raw/input_list.txt \
  --output_dir ./output_npu_int8 \
  --use_dsp \
  --perf_profile high_performance \
  --profiling_level detailed
echo "NPU (int8) execution complete."
echo ""
echo "----------------------------------------------------"
echo ""

# --- INFERENCE 5: NPU (w8a16) ---
echo "--- [5/5] Executing Inference: NPU (w8a16) ---"
time ./snpe-net-run \
  --container ./mymodel_quantized_w8a16.dlc \
  --input_list ./gestures_test_raw/input_list.txt \
  --output_dir ./output_npu_w8a16 \
  --use_dsp \
  --perf_profile high_performance \
  --profiling_level detailed
echo "NPU (w8a16) execution complete."
echo ""
echo "----------------------------------------------------"
echo ""


echo "--- ALL INFERENCE EXECUTIONS COMPLETE ---"
echo "You can now pull the 'output_*' directories to your computer for analysis."
echo "Check 'SNPEDiag.log' in each directory for latency and the 'time' command output for total execution time."
