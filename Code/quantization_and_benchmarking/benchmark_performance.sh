#!/bin/sh
# This script runs snpe-diagview on all benchmark logs and appends the
# human-readable summary output into a single Markdown report file.

REPORT_FILE="report.md"

echo "--- Generating Performance Report ---"
# Initialize the report file with a title
echo "# Gesture Model Performance Benchmark" > $REPORT_FILE
echo "" >> $REPORT_FILE

# --- Define Benchmark Cases ---
# This associates a clean name with each directory
define_benchmarks() {
    # The format is "DirectoryName,ReportTitle"
    BENCHMARKS="
work_dir/output_cpu_fp32,CPU (fp32)
work_dir/output_gpu_fp32,GPU (fp32)
work_dir/output_gpu_fp16,GPU (fp16)
work_dir/output_npu_int8,NPU (int8)
work_dir/output_npu_w8a16,NPU (w8a16)
"
}

define_benchmarks

# --- Process Each Log ---
# Use IFS to handle the loop correctly
OLD_IFS=$IFS
IFS='
'
for bench in $BENCHMARKS; do
    # Split the line into directory and title
    dir=$(echo "$bench" | cut -d',' -f1)
    name=$(echo "$bench" | cut -d',' -f2)
    
    INPUT_LOG="$dir/SNPEDiag_0.log"

    if [ -f "$INPUT_LOG" ]; then
        echo "Processing log for $name..."
        
        # Append the title for this section to the report
        echo "## $name Performance" >> $REPORT_FILE
        echo '```' >> $REPORT_FILE # Start a code block for clean formatting
        
        # Run snpe-diagview and append its stdout to the report
        snpe-diagview --input_log "$INPUT_LOG" >> $REPORT_FILE
        
        # End the code block and add a separator
        echo '```' >> $REPORT_FILE
        echo "" >> $REPORT_FILE # Add a newline for spacing
        echo "---" >> $REPORT_FILE
        echo "" >> $REPORT_FILE

    else
        echo "Warning: Log file not found for $dir. Skipping."
    fi
done
IFS=$OLD_IFS

echo "--- Performance Report Complete ---"
echo "✅ Report saved to: $REPORT_FILE"