# Gesture Model Benchmark

## Accuracy Report
```
--- Loading and Analyzing Predictions ---

Processing CPU (fp32)...
  - Processing samples: .......................... Done.
  - Found and processed 654 prediction files.

Processing GPU (fp32)...
  - Processing samples: .......................... Done.
  - Found and processed 654 prediction files.

Processing GPU (fp16)...
  - Processing samples: .......................... Done.
  - Found and processed 654 prediction files.

Processing NPU (int8)...
  - Processing samples: .......................... Done.
  - Found and processed 654 prediction files.

Processing NPU (w8a16)...
  - Processing samples: .......................... Done.
  - Found and processed 654 prediction files.

--- Searching for Prediction Mismatches ---
✅ No mismatches found. All models produced identical predictions.

==================================================
               Overall Accuracy Report
==================================================
Test Case                 | Accuracy (%)
--------------------------------------------------
CPU (fp32)                | 99.54%
GPU (fp32)                | 99.54%
GPU (fp16)                | 99.54%
NPU (int8)                | 99.54%
NPU (w8a16)               | 99.54%
--------------------------------------------------
```

## CPU (fp32) Performance
```
Log File Created: Sun Oct  5 19:40:08 2025
Time Scale: 1e-06
Epoch Timestamp: 1759693208760434 Steady Clock Timestamp: 9471194018
Software library version: 2.37.0.250724175447_124859
------------------------------------------------------------

SNPE Create Statistics:
----------------------------------------
  Load: 494 us
  Deserialize: 240 us
  Create: 2668 us

  Init: 4009 us
  De-Init: 400 us

  Create Network(s): 2050 us
  RPC Init Time: 0 
  Snpe Accelerator Init Time: 0 
  Accelerator Init Time: 0 

SNPE Execute Statistics (Averaged):
----------------------------------------
  Total Inference Time: 69 us
  Forward Propagate Time: 67 us
  RPC Execute Time: 0 
  Snpe Accelerator Time: 0 
  Num times yield occurred: 0
  Accelerator Time: 0 

  Per-Subnet Execution Times: 
  ------------------------------
    Subnet 0 (CPU) : 61 us

  Per-Layer Execution Times: 
  ------------------------------
    0: _sequential_1/dense_1/MatMul : 23 us : CPU
    1: _sequential_1/dense_1/Relu : 1 us : CPU
    2: _sequential_1/dense_1_2/MatMul : 27 us : CPU
    3: _sequential_1/dense_1_2/Relu : 1 us : CPU
    4: _sequential_1/dense_2_1/MatMul : 1 us : CPU
    5: _sequential_1/dense_2_1/Softmax : 1 us : CPU

```

---

## GPU (fp32) Performance
```
Log File Created: Sun Oct  5 19:40:09 2025
Time Scale: 1e-06
Epoch Timestamp: 1759693209138249 Steady Clock Timestamp: 9471571833
Software library version: 2.37.0.250724175447_124859
------------------------------------------------------------

SNPE Create Statistics:
----------------------------------------
  Load: 1646 us
  Deserialize: 1026 us
  Create: 204616 us

  Init: 207921 us
  De-Init: 8736 us

  Create Network(s): 202401 us
  RPC Init Time: 0 
  Snpe Accelerator Init Time: 0 
  Accelerator Init Time: 0 

SNPE Execute Statistics (Averaged):
----------------------------------------
  Total Inference Time: 791 us
  Forward Propagate Time: 776 us
  RPC Execute Time: 0 
  Snpe Accelerator Time: 0 
  Num times yield occurred: 0
  Accelerator Time: 0 

  Per-Subnet Execution Times: 
  ------------------------------
    Subnet 0 (GPU) : 757 us

  Per-Layer Execution Times: 
  ------------------------------
    0: sequential_1/dense_1/MatMul : 8 us : GPU
    1: sequential_1/dense_1_2/MatMul : 4 us : GPU
    2: sequential_1/dense_2_1/MatMul : 4 us : GPU
    3: sequential_1/dense_2_1/Softmax : 8 us : GPU

```

---

## GPU (fp16) Performance
```
Log File Created: Sun Oct  5 19:40:10 2025
Time Scale: 1e-06
Epoch Timestamp: 1759693210258795 Steady Clock Timestamp: 9472692379
Software library version: 2.37.0.250724175447_124859
------------------------------------------------------------

SNPE Create Statistics:
----------------------------------------
  Load: 1651 us
  Deserialize: 964 us
  Create: 201191 us

  Init: 204416 us
  De-Init: 9051 us

  Create Network(s): 199047 us
  RPC Init Time: 0 
  Snpe Accelerator Init Time: 0 
  Accelerator Init Time: 0 

SNPE Execute Statistics (Averaged):
----------------------------------------
  Total Inference Time: 768 us
  Forward Propagate Time: 754 us
  RPC Execute Time: 0 
  Snpe Accelerator Time: 0 
  Num times yield occurred: 0
  Accelerator Time: 0 

  Per-Subnet Execution Times: 
  ------------------------------
    Subnet 0 (GPU) : 738 us

  Per-Layer Execution Times: 
  ------------------------------
    0: sequential_1/dense_1/MatMul : 8 us : GPU
    1: sequential_1/dense_1_2/MatMul : 4 us : GPU
    2: sequential_1/dense_2_1/MatMul : 4 us : GPU
    3: sequential_1/dense_2_1/Softmax : 8 us : GPU

```

---

## NPU (int8) Performance
```
Log File Created: Sun Oct  5 19:40:11 2025
Time Scale: 1e-06
Epoch Timestamp: 1759693211491502 Steady Clock Timestamp: 9473925087
Software library version: 2.37.0.250724175447_124859
------------------------------------------------------------

SNPE Create Statistics:
----------------------------------------
  Load: 1813 us
  Deserialize: 919 us
  Create: 154620 us

  Init: 158658 us
  De-Init: 6636 us

  Create Network(s): 152110 us
  RPC Init Time: 1158 us
  Snpe Accelerator Init Time: 1090 us
  Accelerator Init Time: 1009 us

SNPE Execute Statistics (Averaged):
----------------------------------------
  Total Inference Time: 1328 us
  Forward Propagate Time: 1315 us
  RPC Execute Time: 290 us
  Snpe Accelerator Time: 250 us
  Num times yield occurred: 0
  Accelerator Time: 59 us

  Per-Subnet Execution Times: 
  ------------------------------
    Subnet 0 (DSP) : 13545 cycles

  Per-Layer Execution Times: 
  ------------------------------
    0: Input OpId_2 (cycles) : 2142 cycles : DSP
    1: sequential_1/dense_1/MatMul:OpId_18 (cycles) : 0 cycles : DSP
    2: sequential_1/dense_1/Relu:OpId_20 (cycles) : 1252 cycles : DSP
    3: sequential_1/dense_1_2/MatMul:OpId_29 (cycles) : 0 cycles : DSP
    4: sequential_1/dense_1_2/Relu:OpId_31 (cycles) : 510 cycles : DSP
    5: sequential_1/dense_2_1/MatMul:OpId_40 (cycles) : 1447 cycles : DSP
    6: sequential_1/dense_2_1/Softmax:OpId_42 (cycles) : 2152 cycles : DSP
    7: Output OpId_3 (cycles) : 6038 cycles : DSP

```

---

## NPU (w8a16) Performance
```
Log File Created: Sun Oct  5 19:40:13 2025
Time Scale: 1e-06
Epoch Timestamp: 1759693213096193 Steady Clock Timestamp: 9475529777
Software library version: 2.37.0.250724175447_124859
------------------------------------------------------------

SNPE Create Statistics:
----------------------------------------
  Load: 1719 us
  Deserialize: 1040 us
  Create: 148341 us

  Init: 152530 us
  De-Init: 6959 us

  Create Network(s): 145848 us
  RPC Init Time: 1304 us
  Snpe Accelerator Init Time: 1232 us
  Accelerator Init Time: 1150 us

SNPE Execute Statistics (Averaged):
----------------------------------------
  Total Inference Time: 1354 us
  Forward Propagate Time: 1341 us
  RPC Execute Time: 287 us
  Snpe Accelerator Time: 248 us
  Num times yield occurred: 0
  Accelerator Time: 56 us

  Per-Subnet Execution Times: 
  ------------------------------
    Subnet 0 (DSP) : 12512 cycles

  Per-Layer Execution Times: 
  ------------------------------
    0: Input OpId_2 (cycles) : 2140 cycles : DSP
    1: sequential_1/dense_1/MatMul:OpId_18 (cycles) : 0 cycles : DSP
    2: sequential_1/dense_1/Relu:OpId_20 (cycles) : 1470 cycles : DSP
    3: sequential_1/dense_1_2/MatMul:OpId_29 (cycles) : 0 cycles : DSP
    4: sequential_1/dense_1_2/Relu:OpId_31 (cycles) : 496 cycles : DSP
    5: sequential_1/dense_2_1/MatMul:OpId_40 (cycles) : 1175 cycles : DSP
    6: sequential_1/dense_2_1/Softmax:OpId_42 (cycles) : 2721 cycles : DSP
    7: Output OpId_3 (cycles) : 4507 cycles : DSP

```

---

