# Edge_AI_Perf

Alpha version.

*Tested for the specified use case: (1) Using Colab T4, (2) Calculate CO2-EQ for Thailand from Jan 2026.*

## Objective & Problem:

* **Goal:** A library to make simple/ quick benchmark estimates of DL/LLM machine learning models for their deployment viability onto edge computing devices. 

* **Problem:** Estimating the feasibility and viability for (edge) hardware deployments is hard. Benchmarking data is patchy (in product specs, datasheets and benchmark tools). This library serves to give simple (*approximate*) estimates for a broad range of device deployment, measured by a ratio comparison from a baseline performance, without long benchmarking trials (differing from MLPerf / MLCommons).



## Installation:

(Use `!pip` for notebook installations, Jupyter, Colab, etc.)
```bash
pip install git+https://github.com/pmdscully/Edge_AI_Perf.git --upgrade
```



## Basic Usage Example:

##### First, specify your model's characteristics: *(see below for calculation example)*

* Measurement of **average time in milliseconds** for an inference (1-unit) via a warm-start model inference (forward pass) trial (e.g. mean time to infer 100 inputs).
* in-memory byte size of the model *(e.g. quantized, compressed or original)*,
* datatype of the parameters as used in the inference time measurement. 
  * Used to estimate inference time as the another category type: TOPS or TFLOPS.
  * i.e. quantized / unquantized, int32, fp16 (bf16), int4, int8, etc. 
  * Currently only supports ['int8','fp8'] (which are int8, and floats of any unfixed size). *Improve the accuracy and completeness of the hardware data source table, to improve this estimate.*

```python
# ---- Data from model (e.g. ExecuTorch/ PyTorch):
baseline_time_ms = 1   # Avg. Time for an Inference (warm-start forward pass).
pte_size_bytes = 9e5
weight_dtype = 'int8'  # Parameter dtype
```

##### Specify the baseline hardware from the device table:

* Specify the hardware used in your current environment for the baseline.
  * Currently tested on **`Colab T4`**, which is row index 8 in the table.

```python
from lib_edge_eval import fetch_edge_hardware_dataframe

# ======= Load the Hardware Table: ========
df_final = fetch_edge_hardware_dataframe()
display(df_final)

# ======= Specify the Baseline from the Hardware Table: ========
baseline_index = 8 # 8 = NVIDIA T4 GPU
```
##### Calculate Deployment on Range of Hardware:

Here we use the `Edge_AI_Perf` python library to estimate performance over a range of tiny to large hardware:
- inference time comparisons.
- in-memory feasibility (Out of memory).
- Energy consumption (KwH) efficiency.
- CO2-EQ carbon footprint efficiency.

```python
from lib_edge_eval import calculate_edge_metrics, Plotting

# ======= Calculate from Baseline: ========
df_comparison = calculate_edge_metrics(
    df=df_final,
    baseline_idx=baseline_index,
    baseline_ms=baseline_time_ms,
    baseline_mem=pte_size_bytes,
    baseline_dtype=weight_dtype if weight_dtype ['int8','fp8'] else 'fp8'
)

display(df_comparison.round(2))
Plotting.plot_comparison_metrics(df_comparison)
```

- Output the Table of Results and Plot:
  - Notice the `Int8/FP8 Params Fit (Est.)` column specifies a tick if the model's parameters (with input data / context) will fit in RAM of the device.


|      | Device Name                 | Baseline | Int8/FP8 Params Fit (Est.) | Hours per 1M INT8 | Hours per 1M FP16 | KWh per 1M INT8 | KWh per 1M FP16 | CO2-EQ per 1M INT8 | CO2-EQ per 1M FP16 |
| ---- | --------------------------- | -------- | -------------------------- | ----------------- | ----------------- | --------------- | --------------- | ------------------ | ------------------ |
| 0    | Syntiant NDP200             |          | 🗸                          | 712.20            | NaN               | 0.00            | NaN             | 0.00               | NaN                |
| 1    | Ambiq Apollo 510            |          | 🗸                          | 569.76            | 569.76            | 0.01            | 0.01            | 0.01               | 0.01               |
| 2    | ESP32-S3-WROOM-1            |          | 🗸                          | 593.50            | 9116.16           | 0.30            | 4.56            | 0.17               | 2.54               |
| 3    | Raspberry Pi 4              |          | 🗸                          | 294.07            | 56.98             | 4.41            | 0.85            | 2.45               | 0.48               |
| 4    | Raspberry Pi 5 8GB          |          | 🗸                          | 71.22             | 18.99             | 1.78            | 0.47            | 0.99               | 0.26               |
| 5    | Snapdragon 7+ Gen 3         |          | 🗸                          | 0.15              | 1.56              | 0.00            | 0.01            | 0.00               | 0.00               |
| 6    | Apple A19 Pro               |          | 🗸                          | 0.13              | 0.92              | 0.00            | 0.01            | 0.00               | 0.01               |
| 7    | NVIDIA Jetson AGX Orin 64GB |          | 🗸                          | 0.02              | 0.86              | 0.00            | 0.05            | 0.00               | 0.03               |
| 8    | NVIDIA Tesla T4             | 🗸        | 🗸                          | 0.04              | 0.07              | 0.00            | 0.01            | 0.00               | 0.00               |
| 9    | NVIDIA RTX 4090             |          | 🗸                          | 0.00              | 0.02              | 0.00            | 0.02            | 0.00               | 0.01               |
| 10   | NVIDIA DGX B200             |          | 🗸                          | 0.00              | 0.00              | 0.00            | 0.00            | 0.00               | 0.00               |

* The output plots illustrate:
  1. `Inference Time Gains/losses` by dtypes choices and hardware,  i.e. quantizing into `int8` or floating points. 
     *Note, missing bars indicate no support for that dtype, (i.e. Syntiant NDP200 has no data on FP performance).*
  2. `CO2-eq Efficiencies`, according to an IPCC Tier 1 emissions factor for CO2-eq (KG) per KWH of energy consumption.

![Barplot_Latency_CO2](README.assets/Barplot_Latency_CO2.png)



## Methodology:

The methodology is simple:

1. Measure the performance characteristics on known hardware (currently Google Colab's T4 GPU).
2. Quantify their performance values for other hardware profiles, according to their ratio differences from T4.

This approach makes it simple and viable to update a known table of hardware, and benchmark many devices simultaneously.

It differs from ML Commons' ML Perf which perform benchmarks through tests to give precise results for limited hardware. Whereas this methodology gives **quick estimated results for all hardware specs extracted from a data source** (below). 

### **Key Performance Measurements:**

The output estimated measures are **per 1 million inference-time inputs** for both ***`INT8`*** and ***`FP16`*** data types, for the following metrics:

* Inference time, 
* Watt Hours, 
* CO2EQ
* Model fitment viability (i.e. inputs, model param and OS) into operational runtime memory.

### **Data Sources:**

Markdown table data is loaded (ETL) into a `pandas Dataframe` for processing.

* `edge_hardware_table.md`
  * A fixed set of devices from a hardware table, which specifies their (verified) hardware profile characteristics.
  * The original table provides a range of 10 devices covering different form-factors, weights, compute capacities, prices, release dates and sizes
    * 2x Embedded:
    * 3x Embedded Dev Boards
    * 2x Smartphones
    * 2x PCIe Boards for desktops
    * 1x GPU Server Unit.
* `EmissionsFactor_Calculation.md` 
  * A known CO2-EQ emissions factor calculation (Scope 2 + Scope 3) from kilowatt hours (KWH), collected from national greenhouse gas organization source(s).
  * Currently implemented for: `Thailand` for runtimes from  `Jan 2026` to  `March 2026`.





---

## Example Code to Calculate Model Baseline Performance data:

Here we show how to calculate three PyTorch model metrics, as required inputs for the library functions.

1. `weight_dtype` : - data type of model parameters
2. `pte_size_bytes` : - size of the model parameters in RAM.
3. `time_ms_inf` : - (average) time in milliseconds for a single model inference (forward pass).

Below are model-specific performance characteristics to help evaluate the model; in addition to its performance across other hardware.

* We start with a loaded PyTorch model:


```python
import torchvision.models as models
import torch
# model = models.alexnet(); input_shape=(batch_size, 3, 224, 224) # For example.
model = ... # PyTorch model
```

#### 1. Calculate - FLOPS, MACs and Param (byte) Size :

* Example using [FLOPS Profiler](https://www.deepspeed.ai/tutorials/flops-profiler/#example-alexnet) *[DeepSpeed]*

```bash
!uv pip install deepspeed
```


```python
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

with get_accelerator().device(0):
    batch_size = 1
    flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(batch_size, 10, 5), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=False, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
flops, macs, params
```

Output example for a small Temporal-Convolutional Network (TCN) architecture, for sequence input classification:

> ('1.95 M', '950.75 KMACs', '66.11 K')

#### 2. In-Memory Model Size: 

##### 2.1 PyTorch - Export Model to (uncompressed) native PyTorch file: (PTH)

```python
import os
import torch

def export_pytorch_and_get_size(model, filename="model_pytorch.pth"):
    """Exports the PyTorch model to a .pth file and returns its size in bytes."""
    # Save the model state dict
    torch.save(model.state_dict(), filename)

    # Calculate size
    size_bytes = os.path.getsize(filename)
    return size_bytes

# Run the function
pth_filename = "model_pytorch.pth"
pth_size_bytes = export_pytorch_and_get_size(model, pth_filename)

print(f"PyTorch Model (.pth) size: {pth_size_bytes} bytes ({pth_size_bytes/1024:.2f} KB)")
```

Output example:

> PyTorch Model (.pth) size: 283159 bytes (276.52 KB)

##### 2.2 Parameter dtypes:

```python
# Get weight dtype
weight_dtype = str(next(model.parameters()).dtype).split('.')[-1]   
```

##### 2.3 ExecuTorch - Export Model to ExecuTorch file (PTE):

- See ExecuTorch library [example](https://www.google.com/url?q=https%3A%2F%2Fpytorch.org%2Fexecutorch%2Fstable%2Fgetting-started-setup.html)

```bash
!uv pip install executorch 
```

- Export model to `pte` file.

```python
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.export import export
import torch
import os

def export_to_executorch(model, example_inputs):
    """Exports a PyTorch model to an ExecuTorch program using XNNPACK."""
    model = model.eval()    
    # Get weight dtype
    weight_dtype = str(next(model.parameters()).dtype).split('.')[-1]   
    # Export to Program
    exported_program = export(model, example_inputs)
    # Lower to Edge and convert to ExecuTorch
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    
    return executorch_program, weight_dtype

# Define inputs
# The SimpleTCN model expects (batch_size, sequence_length, num_features)
example_inputs = (torch.randn(1, 10, 5),)

# Call the function
executorch_program, weight_dtype = export_to_executorch(model, example_inputs)
```

- Calculate size of `pte` model file, required for loading weights into RAM.

```python
# The .pte file size represents the binary size of the program including weights
pte_size_bytes = os.path.getsize(pte_path)

print(f"ExecuTorch Program (.pte) size: {pte_size_bytes} bytes ({pte_size_bytes/1024:.2f} KB)")
print(f"PyTorch Model (.pth) size: {pth_size_bytes} bytes ({pth_size_bytes/1024:.2f} KB)")
reduction = (1 - (pte_size_bytes / pth_size_bytes)) * 100
print(f"\nSize Difference: {reduction:.2f}% change in footprint")

print(f"\nInference weights size (approx): {pte_size_bytes} bytes")
print(f"Weight datatype: {weight_dtype}")

```

Output example:

> ExecuTorch Program (.pte) size: 276364 bytes (269.89 KB)
>
> PyTorch Model (.pth) size: 283159 bytes (276.52 KB)
>
> 
>
> Size Difference: 2.40% change in footprint
>
> Inference weights size (approx): 276364 bytes
>
> Weight datatype: float32

#### 3. Inference Time in MS:

```python
import time, torch

def benchmark_executorch_ms(method, input_data, trials=20):
    # Warm-up
    method.execute([input_data])
    start = time.perf_counter()
    for _ in range(trials): method.execute([input_data])
    avg_ms = ((time.perf_counter() - start) / trials) * 1000
    return avg_ms

def benchmark_pytorch_ms(model, input_data, trials=20):
    model.eval()
    # Warm-up
    with torch.no_grad():
        model(input_data)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(trials): model(input_data)
    avg_ms = ((time.perf_counter() - start) / trials) * 1000
    return avg_ms

# Run benchmarks
t = torch.randn(1, 10, 5)
num_trials = 150
time_ms_pytorch = benchmark_pytorch_ms(model, t, num_trials)
time_ms_inf = benchmark_executorch_ms(method, t, num_trials)

print(f"ExecuTorch Avg Inference Time ({num_trials} trials): {time_ms_inf:.3f} ms")
print(f"PyTorch Avg Inference Time ({num_trials} trials): {time_ms_pytorch:.3f} ms")
print(f"Speedup: {time_ms_pytorch / time_ms_inf:.2f}x")
```

Output example:

> ExecuTorch Avg Inference Time (150 trials): 0.074 ms
>
> PyTorch Avg Inference Time (150 trials): 1.572 ms
>
> Speedup: 21.30x

#### 4. Summary Model Deployment Performance Comparison Report:

```python
import pandas as pd
import sys

def generate_model_performance_report(model, executorch_program,
                                      pth_size_bytes, pte_size_bytes,
                                      time_ms_pytorch, time_ms_inf,
                                      weight_dtype, params, flops):
    """Generates and displays a comprehensive performance comparison report."""

    # 1. Calculate dynamic differences
    size_pct = ((pte_size_bytes - pth_size_bytes) / pth_size_bytes) * 100
    latency_pct = ((time_ms_inf - time_ms_pytorch) / time_ms_pytorch) * 100

    performance_data = {
        "Metric": ["Precision", "Parameters", "FLOPs", "Model Size (KB)", "Inference Time (ms)"],
        "PyTorch (Source)": [
            weight_dtype,
            params,
            flops,
            f"{pth_size_bytes / 1024:.2f}",
            f"{time_ms_pytorch:.3f}"
        ],
        "ExecuTorch (.pte)": [
            weight_dtype,
            params,
            flops,
            f"{pte_size_bytes / 1024:.2f}",
            f"{time_ms_inf:.3f}"
        ],
        "Diff": [
            "--",
            "--",
            "--",
            f"{size_pct:+.2f}%",
            f"{latency_pct:+.2f}%"
        ]
    }

    # Display Report
    report_df = pd.DataFrame(performance_data)
    print("=== Model Performance Report ===")
    display(report_df)

    # 2. Print File Size Comparison (Memory Footprint)
    print(f"\n--- In-Memory (RAM) Footprint Change ---")
    print(f"PyTorch Model: {pth_size_bytes / 1024:.2f} KB")
    print(f"ExecuTorch Program: {pte_size_bytes / 1024:.2f} KB")
    mem_diff = ((pte_size_bytes - pth_size_bytes) / pth_size_bytes) * 100
    print(f"Difference: {mem_diff:+.2f}%")

    # 3. Throughput Analysis
    numeric_flops = float(flops.split()[0]) * 1e6
    et_gflops = (numeric_flops / (time_ms_inf / 1000)) / 1e9
    pt_gflops = (numeric_flops / (time_ms_pytorch / 1000)) / 1e9

    print(f"\n--- Throughput Change (on baseline hardware) ---")
    print(f"PyTorch: {pt_gflops:.4f} GFLOPS")
    print(f"ExecuTorch: {et_gflops:.4f} GFLOPS")
    print(f"Speedup Factor: {time_ms_pytorch / time_ms_inf:.2f}x faster")

# Execute the function
generate_model_performance_report(
    model,
    executorch_program,
    pth_size_bytes,
    pte_size_bytes,
    time_ms_pytorch,
    time_ms_inf,
    weight_dtype,
    params,
    flops
)
```

Output example:

> \=\=\= Model Performance Report \=\=\=
>
> |      | Metric              | PyTorch (Source) | ExecuTorch (.pte) | Diff    |
> | ---- | ------------------- | ---------------- | ----------------- | ------- |
> | 0    | Precision           | float32          | float32           | --      |
> | 1    | Parameters          | 66.11 K          | 66.11 K           | --      |
> | 2    | FLOPs               | 1.95 M           | 1.95 M            | --      |
> | 3    | Model Size (KB)     | 276.52           | 269.89            | -2.40%  |
> | 4    | Inference Time (ms) | 1.572            | 0.074             | -95.31% |
>
> --- In-Memory (RAM) Footprint Change ---
>
> PyTorch Model: 276.52 KB
>
> ExecuTorch Program: 269.89 KB
>
> Difference: -2.40%
>
> 
>
> --- Throughput Change (on baseline hardware) ---
>
> PyTorch: 1.2404 GFLOPS
>
> ExecuTorch: 26.4224 GFLOPS
>
> Speedup Factor: 21.30x faster



#### Return to *Basic Usage Example*:

* With those three model characteristic variables calculated, you are ready to use them as inputs into the library.
  * `weight_dtype` 
  * `pte_size_bytes`
  * `time_ms_inf` 




---

## Limitations / Warnings / Future Improvements:

1. **Currently, only two types of memory data types are estimated (`INT8` and `FP8`).**  Warnings:
   - The mapping from model param dtype to the columns in the table are approximate, see [code line](https://github.com/pmdscully/Edge_AI_Perf/blob/bf07357b21b540eb01d1a473813121246f37b148/lib_edge_eval/lib_edge_eval.py#L113).
   - The hardware columns in the table specify `TOPS` (i.e. `INT`) and `TFLOPS` (i.e. `FP`), this is draft, as a broad estimate of the two dtype performance choices. 
   - Improvements are invited to: 
     - Replace TOPS / TFLOPS  with specific columns  for `OPS per dtype`  (e.g. INT4, INT8, FP8, BF16, TF32) into the processors hardware table for per data type performance throughput (from datasheets).
     - Update specific operations per second (OPS) per data type to the code mappings.

2. Incorporate efficiencies information into estimates: i.e.
   1.  `memory speed efficiencies`; a distinct performance bottleneck for large scale models and hardware.
   2. `ops to watts efficiencies`:  this is illustrated by the relationship from TOPS/TFLOPS to Watts, and shown in CO2-EQ-kg plot.




