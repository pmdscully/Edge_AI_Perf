# Edge_AI_Perf

(Alpha version: correct for the specified use case: T4, Thailand, from Jan 2026)

## Objective:

The goal of this python library is to make simple/ quick benchmark estimates of DL/LLM machine learning models for their deployment viability onto edge computing devices. 

### Approach

The methodology is simple:

1. Measure the performance characteristics on known hardware (currently Google Colab's T4 GPU).
2. Quantify their performance values for other hardware profiles, according to their ratio differences from T4.

This approach makes it simple and viable to update a known table of hardware, and benchmark many devices simultaneously.

It differs from ML Commons' ML Perf which perform benchmarks through tests to give precise results for limited hardware. Whereas this methodology gives **quick estimated results for all hardware specs extracted from a data source** (below). 

### **Performance Measurements:**

The output estimated measures are **per 1 million inference-time inputs** for both ***`INT8`*** and ***`FP16`*** data types, for the following metrics:

* Inference time, 
* Watt Hours, 
* CO2EQ
* Model fitment viability (i.e. inputs, model param and OS) into operational runtime memory.

### **Data Sources:**

Markdown table data is loaded (ETL) into a `pandas Dataframe` for processing.

* `edge_hardware_table.md`
  * A fixed set of devices from a hardware table, which specifies their (verified) hardware profile characteristics.
* `EmissionsFactor_Calculation.md` 
  * A known CO2-EQ emissions factor calculation (Scope 2 + Scope 3) from kilowatt hours (KWH), collected from national greenhouse gas organization source(s).
  * Currently implemented for: `Thailand` for runtimes from  `Jan 2026` to  `March 2026`.



## Installation:

(Use `!pip` for notebook installations, Jupyter, Colab, etc.)
```bash
pip install git+https://github.com/pmdscully/Edge_AI_Perf.git --upgrade
```



## Usage Example:

* **First, specify or calculate from your model the following items:**
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

* Specify the hardware used (in your current environment) for the baseline (use **`Colab T4`**)

```python
from lib_edge_eval import fetch_edge_hardware_dataframe, calculate_edge_metrics, Plotting

# ======= Load the Hardware Table: ========
df_final = fetch_edge_hardware_dataframe()

# ======= Specify the Baseline from the Hardware Table: ========
baseline_index = 8 # 8 = NVIDIA T4 GPU

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
  1. Gains/losses of dtypes choices, i.e. quantizing into `int8` or floating points.
  2. CO2-eq Efficiencies, according to the 

![Barplot_Latency_CO2](README.assets/Barplot_Latency_CO2.png)

## Future Improvements:

1. Incorporate memory speed efficiencies; a distinct performance bottleneck for large scale models and hardware.
2. Add and specify columns  for `OPS per dtype`  (e.g. INT4, INT8, FP8, BF16, TF32) into the processors hardware table for per data type performance throughput (from datasheets).



