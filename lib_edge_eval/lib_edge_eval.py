import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
import requests
import io
import re

def parse_markdown_table(content: str) -> pd.DataFrame:
    """
    Function 1: Parses a raw markdown table from string into a pandas DataFrame.
    """
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    table_data = []
    for line in lines:
        # Check if it's a table row and skip the alignment/separator row (e.g., |---|---|)
        if line.startswith("|"):
            if re.match(r"^\|[\s\-:|]+\|$", line) or re.match(r"^\|[\s\-:|]+$", line):
                continue
            # Split by pipe and remove leading/trailing whitespace from elements
            row = [cell.strip() for cell in line.split("|")[1:-1]]
            table_data.append(row)
    if not table_data:
        raise ValueError("No valid Markdown table data found in file.")
    # The first row contains the headers, subsequent rows contain the data
    raw_headers = table_data[0]
    data_rows = table_data[1:]
    # Clean markdown bolding tokens (**) out of headers
    headers = [re.sub(r"\*\*|__", "", h).strip() for h in raw_headers]
    return pd.DataFrame(data_rows, columns=headers).replace(r'\*\*', '', regex=True)

def fetch_edge_hardware_dataframe():
    def extract_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function 2: Extracts numerical values for columns required for arithmetic calculations.
        Handles units (g, W), prefixes (~, $), missing dashes (--), commas, and text notes like (FP16).
        """
        def _get_first_float(val):
            """Extracts the first valid floating point number from a string."""
            if pd.isna(val) or val == "--" or val == "N/A":
                return None
            val_str = str(val).replace(",", "").strip()
            # Regex looks for an integer or a decimal number
            match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(match.group()) if match else None
        def _get_wattage(val):
            """Specifically extracts the wattage which is highlighted inside bolding tokens (e.g., **15W**)."""
            if pd.isna(val):
                return None
            val_str = str(val).replace(",", "")
            # Find numeric value immediately preceding a 'W' inside bold asterisks
            w = val_str.split('/')[-1].replace('W','')
            return float(w)
        df_clean = df.copy()
        # Apply extractions to target columns
        if 'Peak TOPS (INT8/NPU)' in df_clean.columns:
            df_clean['Peak TOPS (INT8/NPU)'] = df_clean['Peak TOPS (INT8/NPU)'].apply(_get_first_float)
        if 'Peak TFLOPS (FP32/GPU)' in df_clean.columns:
            df_clean['Peak TFLOPS (FP32/GPU)'] = df_clean['Peak TFLOPS (FP32/GPU)'].apply(_get_first_float)
        if 'Total Unit Weight' in df_clean.columns:
            df_clean['Total Unit Weight'] = df_clean['Total Unit Weight'].apply(_get_first_float)
        if 'Approx. Unit Cost ($)' in df_clean.columns:
            df_clean['Approx. Unit Cost ($)'] = df_clean['Approx. Unit Cost ($)'].apply(_get_first_float)
        # Create a brand new column just for Wattage
        if 'Peak Electrical Profile [V, A, W]' in df_clean.columns:
            df_clean['Peak Wattage (W)'] = df_clean['Peak Electrical Profile [V, A, W]'].apply(_get_wattage)
        if 'Est. RAM for Params (Bytes)' in df_clean.columns:
            df_clean['Est. RAM for Params (Bytes)'] = df_clean['Est. RAM for Params (Bytes)'].astype(float)
        return df_clean

    def convert_stars_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function 3: Converts columns containing star symbols (★) into numeric integer ratings (1-3).
        """
        df_stars = df.copy()
        def count_stars(val):
            if pd.isna(val):
                return 0
            return str(val).count("★")
        if 'Memory Speed' in df_stars.columns:
            df_stars['Memory Speed'] = df_stars['Memory Speed'].apply(count_stars)
        if 'Power Efficiency' in df_stars.columns:
            df_stars['Power Efficiency'] = df_stars['Power Efficiency'].apply(count_stars)
        return df_stars
        
    url = 'https://raw.githubusercontent.com/pmdscully/Edge_AI_Perf/refs/heads/main/edge_hardware_table.md'
    response = requests.get(url)
    df_raw = parse_markdown_table(response.text)
    df_numeric = extract_numeric_values(df_raw)
    return convert_stars_to_numeric(df_numeric)

def get_co2_emissions(kwh_per_1M, country_code='THA'):
    url = 'https://raw.githubusercontent.com/pmdscully/Edge_AI_Perf/refs/heads/main/EmissionsFactor_Calculation.md'
    response = requests.get(url)
    dfe = parse_markdown_table(response.text)
    factor_kwh_co2eq = dfe[dfe['Country Code']==country_code]['Emission Factor Value  (kgCO₂e / kWh)'].astype(float).sum()
    co2eq_per_1M = kwh_per_1M * factor_kwh_co2eq
    return co2eq_per_1M

def calculate_edge_metrics(df, baseline_idx, baseline_ms, baseline_mem, baseline_dtype='int8', country='THA'):
    """
    Calculates comparative performance, energy, and CO2 metrics across edge devices.
    Handles cross-dtype time correction based on the baseline device's throughput ratio.
    Includes per-inference latency metrics in milliseconds.
    """
    def get_time_baselines(b_idx, b_ms, b_dtype):
        """Sub-function to calculate correctly scaled time baselines for both dtypes."""
        t4_tops = df['Peak TOPS (INT8/NPU)'].iloc[b_idx]
        t4_tflops = df['Peak TFLOPS (FP32/GPU)'].iloc[b_idx]
        
        measured_hr_per_1M = ms_to_hr(b_ms) * 1e6
        
        if b_dtype.lower() in ['int8', 'fp8']:
            # Measured time is for INT8; calculate what FP32 would have taken
            int8_base = measured_hr_per_1M
            fp32_base = measured_hr_per_1M * (t4_tops / t4_tflops)
        else:
            # Measured time is for FP32/FP16; calculate what INT8 would have taken
            fp32_base = measured_hr_per_1M
            int8_base = measured_hr_per_1M * (t4_tflops / t4_tops)
        return int8_base, fp32_base, t4_tops, t4_tflops

    def get_kwh(time_in_hrs, baseline_watts):
        return (baseline_watts / 1000) * time_in_hrs

    def ms_to_hr(time_ms):
        return time_ms / 1000 / 60 / 60
    
    # 1. Get corrected baselines
    int8_base_hr, fp32_base_hr, t4_tops, t4_tflops = get_time_baselines(baseline_idx, baseline_ms, baseline_dtype)
    
    # Initialize results DataFrame
    dfr = pd.DataFrame(df[['Device Name', 'Peak Wattage (W)']].copy())
    dfr['Baseline'] = ''
    dfr.loc[baseline_idx, 'Baseline'] = '🗸'
    
    # 2. RAM Fit Estimation
    dfr['Int8/FP8 Params Fit (Est.)'] = (df['Est. RAM for Params (Bytes)'] >= baseline_mem).map({True: '🗸', False: ''})
    
    # 3. Performance Scaling using respective baselines
    dfr['Hours per 1M Inf. INT8'] = (t4_tops / df['Peak TOPS (INT8/NPU)']) * int8_base_hr
    dfr['Hours per 1M Inf. FP16'] = (t4_tflops / df['Peak TFLOPS (FP32/GPU)']) * fp32_base_hr
    
    # --- ADDED: Latency per Single Inference (ms) ---
    # Conversion: (Hours / 1,000,000) * 3,600,000 ms/hr = Hours * 3.6 # to ms
    # Conversion: (Hours / 1,000,000) * 3,600,000 ms/hr = Hours * 0.0036 # to sec
    dfr['Latency per Inf. INT8 (s)'] = dfr['Hours per 1M Inf. INT8'] * 0.0036
    dfr['Latency per Inf. FP16 (s)'] = dfr['Hours per 1M Inf. FP16'] * 0.0036
    
    # 4. Energy Consumption
    dfr['KWh per 1M INT8'] = dfr.apply(lambda x: get_kwh(x['Hours per 1M Inf. INT8'], x['Peak Wattage (W)']), axis=1)
    dfr['KWh per 1M FP16'] = dfr.apply(lambda x: get_kwh(x['Hours per 1M Inf. FP16'], x['Peak Wattage (W)']), axis=1)
    
    # 5. Carbon Emissions
    dfr['CO2-EQ-kg per 1M INT8'] = dfr['KWh per 1M INT8'].apply(lambda x: get_co2_emissions(x, country_code=country))
    dfr['CO2-EQ-kg per 1M FP16'] = dfr['KWh per 1M FP16'].apply(lambda x: get_co2_emissions(x, country_code=country))
    dfr = dfr.drop(columns=['Peak Wattage (W)'])
    dfr = dfr[['Device Name', 'Baseline', 'Int8/FP8 Params Fit (Est.)',
               'Latency per Inf. INT8 (s)', 
               'Latency per Inf. FP16 (s)', 
               'Hours per 1M Inf. INT8', 'Hours per 1M Inf. FP16',
               'KWh per 1M INT8',
               'KWh per 1M FP16', 
               'CO2-EQ-kg per 1M INT8', 
               'CO2-EQ-kg per 1M FP16']]
    return dfr

class Plotting:
    def plot_hardware_comparisons(df, 
                                  metrics = ['Peak Wattage (W)', 'Total Unit Weight', 'Approx. Unit Cost ($)', 'Memory Speed', 'Power Efficiency'],
                                  titles = ['Peak Power Consumption (Watts)', 'Total Unit Weight (grams)', 'Approx. Unit Cost ($)', 'Memory Speed', 'Power Efficiency']
                                  ):
        """Generates a multi-panel comparison of edge hardware metrics."""
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 3), sharey=True)
    
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                axes[i].barh(df['Device Name'], df[metric], color='skyblue', edgecolor='navy')
                axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
                axes[i].set_xlabel(metric+' (log-scale)')
                axes[i].grid(axis='x', linestyle='--', alpha=0.6)
    
                # Apply log scale to handle wide range of values
                axes[i].set_xscale('log')
            else:
                axes[i].set_title(f'Column {metric} not found')
        axes[0].invert_yaxis()
        plt.tight_layout()
        plt.show()
        
    def plot_comparison_metrics(df_results, title_suffix=""):
        """
        Generates a professional, condensed three-panel horizontal bar chart.
        Uses darkgrid for reduced brightness and tightened vertical layout.
        """
        # Professional, lower-brightness style
        sns.set_theme(style="darkgrid")
        
        # Increased height to fit the third panel while keeping it condensed
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10))
        
        label_fontsize = 8
        title_fontsize = 10
        
        # Legend with muted colors
        red_patch = mpatches.Patch(color='#d62728', label='Baseline')
        int8_legend = mpatches.Patch(color='#4c72b0', label='INT8 (Solid)')
        fp16_legend = mpatches.Patch(edgecolor='#4c72b0', facecolor='white', hatch='//', linewidth=1.2, label='FP16/FP8 (Hatched)')
    
        # Added 'xscale' parameter to toggle linear vs log per panel
        def draw_paired_bars(ax, col_int8, col_fp16, title, xscale='log'):
            y_pos = np.arange(len(df_results))
            width = 0.35  
    
            # Professional muted palette
            palette = sns.color_palette("crest", len(df_results))
            
            # 1. INT8 Bars
            bars_int8 = ax.barh(y_pos + width/2, df_results[col_int8], width, 
                                color=palette, alpha=0.85, zorder=3)
            
            # 2. FP16/FP8 Hatched Bars
            bars_fp16 = ax.barh(y_pos - width/2, df_results[col_fp16], width, 
                                color='white', edgecolor=palette, hatch='//', linewidth=0.75, zorder=3)
    
            # Baseline Highlighting
            for i in range(len(df_results)):
                if df_results.iloc[i]['Baseline'] == '🗸':
                    bars_int8[i].set_color('#d62728')
                    bars_fp16[i].set_edgecolor('#d62728')
    
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_results['Device Name'], fontsize=label_fontsize)
            
            # Style baseline text
            for label in ax.get_yticklabels():
                device_name = label.get_text()
                if df_results.loc[df_results['Device Name'] == device_name, 'Baseline'].values[0] == '🗸':
                    label.set_color('#d62728')
                    label.set_fontweight('bold')
    
            ax.set_xscale(xscale)
            ax.set_title(title, loc='left', fontsize=title_fontsize, fontweight='bold', pad=8)
            ax.legend(handles=[red_patch, int8_legend, fp16_legend], loc='lower right', fontsize=7, frameon=True)
            ax.invert_yaxis() 
    
        # --- PANEL 0: Time per Inference (Linear) ---
        # Using the new dataframe columns calculated natively in seconds
        draw_paired_bars(ax0, 'Latency per Inf. INT8 (s)', 'Latency per Inf. FP16 (s)', 
                         f'Latency per Single Inference {title_suffix}', xscale='linear')
        ax0.set_xlabel('Seconds (Linear Scale)', fontsize=label_fontsize)
    
        # --- PANEL 1: Hours per 1M (Log) ---
        # Updated to the new 'Inf.' column names
        draw_paired_bars(ax1, 'Hours per 1M Inf. INT8', 'Hours per 1M Inf. FP16', 
                         f'Total Time: Hours per 1M Ops {title_suffix}', xscale='log')
        ax1.set_xlabel('Hours (Log Scale)', fontsize=label_fontsize)
    
        # --- PANEL 2: CO2-EQ (Log) ---
        # Updated to the new '-kg' column names
        draw_paired_bars(ax2, 'CO2-EQ-kg per 1M INT8', 'CO2-EQ-kg per 1M FP16', 
                         f'Environmental Impact: kg CO2-eq per 1M Ops {title_suffix}', xscale='log')
        ax2.set_xlabel('kg CO2-eq (Log Scale)', fontsize=label_fontsize)
    
        plt.subplots_adjust(hspace=0.4) 
        plt.tight_layout()
        plt.show()

    
if __name__ == '__main__':
    # ======= Calculate from Baseline: ========
    df_final = fetch_edge_hardware_dataframe()
    baseline_index = 8 # 8 = T4
    baseline_memory_bytes = 1000
    baseline_time_ms = 1
    baseline_dtype = ['int8','fp8'][0]

    df_comparison = calculate_edge_metrics(
        df=df_final,
        baseline_idx=baseline_index,
        baseline_ms=baseline_time_ms,
        baseline_mem=baseline_memory_bytes,
        baseline_dtype=baseline_dtype
    )

    display(df_comparison.round(2))
    # Execution call
    Plotting.plot_comparison_metrics(df_comparison)
