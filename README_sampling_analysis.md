# Sampling Rate Analysis Scripts

This repository contains scripts to analyze the sampling rates of eye tracking and body tracking data.

## Files

1. **`sampling_rate_analysis.py`** - Comprehensive analysis script with detailed statistics and visualizations
2. **`simple_sampling_analysis.py`** - Simplified version for quick analysis
3. **`sampling_rate_analysis_notebook.py`** - Version designed for Jupyter notebook usage

## Usage

### Option 1: Jupyter Notebook (Recommended)

1. Open your Jupyter notebook
2. Copy the contents of `sampling_rate_analysis_notebook.py`
3. Run the following code in a cell:

```python
# Import the analysis function
from sampling_rate_analysis_notebook import run_analysis

# Run the analysis
run_analysis()
```

### Option 2: Standalone Script

1. Update the file paths in the script to match your data locations
2. Run the script:

```bash
python sampling_rate_analysis.py
```

### Option 3: Direct Function Call

You can also use the analysis function directly:

```python
import pandas as pd
from sampling_rate_analysis_notebook import analyze_sampling_rate

# Load your data
eye_data = pd.read_csv("path/to/eye_data.csv")
body_data = pd.read_csv("path/to/body_data.csv")

# Analyze eye data
eye_stats = analyze_sampling_rate(eye_data, 'gaze_capture_time', 'Eye Data')

# Analyze body data (using first column as time)
body_stats = analyze_sampling_rate(body_data, body_data.columns[0], 'Body Data')
```

## Data Requirements

### Eye Data
- Must contain a column named `gaze_capture_time`
- Time values should be in a format that pandas can convert to datetime

### Body Data
- The first column should contain time information
- Time values should be in a format that pandas can convert to datetime

## Output

The scripts provide:

1. **Statistical Analysis**:
   - Average sampling rate (Hz)
   - Median sampling rate (Hz)
   - Standard deviation
   - Min/Max sampling rates
   - Total duration and sample count

2. **Visualizations**:
   - Time differences between samples over time
   - Sampling rate distribution histogram

3. **Comparison**:
   - Side-by-side comparison of eye and body data sampling rates
   - Assessment of how similar the rates are

## File Paths

Update these paths in the scripts to match your data locations:

```python
eye_data_path = "D:/LegoVR/unity-lego-vr/Other_than_in_project_files/ET_Data/P001_ET_Data_2025-07-31.csv"
body_data_path = "D:/LegoVR/unity-lego-vr/Other_than_in_project_files/BT_Data/test_BT_Data_2025-07-31.csv"
```

## Example Output

```
=== Eye Data Sampling Rate Analysis ===
Total samples: 15000
Time span: 79.50 seconds
Average sampling rate: 188.57 Hz
Median sampling rate: 188.68 Hz
Standard deviation: 2.34 Hz
Min sampling rate: 180.12 Hz
Max sampling rate: 195.23 Hz

=== Body Data Sampling Rate Analysis ===
Total samples: 8000
Time span: 79.50 seconds
Average sampling rate: 100.63 Hz
Median sampling rate: 100.00 Hz
Standard deviation: 1.45 Hz
Min sampling rate: 95.24 Hz
Max sampling rate: 105.26 Hz

==================================================
SUMMARY COMPARISON
==================================================
Metric               Eye Data        Body Data      
--------------------------------------------------
Avg Rate (Hz)        188.57          100.63        
Median Rate (Hz)     188.68          100.00        
Std Dev (Hz)         2.34            1.45          
Duration (s)         79.50           79.50         
Total Samples        15000           8000          

Rate difference: 87.94 Hz
✗ Sampling rates are significantly different (> 5 Hz difference)
```

## Troubleshooting

1. **File not found**: Update the file paths in the script
2. **Column not found**: Check that your data has the expected column names
3. **Time format error**: Ensure your time data is in a recognizable format
4. **Python not found**: Use the virtual environment: `bricks-analysis/Scripts/python.exe`

## Dependencies

- pandas
- matplotlib
- numpy

These should already be installed in your virtual environment. 