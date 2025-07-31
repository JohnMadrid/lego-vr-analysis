import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def analyze_sampling_rate(data, time_column, data_name):
    """
    Analyze the sampling rate of a dataset using a time column.
    
    Args:
        data: pandas DataFrame
        time_column: name of the time column to analyze
        data_name: name of the dataset for reporting
    """
    print(f"\n=== {data_name} Sampling Rate Analysis ===")
    print(f"Total samples: {len(data)}")
    
    # Check if time column exists
    if time_column not in data.columns:
        print(f"Error: Column '{time_column}' not found in {data_name} data")
        print(f"Available columns: {list(data.columns)}")
        return None
    
    # Convert time column to datetime if it's not already
    if data[time_column].dtype == 'object':
        try:
            # Try to convert to datetime
            data[time_column] = pd.to_datetime(data[time_column])
        except:
            print(f"Warning: Could not convert {time_column} to datetime format")
            return None
    
    # Calculate time differences
    time_diffs = data[time_column].diff().dropna()
    
    # Convert to seconds if needed
    if time_diffs.dtype == 'timedelta64[ns]':
        time_diffs_seconds = time_diffs.dt.total_seconds()
    else:
        # Assume it's already in seconds or convert from nanoseconds
        time_diffs_seconds = time_diffs / 1e9 if time_diffs.max() > 1e6 else time_diffs
    
    # Calculate sampling rate statistics
    sampling_rates = 1 / time_diffs_seconds
    avg_sampling_rate = sampling_rates.mean()
    median_sampling_rate = sampling_rates.median()
    std_sampling_rate = sampling_rates.std()
    min_sampling_rate = sampling_rates.min()
    max_sampling_rate = sampling_rates.max()
    
    # Calculate time span
    total_duration = (data[time_column].max() - data[time_column].min())
    if hasattr(total_duration, 'total_seconds'):
        total_duration_seconds = total_duration.total_seconds()
    else:
        total_duration_seconds = total_duration / 1e9 if total_duration > 1e6 else total_duration
    
    print(f"Time span: {total_duration_seconds:.2f} seconds")
    print(f"Average sampling rate: {avg_sampling_rate:.2f} Hz")
    print(f"Median sampling rate: {median_sampling_rate:.2f} Hz")
    print(f"Standard deviation: {std_sampling_rate:.2f} Hz")
    print(f"Min sampling rate: {min_sampling_rate:.2f} Hz")
    print(f"Max sampling rate: {max_sampling_rate:.2f} Hz")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Time differences over time
    ax1.plot(time_diffs_seconds.index, time_diffs_seconds, 'b-', alpha=0.7, linewidth=0.5)
    ax1.axhline(time_diffs_seconds.mean(), color='red', linestyle='--', label=f'Mean: {time_diffs_seconds.mean():.4f}s')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Time Difference (seconds)')
    ax1.set_title(f'{data_name}: Time Between Samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sampling rate histogram
    ax2.hist(sampling_rates, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(avg_sampling_rate, color='red', linestyle='--', label=f'Mean: {avg_sampling_rate:.2f} Hz')
    ax2.axvline(median_sampling_rate, color='orange', linestyle='--', label=f'Median: {median_sampling_rate:.2f} Hz')
    ax2.set_xlabel('Sampling Rate (Hz)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{data_name}: Sampling Rate Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'avg_rate': avg_sampling_rate,
        'median_rate': median_sampling_rate,
        'std_rate': std_sampling_rate,
        'min_rate': min_sampling_rate,
        'max_rate': max_sampling_rate,
        'total_duration': total_duration_seconds,
        'total_samples': len(data)
    }

def main():
    """Main function to analyze sampling rates of eye and body data."""
    
    # File paths
    eye_data_path = "D:/LegoVR/unity-lego-vr/Other_than_in_project_files/ET_Data/P001_ET_Data_2025-07-31.csv"
    body_data_path = "D:/LegoVR/unity-lego-vr/Other_than_in_project_files/BT_Data/test_BT_Data_2025-07-31.csv"
    
    # Check if files exist
    if not os.path.exists(eye_data_path):
        print(f"Warning: Eye data file not found at {eye_data_path}")
        print("Please update the file path in the script.")
        return
    
    if not os.path.exists(body_data_path):
        print(f"Warning: Body data file not found at {body_data_path}")
        print("Please update the file path in the script.")
        return
    
    try:
        # Load eye data
        print("Loading eye data...")
        eye_data = pd.read_csv(eye_data_path)
        print(f"Eye data loaded successfully. Shape: {eye_data.shape}")
        print(f"Eye data columns: {list(eye_data.columns)}")
        
        # Load body data
        print("\nLoading body data...")
        body_data = pd.read_csv(body_data_path)
        print(f"Body data loaded successfully. Shape: {body_data.shape}")
        print(f"Body data columns: {list(body_data.columns)}")
        
        # Analyze eye data sampling rate
        eye_stats = analyze_sampling_rate(eye_data, 'gaze_capture_time', 'Eye Data')
        
        # Analyze body data sampling rate (using first column)
        body_time_column = body_data.columns[0]
        print(f"\nUsing '{body_time_column}' as the time column for body data")
        body_stats = analyze_sampling_rate(body_data, body_time_column, 'Body Data')
        
        # Summary comparison
        if eye_stats and body_stats:
            print("\n" + "="*50)
            print("SUMMARY COMPARISON")
            print("="*50)
            print(f"{'Metric':<20} {'Eye Data':<15} {'Body Data':<15}")
            print("-" * 50)
            print(f"{'Avg Rate (Hz)':<20} {eye_stats['avg_rate']:<15.2f} {body_stats['avg_rate']:<15.2f}")
            print(f"{'Median Rate (Hz)':<20} {eye_stats['median_rate']:<15.2f} {body_stats['median_rate']:<15.2f}")
            print(f"{'Std Dev (Hz)':<20} {eye_stats['std_rate']:<15.2f} {body_stats['std_rate']:<15.2f}")
            print(f"{'Duration (s)':<20} {eye_stats['total_duration']:<15.2f} {body_stats['total_duration']:<15.2f}")
            print(f"{'Total Samples':<20} {eye_stats['total_samples']:<15} {body_stats['total_samples']:<15}")
            
            # Calculate rate difference
            rate_diff = abs(eye_stats['avg_rate'] - body_stats['avg_rate'])
            print(f"\nRate difference: {rate_diff:.2f} Hz")
            
            if rate_diff < 1:
                print("✓ Sampling rates are very similar (< 1 Hz difference)")
            elif rate_diff < 5:
                print("⚠ Sampling rates are moderately different (1-5 Hz difference)")
            else:
                print("✗ Sampling rates are significantly different (> 5 Hz difference)")
        
    except Exception as e:
        print(f"Error loading or analyzing data: {str(e)}")
        print("Please check the file paths and data format.")

if __name__ == "__main__":
    main() 