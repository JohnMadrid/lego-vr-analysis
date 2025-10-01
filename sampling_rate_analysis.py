import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def detect_timestamp_format(timestamps):
    """
    Detect the format of timestamp values based on the number of digits in a typical value.
    
    Args:
        timestamps: pandas Series of timestamp values
    
    Returns:
        str: 'seconds', 'milliseconds', 'microseconds', 'nanoseconds', or 'unknown'
    """
    if not pd.api.types.is_numeric_dtype(timestamps):
        return 'unknown'
    
    # Get a sample of non-null values to determine the typical magnitude
    sample = timestamps.dropna().head(100)
    if len(sample) == 0:
        return 'unknown'
    
    # Use the median value to get a representative timestamp
    typical_value = sample.median()
    
    # Convert to string to count digits
    typical_value_str = str(int(typical_value))
    num_digits = len(typical_value_str)
    
    # Determine format based on the number of digits
    # These ranges are based on standard Unix timestamp lengths
    if 10 <= num_digits <= 11:
        return 'seconds'
    elif 12 <= num_digits <= 14:
        return 'milliseconds'
    elif 15 <= num_digits <= 17:
        return 'microseconds'
    elif 18 <= num_digits <= 20:
        return 'nanoseconds'
    else:
        return 'unknown'

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
    
    print(f"First 5 timestamps from '{time_column}':\n{data[time_column].head().to_string(index=False)}")
    
    # Convert time column to datetime if it's not already
    if data[time_column].dtype == 'object':
        try:
            # Try to convert to datetime
            data[time_column] = pd.to_datetime(data[time_column])
        except:
            print(f"Warning: Could not convert {time_column} to datetime format")
            return None
    
    # Detect timestamp format for numeric timestamps
    if pd.api.types.is_numeric_dtype(data[time_column]):
        timestamp_format = detect_timestamp_format(data[time_column])
        print(f"Detected timestamp format: {timestamp_format}")
        
        # Convert to seconds based on detected format
        if timestamp_format == 'seconds':
            data['time'] = (data[time_column] - data[time_column].iloc[0])
        elif timestamp_format == 'milliseconds':
            data['time'] = (data[time_column] - data[time_column].iloc[0]) / 1000
        elif timestamp_format == 'microseconds':
            data['time'] = (data[time_column] - data[time_column].iloc[0]) / 1e6
        elif timestamp_format == 'nanoseconds':
            data['time'] = (data[time_column] - data[time_column].iloc[0]) / 1e9
        else:
            print(f"Warning: Unknown timestamp format. Assuming nanoseconds as a fallback.")
            data['time'] = (data[time_column] - data[time_column].iloc[0]) / 1e9
    else:
        # If already datetime, use total seconds from the first timestamp
        data['time'] = (pd.to_datetime(data[time_column]) - pd.to_datetime(data[time_column].iloc[0])).dt.total_seconds()

    # Calculate time differences
    time_diffs = data['time'].diff().dropna()
    
    # Calculate sampling rate statistics
    sampling_rates = 1 / time_diffs
    avg_sampling_rate = sampling_rates.mean()
    median_sampling_rate = sampling_rates.median()
    std_sampling_rate = sampling_rates.std()
    min_sampling_rate = sampling_rates.min()
    max_sampling_rate = sampling_rates.max()
    
    # Calculate time span
    total_duration = data['time'].max() - data['time'].min()
    
    print(f"Time span: {total_duration:.2f} seconds")
    print(f"Average sampling rate: {avg_sampling_rate:.2f} Hz")
    print(f"Median sampling rate: {median_sampling_rate:.2f} Hz")
    print(f"Standard deviation: {std_sampling_rate:.2f} Hz")
    print(f"Min sampling rate: {min_sampling_rate:.2f} Hz")
    print(f"Max sampling rate: {max_sampling_rate:.2f} Hz")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Time differences over time
    ax1.plot(data['time'][1:], time_diffs, 'b-', alpha=0.7, linewidth=0.5)
    ax1.axhline(time_diffs.mean(), color='red', linestyle='--', label=f'Mean: {time_diffs.mean():.4f}s')
    ax1.set_xlim(0, data['time'].max())
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Time Difference (seconds)')
    ax1.set_title(f'{data_name}: Time Between Samples')
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for building model periods if column exists
    if 'is_building_model' in data.columns:
        # Find the start and end times of building periods
        building_periods = []
        in_building = False
        start_time = None
        
        for idx, is_building in enumerate(data['is_building_model']):
            if is_building and not in_building:
                # Start of building period
                start_time = data['time'].iloc[idx]
                in_building = True
            elif not is_building and in_building:
                # End of building period
                end_time = data['time'].iloc[idx-1]  # Use previous timestamp as end
                building_periods.append((start_time, end_time))
                in_building = False
        
        # Handle case where building period extends to the end
        if in_building:
            end_time = data['time'].iloc[-1]
            building_periods.append((start_time, end_time))
        
        if building_periods:
            # Add vertical lines at start and end of each building period
            for i, (start_time, end_time) in enumerate(building_periods):
                # Only add labels for the first occurrence of each type
                start_label = 'Building Start' if i == 0 else ""
                end_label = 'Building End' if i == 0 else ""
                
                ax1.axvline(x=start_time, color='green', alpha=0.7, linewidth=1, label=start_label)
                ax1.axvline(x=end_time, color='gray', alpha=0.7, linewidth=1, label=end_label)
                
                # Add text label in the middle of the building period
                mid_time = (start_time + end_time) / 2
                
                # Find the model_name for this building period
                building_mask = (data['time'] >= start_time) & (data['time'] <= end_time) & (data['is_building_model'] == True)
                if building_mask.any():
                    model_names = data.loc[building_mask, 'model_name'].dropna().unique()
                    if len(model_names) > 0:
                        model_name = model_names[0]  # Use the first unique model name
                        
                        # Get y-axis limits for text positioning
                        y_min, y_max = ax1.get_ylim()
                        text_y = (y_min + y_max) / 2  # Middle of y-axis
                        
                        # Add text label
                        ax1.text(mid_time, text_y, model_name, 
                                ha='center', va='center', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                                fontsize=8, rotation=90)
            
            print(f"Added {len(building_periods)} building periods with start/end lines and model labels")
            print(f"Building periods: {building_periods}")
    ax1.legend()
    
    # Plot 2: Sampling rate histogram
    ax2.hist(sampling_rates, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(avg_sampling_rate, color='red', linestyle='--', label=f'Mean: {avg_sampling_rate:.2f} Hz')
    ax2.axvline(median_sampling_rate, color='orange', linestyle='--', label=f'Median: {median_sampling_rate:.2f} Hz')
    ax2.set_xlim(0, None)
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
        'total_duration': total_duration,
        'total_samples': len(data)
    }

def main():
    """Main function to analyze sampling rates of eye and body data."""
    
    print("Starting sampling rate analysis...")
    
    # File paths
    eye_data_path = "D:/LegoVR/unity-lego-vr/Other_than_in_project_files/ET_Data/09_ET_Data_2025-08-29.csv"
    body_data_path = "D:/LegoVR/unity-lego-vr/Other_than_in_project_files/BT_Data/09_BT_Data_2025-08-29.csv"

    print(f"Looking for eye data at: {eye_data_path}")
    print(f"Looking for body data at: {body_data_path}")
    
    # Check if files exist
    if not os.path.exists(eye_data_path):
        print(f"ERROR: Eye data file not found at {eye_data_path}")
        print("Please update the file path in the script or ensure the file exists.")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    if not os.path.exists(body_data_path):
        print(f"ERROR: Body data file not found at {body_data_path}")
        print("Please update the file path in the script or ensure the file exists.")
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
        print("\nAnalyzing eye data...")
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 