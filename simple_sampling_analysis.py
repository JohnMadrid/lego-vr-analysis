import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def quick_sampling_analysis(data, time_column, data_name):
    """
    Quick analysis of sampling rate for a dataset.
    """
    print(f"\n=== {data_name} Analysis ===")
    print(f"Data shape: {data.shape}")
    print(f"Time column: {time_column}")
    
    if time_column not in data.columns:
        print(f"Error: Column '{time_column}' not found!")
        print(f"Available columns: {list(data.columns)}")
        return
    
    # Convert to datetime if needed
    if data[time_column].dtype == 'object':
        try:
            data[time_column] = pd.to_datetime(data[time_column])
        except:
            print(f"Warning: Could not convert {time_column} to datetime")
            return
    
    # Calculate time differences
    time_diffs = data[time_column].diff().dropna()
    
    # Convert to seconds
    if time_diffs.dtype == 'timedelta64[ns]':
        time_diffs_seconds = time_diffs.dt.total_seconds()
    else:
        # Handle different time formats
        time_diffs_seconds = time_diffs / 1e9 if time_diffs.max() > 1e6 else time_diffs
    
    # Calculate sampling rate
    sampling_rates = 1 / time_diffs_seconds
    avg_rate = sampling_rates.mean()
    
    print(f"Average sampling rate: {avg_rate:.2f} Hz")
    print(f"Min sampling rate: {sampling_rates.min():.2f} Hz")
    print(f"Max sampling rate: {sampling_rates.max():.2f} Hz")
    print(f"Standard deviation: {sampling_rates.std():.2f} Hz")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_diffs_seconds.index, time_diffs_seconds, 'b-', alpha=0.7)
    plt.axhline(time_diffs_seconds.mean(), color='red', linestyle='--', 
                label=f'Mean: {time_diffs_seconds.mean():.4f}s')
    plt.xlabel('Sample Index')
    plt.ylabel('Time Difference (seconds)')
    plt.title(f'{data_name}: Time Between Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.hist(sampling_rates, bins=30, alpha=0.7, color='green')
    plt.axvline(avg_rate, color='red', linestyle='--', label=f'Mean: {avg_rate:.2f} Hz')
    plt.xlabel('Sampling Rate (Hz)')
    plt.ylabel('Frequency')
    plt.title(f'{data_name}: Sampling Rate Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return avg_rate

# Example usage with the data files
if __name__ == "__main__":
    try:
        # Load data
        print("Loading data files...")
        
        # Eye data
        eye_data = pd.read_csv("D:/LegoVR/unity-lego-vr/Other_than_in_project_files/ET_Data/P001_ET_Data_2025-07-31.csv")
        eye_rate = quick_sampling_analysis(eye_data, 'gaze_capture_time', 'Eye Data')
        
        # Body data
        body_data = pd.read_csv("D:/LegoVR/unity-lego-vr/Other_than_in_project_files/BT_Data/test_BT_Data_2025-07-31.csv")
        body_time_col = body_data.columns[0]  # First column
        body_rate = quick_sampling_analysis(body_data, body_time_col, 'Body Data')
        
        # Comparison
        if eye_rate and body_rate:
            print(f"\n=== COMPARISON ===")
            print(f"Eye data sampling rate: {eye_rate:.2f} Hz")
            print(f"Body data sampling rate: {body_rate:.2f} Hz")
            print(f"Difference: {abs(eye_rate - body_rate):.2f} Hz")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please check the file paths in the script.")
    except Exception as e:
        print(f"Error: {e}") 