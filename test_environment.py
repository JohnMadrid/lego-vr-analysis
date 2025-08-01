#!/usr/bin/env python3
"""
Test script to verify the bricks-analysis environment is working correctly.
"""

import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def test_environment():
    """Test that all required packages are working."""
    print("Testing bricks-analysis environment...")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Seaborn version: {sns.__version__}")
    
    # Create a simple test plot
    print("\nCreating a test plot...")
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='x', y='y')
    plt.title('Test Plot - Bricks Analysis Environment')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.savefig('test_plot.png')
    plt.close()
    
    print("✅ Environment test completed successfully!")
    print("✅ Test plot saved as 'test_plot.png'")

if __name__ == "__main__":
    test_environment() 