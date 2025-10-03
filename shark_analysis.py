#!/usr/bin/env python3
"""
Shark Movement Analysis Script
Analyzes shark tracking data to calculate velocities and create visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) using the Haversine formula.
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def calculate_velocities(df):
    """
    Calculate average velocities between consecutive points.
    Returns lists of timestamps, velocities (km/h), and time differences.
    """
    velocities = []
    timestamps = []
    time_diffs = []
    
    for i in range(1, len(df)):
        # Get consecutive points
        prev_point = df.iloc[i-1]
        curr_point = df.iloc[i]
        
        # Parse timestamps
        prev_time = pd.to_datetime(prev_point['date'])
        curr_time = pd.to_datetime(curr_point['date'])
        
        # Calculate time difference in hours
        time_diff = (curr_time - prev_time).total_seconds() / 3600
        
        if time_diff > 0:  # Avoid division by zero
            # Calculate distance using Haversine formula
            distance = haversine_distance(
                prev_point['lon'], prev_point['lat'],
                curr_point['lon'], curr_point['lat']
            )
            
            # Calculate velocity in km/h
            velocity = distance / time_diff
            
            velocities.append(velocity)
            timestamps.append(curr_time)
            time_diffs.append(time_diff)
    
    return timestamps, velocities, time_diffs

def create_combined_plot(timestamps, velocities, shark_id, output_path):
    """
    Create a combined plot with line graph of velocity over time and
    histogram of velocity distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Line graph of velocity over time
    ax1.plot(timestamps, velocities, 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average Velocity (km/h)')
    ax1.set_title(f'Velocity Over Time - {shark_id}')
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Histogram of velocity distribution
    ax2.hist(velocities, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Average Velocity (km/h)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Velocity Distribution - {shark_id}')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to the histogram
    mean_vel = np.mean(velocities)
    median_vel = np.median(velocities)
    ax2.axvline(mean_vel, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_vel:.2f} km/h')
    ax2.axvline(median_vel, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_vel:.2f} km/h')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_vel, median_vel, len(velocities)

def analyze_shark_data(file_path):
    """
    Analyze a single shark tracking data file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract shark ID from filename
        filename = os.path.basename(file_path)
        shark_id = filename.replace('.csv', '')
        
        print(f"Processing {shark_id}...")
        print(f"  Total tracking points: {len(df)}")
        
        if len(df) < 2:
            print(f"  Skipping {shark_id} - insufficient data points")
            return None
        
        # Calculate velocities
        timestamps, velocities, time_diffs = calculate_velocities(df)
        
        if not velocities:
            print(f"  Skipping {shark_id} - no valid velocity calculations")
            return None
        
        print(f"  Valid velocity calculations: {len(velocities)}")
        
        # Create output filename
        output_path = f"/Users/anthony/hack/{shark_id}_analysis.png"
        
        # Create combined plot
        mean_vel, median_vel, num_points = create_combined_plot(timestamps, velocities, shark_id, output_path)
        
        print(f"  Mean velocity: {mean_vel:.2f} km/h")
        print(f"  Median velocity: {median_vel:.2f} km/h")
        print(f"  Analysis saved to: {output_path}")
        
        return {
            'shark_id': shark_id,
            'num_points': num_points,
            'mean_velocity': mean_vel,
            'median_velocity': median_vel,
            'max_velocity': max(velocities),
            'min_velocity': min(velocities)
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    """
    Main function to process all shark data files.
    """
    # List of data files to process
    data_files = [
        '160424_2013_132346pnas_atn.csv',
        '160424_2014_141195pnas_atn.csv',
        '160424_2015_141261pnas_atn_filtered.csv',
        '160424_2015_141264pnas_atn.csv',
        '160424_2015_141268pnas_atn.csv',
        '160424_2015_141270pnas_atn.csv',
        '160424_2016_106744pnas_atn_filtered.csv',
        '160424_2016_106745pnas_atn_filtered.csv',
        '160424_2016_106746pnas_atn_filtered.csv',
        '160424_2016_106747pnas_atn_filtered.csv',
        '160424_2016_106748pnas_atn.csv',
        '160424_2016_141262pnas_atn_filtered.csv',
        '160424_2016_141263pnas_atn.csv',
        '160424_2016_141265pnas_atn.csv',
        '160424_2016_141266pnas_atn.csv',
        '160424_2016_165927pnas_atn.csv',
        '160424_2016_165928pnas_atn.csv'
    ]
    
    data_dir = "/Users/anthony/hack/data"
    results = []
    
    print("Starting shark movement analysis...")
    print("=" * 50)
    
    for filename in data_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            result = analyze_shark_data(file_path)
            if result:
                results.append(result)
        else:
            print(f"File not found: {file_path}")
        print("-" * 30)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    print(f"Successfully processed: {len(results)} out of {len(data_files)} files")
    
    if results:
        all_means = [r['mean_velocity'] for r in results]
        all_medians = [r['median_velocity'] for r in results]
        
        print(f"Overall mean velocity across all sharks: {np.mean(all_means):.2f} km/h")
        print(f"Overall median velocity across all sharks: {np.mean(all_medians):.2f} km/h")
        print(f"Fastest recorded velocity: {max([r['max_velocity'] for r in results]):.2f} km/h")
        
        # Create summary table
        print("\nDetailed Results:")
        print(f"{'Shark ID':<35} {'Points':<8} {'Mean (km/h)':<12} {'Median (km/h)':<14} {'Max (km/h)':<12}")
        print("-" * 85)
        for result in results:
            print(f"{result['shark_id']:<35} {result['num_points']:<8} {result['mean_velocity']:<12.2f} {result['median_velocity']:<14.2f} {result['max_velocity']:<12.2f}")

if __name__ == "__main__":
    main()