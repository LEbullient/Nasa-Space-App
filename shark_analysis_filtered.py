#!/usr/bin/env python3
"""
Shark Movement Analysis Script with High-Velocity Point Detection and Filtering
Analyzes shark tracking data, identifies unrealistic velocities, and creates filtered datasets.
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

def identify_high_velocity_points(df, velocity_threshold=80):
    """
    Identify points involved in velocity calculations greater than the threshold.
    Returns indices of problematic points and velocity details.
    """
    high_velocity_segments = []
    problematic_indices = set()
    
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
            
            if velocity > velocity_threshold:
                high_velocity_segments.append({
                    'segment_index': i,
                    'prev_point_index': i-1,
                    'curr_point_index': i,
                    'prev_time': prev_time,
                    'curr_time': curr_time,
                    'prev_coords': (prev_point['lat'], prev_point['lon']),
                    'curr_coords': (curr_point['lat'], curr_point['lon']),
                    'distance_km': distance,
                    'time_diff_hours': time_diff,
                    'velocity_kmh': velocity
                })
                # Add both points involved in the high velocity calculation
                problematic_indices.add(i-1)
                problematic_indices.add(i)
    
    return high_velocity_segments, sorted(list(problematic_indices))

def calculate_velocities_with_filtering(df, velocity_threshold=80):
    """
    Calculate average velocities between consecutive points, excluding high-velocity segments.
    Returns lists of timestamps, velocities (km/h), time differences, and high-velocity info.
    """
    velocities = []
    timestamps = []
    time_diffs = []
    high_velocity_segments = []
    
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
            
            if velocity > velocity_threshold:
                high_velocity_segments.append({
                    'segment': i,
                    'velocity': velocity,
                    'distance': distance,
                    'time_diff': time_diff,
                    'prev_time': prev_time,
                    'curr_time': curr_time
                })
            else:
                velocities.append(velocity)
                timestamps.append(curr_time)
                time_diffs.append(time_diff)
    
    return timestamps, velocities, time_diffs, high_velocity_segments

def create_filtered_dataset(df, high_velocity_indices, output_path):
    """
    Create a new CSV file with high-velocity points removed.
    """
    # Remove the problematic indices
    filtered_df = df.drop(df.index[high_velocity_indices]).reset_index(drop=True)
    
    # Save the filtered dataset
    filtered_df.to_csv(output_path, index=False)
    
    return len(high_velocity_indices), len(filtered_df)

def create_combined_plot_with_filtering(timestamps, velocities, shark_id, output_path, high_velocity_info):
    """
    Create a combined plot with line graph of velocity over time and
    histogram of velocity distribution, including information about filtered points.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Line graph of velocity over time
    ax1.plot(timestamps, velocities, 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average Velocity (km/h)')
    ax1.set_title(f'Velocity Over Time - {shark_id}\n(Filtered: {len(high_velocity_info)} high-velocity segments removed)')
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Histogram of velocity distribution
    ax2.hist(velocities, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Average Velocity (km/h)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Velocity Distribution - {shark_id}\n(After filtering)')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to the histogram
    if velocities:
        mean_vel = np.mean(velocities)
        median_vel = np.median(velocities)
        ax2.axvline(mean_vel, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_vel:.2f} km/h')
        ax2.axvline(median_vel, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_vel:.2f} km/h')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.mean(velocities) if velocities else 0, np.median(velocities) if velocities else 0, len(velocities)

def analyze_shark_data_with_filtering(file_path, velocity_threshold=80, create_filtered_csv=True):
    """
    Analyze a single shark tracking data file with high-velocity filtering.
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
        
        # Identify high-velocity points
        high_velocity_segments, problematic_indices = identify_high_velocity_points(df, velocity_threshold)
        
        print(f"  High-velocity segments found: {len(high_velocity_segments)}")
        print(f"  Problematic points to remove: {len(problematic_indices)}")
        
        # Print details of high-velocity segments
        if high_velocity_segments:
            print(f"  High-velocity segment details:")
            for segment in high_velocity_segments[:5]:  # Show first 5
                print(f"    Segment {segment['segment_index']}: {segment['velocity_kmh']:.2f} km/h "
                     f"({segment['distance_km']:.2f} km in {segment['time_diff_hours']:.2f} hours)")
            if len(high_velocity_segments) > 5:
                print(f"    ... and {len(high_velocity_segments) - 5} more segments")
        
        # Create filtered dataset if requested
        if create_filtered_csv and problematic_indices:
            filtered_csv_path = f"/Users/anthony/hack/data/filtered_{shark_id}.csv"
            removed_count, remaining_count = create_filtered_dataset(df, problematic_indices, filtered_csv_path)
            print(f"  Filtered dataset saved: {filtered_csv_path}")
            print(f"  Points removed: {removed_count}, Points remaining: {remaining_count}")
        
        # Calculate velocities with filtering
        timestamps, velocities, time_diffs, high_vel_info = calculate_velocities_with_filtering(df, velocity_threshold)
        
        if not velocities:
            print(f"  Skipping {shark_id} - no valid velocity calculations after filtering")
            return None
        
        print(f"  Valid velocity calculations after filtering: {len(velocities)}")
        
        # Create output filename
        output_path = f"/Users/anthony/hack/{shark_id}_filtered_analysis.png"
        
        # Create combined plot
        mean_vel, median_vel, num_points = create_combined_plot_with_filtering(
            timestamps, velocities, shark_id, output_path, high_vel_info
        )
        
        print(f"  Mean velocity (filtered): {mean_vel:.2f} km/h")
        print(f"  Median velocity (filtered): {median_vel:.2f} km/h")
        print(f"  Analysis saved to: {output_path}")
        
        return {
            'shark_id': shark_id,
            'original_points': len(df),
            'high_velocity_segments': len(high_velocity_segments),
            'problematic_points': len(problematic_indices),
            'valid_calculations': num_points,
            'mean_velocity': mean_vel,
            'median_velocity': median_vel,
            'max_velocity': max(velocities) if velocities else 0,
            'min_velocity': min(velocities) if velocities else 0,
            'high_velocity_details': high_velocity_segments
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    """
    Main function to process all shark data files with high-velocity filtering.
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
    velocity_threshold = 80  # km/h
    
    print("Starting shark movement analysis with high-velocity filtering...")
    print(f"Velocity threshold: {velocity_threshold} km/h")
    print("=" * 60)
    
    total_high_velocity_segments = 0
    total_problematic_points = 0
    
    for filename in data_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            result = analyze_shark_data_with_filtering(file_path, velocity_threshold, create_filtered_csv=True)
            if result:
                results.append(result)
                total_high_velocity_segments += result['high_velocity_segments']
                total_problematic_points += result['problematic_points']
        else:
            print(f"File not found: {file_path}")
        print("-" * 40)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS (AFTER FILTERING)")
    print("=" * 60)
    print(f"Successfully processed: {len(results)} out of {len(data_files)} files")
    print(f"Total high-velocity segments detected: {total_high_velocity_segments}")
    print(f"Total problematic points removed: {total_problematic_points}")
    
    if results:
        all_means = [r['mean_velocity'] for r in results if r['mean_velocity'] > 0]
        all_medians = [r['median_velocity'] for r in results if r['median_velocity'] > 0]
        
        print(f"Overall mean velocity (filtered): {np.mean(all_means):.2f} km/h")
        print(f"Overall median velocity (filtered): {np.mean(all_medians):.2f} km/h")
        print(f"Highest velocity after filtering: {max([r['max_velocity'] for r in results if r['max_velocity'] > 0]):.2f} km/h")
        
        # Create detailed results table
        print("\nDetailed Results (After Filtering):")
        print(f"{'Shark ID':<35} {'Orig':<6} {'HV Seg':<6} {'Removed':<8} {'Valid':<6} {'Mean':<8} {'Median':<8} {'Max':<8}")
        print("-" * 95)
        for result in results:
            print(f"{result['shark_id']:<35} {result['original_points']:<6} {result['high_velocity_segments']:<6} "
                 f"{result['problematic_points']:<8} {result['valid_calculations']:<6} {result['mean_velocity']:<8.2f} "
                 f"{result['median_velocity']:<8.2f} {result['max_velocity']:<8.2f}")
        
        # Show most problematic datasets
        print("\nMost Problematic Datasets (by number of high-velocity segments):")
        sorted_results = sorted(results, key=lambda x: x['high_velocity_segments'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. {result['shark_id']}: {result['high_velocity_segments']} segments, "
                 f"{result['problematic_points']} points removed")

if __name__ == "__main__":
    main()