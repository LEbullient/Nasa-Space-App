#!/usr/bin/env python3
"""
Process large MPA CSV files into smaller chunks for web loading
"""

import os
import csv
import sys
from pathlib import Path

# Increase CSV field size limit to handle large WKT geometry strings
csv.field_size_limit(sys.maxsize)

def chunk_csv_file(input_file, output_dir, chunk_size=500, prefix=""):
    """
    Split a large CSV file into smaller chunks
    """
    print(f"Processing {input_file}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read header
        
        chunk_num = 0
        row_count = 0
        current_chunk = []
        
        for row in reader:
            current_chunk.append(row)
            row_count += 1
            
            if len(current_chunk) >= chunk_size:
                # Write chunk to file
                chunk_filename = f"{prefix}chunk_{chunk_num:03d}.csv"
                chunk_path = os.path.join(output_dir, chunk_filename)
                
                with open(chunk_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)  # Write header
                    writer.writerows(current_chunk)
                
                print(f"  Created {chunk_filename} with {len(current_chunk)} records")
                
                chunk_num += 1
                current_chunk = []
        
        # Write remaining rows if any
        if current_chunk:
            chunk_filename = f"{prefix}chunk_{chunk_num:03d}.csv"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            with open(chunk_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerows(current_chunk)
            
            print(f"  Created {chunk_filename} with {len(current_chunk)} records")
            chunk_num += 1
    
    print(f"  Total: {row_count} records split into {chunk_num} chunks")
    return chunk_num

def main():
    # Configuration
    data_dir = Path("data/mpa")
    chunks_dir = Path("data/mpa_chunks")
    chunk_size = 300  # Smaller chunks for better web performance
    
    # Files to process
    poly_files = [
        "0poly.csv",
        "1poly.csv", 
        "2poly.csv"
    ]
    
    point_files = [
        "0points.csv",
        "1points.csv",
        "2points.csv"
    ]
    
    print("=== MPA File Chunking Process ===")
    print(f"Chunk size: {chunk_size} records per file")
    print(f"Output directory: {chunks_dir}")
    print()
    
    total_chunks = 0
    
    # Process polygon files
    print("Processing polygon files...")
    for i, filename in enumerate(poly_files):
        input_path = data_dir / filename
        if input_path.exists():
            chunks = chunk_csv_file(
                input_path, 
                chunks_dir, 
                chunk_size, 
                f"poly_{i}_"
            )
            total_chunks += chunks
        else:
            print(f"  Warning: {input_path} not found")
    
    print()
    
    # Process point files (these are smaller)
    print("Processing point files...")
    for i, filename in enumerate(point_files):
        input_path = data_dir / filename
        if input_path.exists():
            chunks = chunk_csv_file(
                input_path, 
                chunks_dir, 
                chunk_size * 2,  # Larger chunks for point files since they're smaller 
                f"point_{i}_"
            )
            total_chunks += chunks
        else:
            print(f"  Warning: {input_path} not found")
    
    print()
    print(f"=== Processing Complete ===")
    print(f"Total chunks created: {total_chunks}")
    print(f"Chunks saved to: {chunks_dir}")
    
    # Create a manifest file listing all chunks
    manifest_path = chunks_dir / "manifest.json"
    import json
    
    chunk_files = []
    for chunk_file in sorted(chunks_dir.glob("*.csv")):
        chunk_files.append(chunk_file.name)
    
    manifest = {
        "total_chunks": len(chunk_files),
        "chunk_size": chunk_size,
        "files": chunk_files
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest created: {manifest_path}")

if __name__ == "__main__":
    main()