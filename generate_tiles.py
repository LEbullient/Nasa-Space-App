#!/usr/bin/env python3
"""
Generate map tiles from NetCDF files for chlorophyll and sea surface temperature data.
Creates tiles in the format: tiles/{chl,sst}/YYYY-MM-01/{z}/{x}/{y}.png
"""

import os
import re
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.colors as mcolors
import math
import warnings
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
MAX_ZOOM = 6  # Maximum zoom level to generate (reduced for faster processing)
TILE_SIZE = 256  # Standard tile size
CHL_DATA_DIR = "data/chl"
SST_DATA_DIR = "data/sst"
POC_DATA_DIR = "data/poc"
OUTPUT_DIR = "tiles"

# Focus on North Atlantic region where sharks are tracked
FOCUS_BOUNDS = {
    'lat_min': 20,   # Southern boundary
    'lat_max': 70,   # Northern boundary  
    'lon_min': -90,  # Western boundary
    'lon_max': 10    # Eastern boundary
}

# Color maps for visualization
def create_chlorophyll_colormap():
    """Create a colormap for chlorophyll concentration (mg/m³)"""
    colors = ['#000080', '#0066cc', '#00ccff', '#66ff66', '#ffff00']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('chl', colors, N=n_bins)
    return cmap

def create_sst_colormap():
    """Create a colormap for sea surface temperature (°C)"""
    colors = ['#000066', '#0033cc', '#ff3300', '#ff9900', '#ffcc00']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('sst', colors, N=n_bins)
    return cmap

def create_poc_colormap():
    """Create a colormap for particulate organic carbon (mg/m³)"""
    colors = ['#2c2c54', '#40407a', '#706fd3', '#f7b731', '#ff5252']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('poc', colors, N=n_bins)
    return cmap

def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile numbers"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def num2deg(x, y, zoom):
    """Convert tile numbers to lat/lon bounds"""
    n = 2.0 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_left, lat_top, lon_right, lat_bottom

def extract_date_from_filename(filename):
    """Extract date from MODIS filename and return first day of month"""
    # Example: AQUA_MODIS.20130701_20130731.L3m.MO.CHL.chlor_a.4km.nc
    match = re.search(r'\.(\d{8})_\d{8}\.', filename)
    if match:
        date_str = match.group(1)  # e.g., '20130701'
        year = date_str[:4]
        month = date_str[4:6]
        return f"{year}-{month}-01"
    return None

def normalize_data(data, data_type):
    """Normalize data to 0-1 range for colormap application"""
    if data_type == 'chl':
        # Chlorophyll is often log-scaled due to wide range
        # Typical range: 0.01 to 100 mg/m³
        data_clean = np.where(data > 0, data, np.nan)
        log_data = np.log10(data_clean)
        # Normalize log data to 0-1 (adjust range as needed)
        vmin, vmax = -2, 2  # 0.01 to 100 mg/m³ in log scale
        normalized = (log_data - vmin) / (vmax - vmin)
    elif data_type == 'sst':
        # SST in Celsius, typical range: -2 to 35°C
        vmin, vmax = -2, 35
        normalized = (data - vmin) / (vmax - vmin)
    elif data_type == 'poc':
        # POC is also log-scaled due to wide range
        # Typical range: 1 to 1000 mg/m³
        data_clean = np.where(data > 0, data, np.nan)
        log_data = np.log10(data_clean)
        # Normalize log data to 0-1 (adjust range as needed)
        vmin, vmax = 0, 3  # 1 to 1000 mg/m³ in log scale
        normalized = (log_data - vmin) / (vmax - vmin)
    else:
        # Generic normalization
        vmin, vmax = np.nanpercentile(data, [2, 98])
        normalized = (data - vmin) / (vmax - vmin)
    
    # Clip to 0-1 range
    return np.clip(normalized, 0, 1)

def create_tile(data, lat, lon, x, y, zoom, colormap, data_type):
    """Create a single tile from the data with proper geographic projection"""
    # Get tile bounds in geographic coordinates
    lon_left, lat_top, lon_right, lat_bottom = num2deg(x, y, zoom)
    
    # Create coordinate grids for the tile (256x256 pixels)
    # For Web Mercator, we need to create the grid in the projected space
    # then convert back to lat/lon for data lookup
    tile_lons = np.linspace(lon_left, lon_right, TILE_SIZE)
    
    # For latitude, we need to account for Web Mercator's non-linear projection
    # Convert tile bounds to Web Mercator Y coordinates
    def lat_to_mercator_y(lat):
        """Convert latitude to Web Mercator Y coordinate (normalized 0-1)"""
        lat_rad = math.radians(lat)
        return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0
    
    def mercator_y_to_lat(y_norm):
        """Convert Web Mercator Y coordinate (normalized 0-1) back to latitude"""
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_norm))))
    
    # Create linear spacing in Web Mercator space
    y_top_merc = lat_to_mercator_y(lat_top)
    y_bottom_merc = lat_to_mercator_y(lat_bottom)
    tile_y_mercs = np.linspace(y_top_merc, y_bottom_merc, TILE_SIZE)
    
    # Convert back to latitude for each pixel
    tile_lats = np.array([mercator_y_to_lat(y_merc) for y_merc in tile_y_mercs])
    
    tile_lon_grid, tile_lat_grid = np.meshgrid(tile_lons, tile_lats)
    
    # Initialize output tile
    tile_output = np.full((TILE_SIZE, TILE_SIZE), np.nan)
    
    # Find bounds in data coordinates
    lat_mask = (lat >= lat_bottom) & (lat <= lat_top)
    lon_mask = (lon >= lon_left) & (lon <= lon_right)
    
    if not np.any(lat_mask) or not np.any(lon_mask):
        return create_transparent_tile()
    
    # Get data subset
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        return create_transparent_tile()
    
    lat_subset = lat[lat_indices]
    lon_subset = lon[lon_indices]
    data_subset = data[np.ix_(lat_indices, lon_indices)]
    
    # Use nearest neighbor interpolation to map data to tile grid
    # Create coordinate arrays for data points
    data_coords = np.column_stack([
        np.repeat(lat_subset, len(lon_subset)),
        np.tile(lon_subset, len(lat_subset))
    ])
    data_values = data_subset.flatten()
    
    # Remove NaN values
    valid_mask = ~np.isnan(data_values)
    if not np.any(valid_mask):
        return create_transparent_tile()
    
    valid_coords = data_coords[valid_mask]
    valid_values = data_values[valid_mask]
    
    # Create tile coordinate array
    tile_coords = np.column_stack([
        tile_lat_grid.flatten(),
        tile_lon_grid.flatten()
    ])
    
    # Build KDTree and find nearest neighbors
    tree = cKDTree(valid_coords)
    distances, indices = tree.query(tile_coords, k=1, distance_upper_bound=0.1)  # 0.1 degree max distance
    
    # Fill tile values
    tile_flat = np.full(TILE_SIZE * TILE_SIZE, np.nan)
    valid_tile_mask = distances < np.inf
    tile_flat[valid_tile_mask] = valid_values[indices[valid_tile_mask]]
    
    tile_data = tile_flat.reshape(TILE_SIZE, TILE_SIZE)
    
    # Check if we have any valid data
    if np.all(np.isnan(tile_data)):
        return create_transparent_tile()
    
    # Normalize data
    normalized_data = normalize_data(tile_data, data_type)
    
    # Apply colormap
    colored_data = colormap(normalized_data)
    
    # Handle transparency for areas with no data
    alpha = np.where(np.isnan(tile_data), 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    img_rgba = (colored_data * 255).astype(np.uint8)
    img_rgba[:, :, 3] = alpha  # Set alpha channel
    
    return Image.fromarray(img_rgba)

def create_transparent_tile():
    """Create a transparent tile"""
    return Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))

def check_if_tiles_exist(data_type, date_str, output_base_dir):
    """Check if all tiles for a given data type and date already exist"""
    for zoom in range(MAX_ZOOM + 1):
        # Check tiles for the entire world at all zoom levels
        x_min = 0
        y_min = 0
        x_max = 2**zoom - 1
        y_max = 2**zoom - 1
        
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tile_path = os.path.join(output_base_dir, data_type, date_str, str(zoom), str(x), f"{y}.png")
                if not os.path.exists(tile_path):
                    return False
    return True

def process_netcdf_file(nc_file_path, data_type, output_base_dir):
    """Process a single NetCDF file and generate tiles"""
    print(f"Processing {nc_file_path}...")
    
    # Extract date from filename
    filename = os.path.basename(nc_file_path)
    date_str = extract_date_from_filename(filename)
    if not date_str:
        print(f"Could not extract date from {filename}")
        return
    
    print(f"Date: {date_str}")
    
    # Check if all tiles for this file already exist
    if check_if_tiles_exist(data_type, date_str, output_base_dir):
        print(f"All tiles for {filename} already exist, skipping...")
        return
    
    # Load NetCDF data
    try:
        ds = xr.open_dataset(nc_file_path)
        
        # Get the data variable (different names for CHL vs SST vs POC)
        if data_type == 'chl':
            data_var = 'chlor_a'
        elif data_type == 'sst':
            data_var = 'sst'
        elif data_type == 'poc':
            data_var = 'poc'
        else:
            print(f"Unknown data type: {data_type}")
            return
        
        if data_var not in ds.variables:
            print(f"Variable {data_var} not found in {filename}")
            print(f"Available variables: {list(ds.variables.keys())}")
            return
        
        data = ds[data_var].values
        lat = ds['lat'].values
        lon = ds['lon'].values
        
        print(f"Data shape: {data.shape}")
        print(f"Lat range: {lat.min():.2f} to {lat.max():.2f}")
        print(f"Lon range: {lon.min():.2f} to {lon.max():.2f}")
        
        # Create colormap
        if data_type == 'chl':
            colormap = create_chlorophyll_colormap()
        elif data_type == 'sst':
            colormap = create_sst_colormap()
        elif data_type == 'poc':
            colormap = create_poc_colormap()
        else:
            colormap = create_sst_colormap()  # Default fallback
        
        # Generate tiles for each zoom level
        for zoom in range(MAX_ZOOM + 1):
            print(f"  Generating zoom level {zoom}...")
            
            # Generate tiles for the entire world at all zoom levels
            x_min = 0
            y_min = 0
            x_max = 2**zoom - 1
            y_max = 2**zoom - 1
            
            tiles_created = 0
            tiles_skipped = 0
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    # Create output directory
                    tile_dir = os.path.join(output_base_dir, data_type, date_str, str(zoom), str(x))
                    os.makedirs(tile_dir, exist_ok=True)
                    
                    # Check if tile already exists
                    tile_path = os.path.join(tile_dir, f"{y}.png")
                    if os.path.exists(tile_path):
                        tiles_skipped += 1
                        continue
                    
                    # Create tile
                    tile_img = create_tile(data, lat, lon, x, y, zoom, colormap, data_type)
                    
                    # Save tile
                    tile_img.save(tile_path, "PNG")
                    tiles_created += 1
            
            print(f"    Created {tiles_created} tiles, skipped {tiles_skipped} existing tiles for zoom {zoom}")
        
        ds.close()
        print(f"Completed processing {filename}")
        
    except Exception as e:
        print(f"Error processing {nc_file_path}: {e}")

def main():
    """Main function to process all NetCDF files"""
    print("Starting tile generation...")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process chlorophyll files
    print("\n=== Processing Chlorophyll Data ===")
    chl_files = sorted([f for f in os.listdir(CHL_DATA_DIR) if f.endswith('.nc')])
    for chl_file in chl_files:
        chl_path = os.path.join(CHL_DATA_DIR, chl_file)
        process_netcdf_file(chl_path, 'chl', OUTPUT_DIR)
    
    # Process SST files
    print("\n=== Processing SST Data ===")
    sst_files = sorted([f for f in os.listdir(SST_DATA_DIR) if f.endswith('.nc')])
    for sst_file in sst_files:
        sst_path = os.path.join(SST_DATA_DIR, sst_file)
        process_netcdf_file(sst_path, 'sst', OUTPUT_DIR)
    
    # Process POC files
    print("\n=== Processing POC Data ===")
    if os.path.exists(POC_DATA_DIR):
        poc_files = sorted([f for f in os.listdir(POC_DATA_DIR) if f.endswith('.nc')])
        for poc_file in poc_files:
            poc_path = os.path.join(POC_DATA_DIR, poc_file)
            process_netcdf_file(poc_path, 'poc', OUTPUT_DIR)
    else:
        print(f"POC data directory {POC_DATA_DIR} not found, skipping POC processing")
    
    print("\nTile generation completed!")
    print(f"Tiles saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()