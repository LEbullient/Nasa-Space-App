#!/usr/bin/env python3
"""
Train MaxEnt model and generate foraging habitat prediction tiles for Leaflet maps.
Generates predictions for every month from July 2013 to January 2017.
Exports as tiles in the format: tiles/predict/YYYY-MM-01/{z}/{x}/{y}.png
"""

import elapid
import pandas as pd
from sklearn import metrics
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
from rasterio.transform import from_origin
import shap
from PIL import Image
import os
import math
import warnings
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings('ignore')

###############################################################################
# 0) Config
###############################################################################

CSV_PATH   = "./train_data/train_data.csv"
FEATURES   = ["sin_m","cos_m","sst","chl","poc"]
ID_COLUMN  = "id"

VAR_SST    = "sst"
VAR_CHL    = "chlor_a"
VAR_POC    = "poc"
GRID_RES   = 0.1

OUT_CAL    = "calibration_curve.png"

# Tile generation config
MAX_ZOOM = 5  # Maximum zoom level for prediction tiles
TILE_SIZE = 256  # Standard tile size
OUTPUT_DIR = "../tiles/predict"

# Data directories
DATA_SST_DIR = "../data/sst"
DATA_CHL_DIR = "../data/chl"
DATA_POC_DIR = "../data/poc"

###############################################################################
# Helper functions for tile generation
###############################################################################

def create_foraging_colormap():
    """Create a colormap for foraging probability: green (low) to red (high)"""
    colors = ['#00ff00', '#ffff00', '#ff0000']  # Green -> Yellow -> Red
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('foraging', colors, N=n_bins)
    return cmap

def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile numbers (Web Mercator)"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def num2deg(x, y, zoom):
    """Convert tile numbers to lat/lon bounds (Web Mercator)"""
    n = 2.0 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_left, lat_top, lon_right, lat_bottom

def lat_to_mercator_y(lat):
    """Convert latitude to Web Mercator Y coordinate (normalized 0-1)"""
    lat_rad = math.radians(lat)
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0

def mercator_y_to_lat(y_norm):
    """Convert Web Mercator Y coordinate (normalized 0-1) back to latitude"""
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_norm))))

def create_transparent_tile():
    """Create a transparent tile"""
    return Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))

def create_tile_from_prediction(prob_grid, xs, ys, x, y, zoom, colormap):
    """Create a single tile from the probability grid with proper Mercator projection"""
    # Get tile bounds in geographic coordinates
    lon_left, lat_top, lon_right, lat_bottom = num2deg(x, y, zoom)
    
    # Create coordinate grids for the tile (256x256 pixels)
    tile_lons = np.linspace(lon_left, lon_right, TILE_SIZE)
    
    # Create linear spacing in Web Mercator space for latitude
    y_top_merc = lat_to_mercator_y(lat_top)
    y_bottom_merc = lat_to_mercator_y(lat_bottom)
    tile_y_mercs = np.linspace(y_top_merc, y_bottom_merc, TILE_SIZE)
    
    # Convert back to latitude for each pixel
    tile_lats = np.array([mercator_y_to_lat(y_merc) for y_merc in tile_y_mercs])
    
    tile_lon_grid, tile_lat_grid = np.meshgrid(tile_lons, tile_lats)
    
    # Find bounds in data coordinates
    lat_mask = (ys >= lat_bottom) & (ys <= lat_top)
    lon_mask = (xs >= lon_left) & (xs <= lon_right)
    
    if not np.any(lat_mask) or not np.any(lon_mask):
        return create_transparent_tile()
    
    # Get data subset
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        return create_transparent_tile()
    
    lat_subset = ys[lat_indices]
    lon_subset = xs[lon_indices]
    # Note: prob_grid is (lat, lon) indexed
    data_subset = prob_grid[np.ix_(lat_indices, lon_indices)]
    
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
    distances, indices = tree.query(tile_coords, k=1, distance_upper_bound=0.2)  # 0.2 degree max distance
    
    # Fill tile values
    tile_flat = np.full(TILE_SIZE * TILE_SIZE, np.nan)
    valid_tile_mask = distances < np.inf
    tile_flat[valid_tile_mask] = valid_values[indices[valid_tile_mask]]
    
    tile_data = tile_flat.reshape(TILE_SIZE, TILE_SIZE)
    
    # Check if we have any valid data
    if np.all(np.isnan(tile_data)):
        return create_transparent_tile()
    
    # Probabilities are already normalized to 0-1, apply colormap directly
    colored_data = colormap(tile_data)
    
    # Handle transparency for areas with no data
    alpha = np.where(np.isnan(tile_data), 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    img_rgba = (colored_data * 255).astype(np.uint8)
    img_rgba[:, :, 3] = alpha  # Set alpha channel
    
    return Image.fromarray(img_rgba)

def generate_tiles_for_month(prob_grid, xs, ys, date_str):
    """Generate all tiles for a given month's prediction"""
    print(f"  Generating tiles for {date_str}...")
    
    colormap = create_foraging_colormap()
    
    # Generate tiles for each zoom level
    for zoom in range(MAX_ZOOM + 1):
        print(f"    Zoom level {zoom}...")
        
        # Calculate tile range that covers the data extent
        # Data extent
        lon_min, lon_max = xs.min(), xs.max()
        lat_min, lat_max = ys.min(), ys.max()
        
        # Get tile range
        x_min, y_max = deg2num(lat_min, lon_min, zoom)
        x_max, y_min = deg2num(lat_max, lon_max, zoom)
        
        # Expand range slightly to ensure coverage
        x_min = max(0, x_min - 1)
        x_max = min(2**zoom - 1, x_max + 1)
        y_min = max(0, y_min - 1)
        y_max = min(2**zoom - 1, y_max + 1)
        
        tiles_created = 0
        tiles_skipped = 0
        
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # Create output directory
                tile_dir = os.path.join(OUTPUT_DIR, date_str, str(zoom), str(x))
                os.makedirs(tile_dir, exist_ok=True)
                
                # Always regenerate tiles (force overwrite)
                tile_path = os.path.join(tile_dir, f"{y}.png")
                
                # Create tile
                tile_img = create_tile_from_prediction(prob_grid, xs, ys, x, y, zoom, colormap)
                
                # Save tile only if it has content
                # Check if tile is completely transparent
                tile_array = np.array(tile_img)
                if np.any(tile_array[:, :, 3] > 0):  # Has non-transparent pixels
                    tile_img.save(tile_path, "PNG")
                    tiles_created += 1
                else:
                    tiles_skipped += 1
        
        print(f"      Created {tiles_created} tiles, skipped {tiles_skipped} for zoom {zoom}")

###############################################################################
# Generate month list from July 2013 to January 2017
###############################################################################

def generate_months_list():
    """Generate list of months from July 2013 to January 2017"""
    months = []
    
    # Start: July 2013 (2013-07)
    # End: January 2017 (2017-01)
    
    for year in range(2013, 2018):
        start_month = 7 if year == 2013 else 1
        end_month = 1 if year == 2017 else 12
        
        for month in range(start_month, end_month + 1):
            if year == 2017 and month > 1:
                break
            
            # Determine last day of month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                end_day = 31
            elif month == 2:
                # Check for leap year
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                    end_day = 29
                else:
                    end_day = 28
            else:
                end_day = 30
            
            # Create timestamp (use 15th as representative date)
            ts = pd.Timestamp(f"{year}-{month:02d}-15")
            
            # Create file paths
            sst_file = f"{DATA_SST_DIR}/AQUA_MODIS.{year}{month:02d}01_{year}{month:02d}{end_day}.L3m.MO.SST.sst.4km.nc"
            chl_file = f"{DATA_CHL_DIR}/AQUA_MODIS.{year}{month:02d}01_{year}{month:02d}{end_day}.L3m.MO.CHL.chlor_a.4km.nc"
            poc_file = f"{DATA_POC_DIR}/AQUA_MODIS.{year}{month:02d}01_{year}{month:02d}{end_day}.L3m.MO.POC.poc.4km.nc"
            
            # Date string for tile directory (first day of month)
            date_str = f"{year}-{month:02d}-01"
            
            months.append({
                "year": year,
                "month": month,
                "TS": ts,
                "SST_FILE": sst_file,
                "CHL_FILE": chl_file,
                "POC_FILE": poc_file,
                "DATE_STR": date_str
            })
    
    return months

MONTHS_PREDICT = generate_months_list()

print(f"Generated {len(MONTHS_PREDICT)} months from July 2013 to January 2017")
print(f"First month: {MONTHS_PREDICT[0]['DATE_STR']}")
print(f"Last month: {MONTHS_PREDICT[-1]['DATE_STR']}")

###############################################################################
# 1) Load & prepare data
###############################################################################

print("\n" + "="*60)
print("LOADING TRAINING DATA")
print("="*60)

df = pd.read_csv(CSV_PATH)
X = df[FEATURES].copy()
y = (df[ID_COLUMN] != "fake").astype(int)

mask = X.notna().all(axis=1)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df = df.loc[mask].reset_index(drop=True)

print(f"Data after NaN removal: {len(X)} samples")
print("Feature statistics:")
for col in FEATURES:
    print(f"  {col}: min={X[col].min():.6f}, max={X[col].max():.6f}, mean={X[col].mean():.6f}")
print(f"Target distribution: {y.value_counts().to_dict()}")

###############################################################################
# 2) Train model
###############################################################################

print("\n" + "="*60)
print("TRAINING MAXENT MODEL")
print("="*60)

model = elapid.MaxentModel()
print("Fitting MaxEnt model...")
model.fit(X, y)
print("Model fitting completed")

import joblib

MODEL_PATH = "./train_data/maxent_model.joblib"
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

yp = model.predict(X)
print("Prediction statistics:")
print(f"  Min: {np.min(yp):.6f}")
print(f"  Max: {np.max(yp):.6f}")
print(f"  Mean: {np.mean(yp):.6f}")
print(f"  NaN count: {np.isnan(yp).sum()}")

# Calculate AUC
if np.isnan(yp).any():
    print("WARNING: Predictions contain NaN values!")
    valid_mask = ~np.isnan(yp)
    y_clean = y[valid_mask]
    yp_clean = yp[valid_mask]
    print(f"Using {len(yp_clean)} valid predictions out of {len(yp)} total")
    if len(yp_clean) > 0:
        print("Training AUC:", metrics.roc_auc_score(y_clean, yp_clean))
else:
    print("Training AUC:", metrics.roc_auc_score(y, yp))

###############################################################################
# 3) Calibration curve
###############################################################################

print("\n" + "="*60)
print("GENERATING CALIBRATION CURVE")
print("="*60)

p_all = model.predict(X)
if np.isnan(p_all).any():
    valid_mask = ~np.isnan(p_all)
    y_cal = y[valid_mask]
    p_cal = p_all[valid_mask]
else:
    y_cal = y
    p_cal = p_all

frac_pos, mean_pred = calibration_curve(y_cal, p_cal, n_bins=10, strategy="quantile")

plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],"--",label="ideal")
plt.plot(mean_pred, frac_pos, marker="o", label="MaxEnt (cloglog)")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration (quantile bins)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_CAL, dpi=120)
plt.close()
print(f"[OK] Saved calibration curve → {OUT_CAL}")

###############################################################################
# 4) Predict to raster grid and generate tiles for all months
###############################################################################

print("\n" + "="*60)
print("GENERATING FORAGING HABITAT PREDICTIONS AND TILES")
print("="*60)

def open_any(path: str) -> xr.Dataset:
    p = Path(path)
    return xr.open_dataset(p)

def interp_grid(ds: xr.Dataset, var: str, xs: np.ndarray, ys: np.ndarray, ts=None):
    da = ds[var]
    dai = da.interp(lat=("y", ys), lon=("x", xs))
    return np.asarray(dai.values)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each month
for i, month_info in enumerate(MONTHS_PREDICT):
    TS = month_info["TS"]
    SST_FILE = month_info["SST_FILE"]
    CHL_FILE = month_info["CHL_FILE"]
    POC_FILE = month_info["POC_FILE"]
    DATE_STR = month_info["DATE_STR"]
    
    print(f"\n[{i+1}/{len(MONTHS_PREDICT)}] Processing {DATE_STR}...")
    
    # Check if files exist
    if not (os.path.exists(SST_FILE) and os.path.exists(CHL_FILE) and os.path.exists(POC_FILE)):
        print(f"  WARNING: Missing data files for {DATE_STR}, skipping...")
        continue
    
    try:
        # Load environmental data
        ds_sst = open_any(SST_FILE)
        ds_chl = open_any(CHL_FILE)
        ds_poc = open_any(POC_FILE)
        
        # Define prediction grid - GLOBAL COVERAGE
        lat_min = -90
        lat_max = 90
        lon_min = -180
        lon_max = 180
        
        xs = np.arange(lon_min, lon_max + GRID_RES, GRID_RES)
        ys = np.arange(lat_min, lat_max + GRID_RES, GRID_RES)
        
        # Get variable names
        v_sst = VAR_SST if VAR_SST in ds_sst.data_vars else list(ds_sst.data_vars)[0]
        v_chl = VAR_CHL if VAR_CHL in ds_chl.data_vars else list(ds_chl.data_vars)[0]
        v_poc = VAR_POC if VAR_POC in ds_poc.data_vars else list(ds_poc.data_vars)[0]
        
        # Interpolate environmental data to grid
        sst_grid = interp_grid(ds_sst, v_sst, xs, ys, TS)
        chl_grid = interp_grid(ds_chl, v_chl, xs, ys, TS)
        poc_grid = interp_grid(ds_poc, v_poc, xs, ys, TS)
        
        # Calculate month features
        month = TS.month
        sin_m = np.sin(2*np.pi*month/12.0)
        cos_m = np.cos(2*np.pi*month/12.0)
        
        # Create prediction dataframe
        mesh_lon, mesh_lat = np.meshgrid(xs, ys)
        n = mesh_lon.size
        grid_df = pd.DataFrame({
            "sin_m": np.full(n, sin_m),
            "cos_m": np.full(n, cos_m),
            "sst": sst_grid.ravel().astype(float),
            "chl": chl_grid.ravel().astype(float),
            "poc": poc_grid.ravel().astype(float) ** 0.5,
        })
        
        # Predict foraging probabilities
        valid = np.isfinite(grid_df["sst"].values) & np.isfinite(grid_df["chl"].values) & np.isfinite(grid_df["poc"].values)
        probs = np.full(n, np.nan, dtype=float)
        probs[valid] = model.predict(grid_df.loc[valid, FEATURES])
        prob_grid = probs.reshape(mesh_lon.shape)
        
        print(f"  Prediction stats: min={np.nanmin(probs):.4f}, max={np.nanmax(probs):.4f}, mean={np.nanmean(probs):.4f}")
        print(f"  Valid pixels: {valid.sum()} / {n} ({100*valid.sum()/n:.1f}%)")
        
        # Generate tiles
        generate_tiles_for_month(prob_grid, xs, ys, DATE_STR)
        
        # Close datasets
        ds_sst.close()
        ds_chl.close()
        ds_poc.close()
        
        print(f"  [OK] Completed {DATE_STR}")
        
    except Exception as e:
        print(f"  ERROR processing {DATE_STR}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*60)
print("TILE GENERATION COMPLETED!")
print("="*60)
print(f"Tiles saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"Processed {len(MONTHS_PREDICT)} months from July 2013 to January 2017")
