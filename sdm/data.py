import xarray as xr
import pandas as pd
import os
import numpy as np
import re

from utils import calc_speed

features = ["sst", "chl", "poc"]
feature_datasets = {}
folders = ["../data/blue-shark-gps", "../data/blue-shark-gbif"]

SPEED_THRESHOLD = 3  # km/h - sharks typically don't swim faster than this

for feature in features:
    feature_dir = f"./train_data/{feature}"
    feature_datasets[feature] = {}
    for fname in os.listdir(feature_dir):
        if fname.endswith(".nc"):
            match = re.search(r'(\d{8}_\d{8})', fname)
            if match:
                date_key = match.group(1)
                feature_datasets[feature][date_key] = xr.open_dataset(os.path.join(feature_dir, fname))

# Pre-compute date ranges for faster lookup
def _build_date_lookup(datasets):
    """Build a lookup table for date ranges to avoid repeated parsing"""
    date_lookup = {}
    for key in datasets.keys():
        start_str, end_str = key.split("_")
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        date_lookup[key] = (start_date, end_date)
    return date_lookup

# Build lookup tables once
sst_date_lookup = _build_date_lookup(feature_datasets["sst"])
chl_date_lookup = _build_date_lookup(feature_datasets["chl"])
poc_date_lookup = _build_date_lookup(feature_datasets["poc"])

def _find_dataset_for_date(date, datasets, date_lookup):
    """Find the appropriate dataset for a given date"""
    for key, (start_date, end_date) in date_lookup.items():
        if start_date <= date <= end_date:
            return datasets[key]
    return None

def env_at_row(r, datasets, var_name, date_lookup):
    date = pd.to_datetime(r["date"])
    ds = _find_dataset_for_date(date, datasets, date_lookup)
    
    if ds is not None:
        val = ds[var_name].interp(
            lat=xr.DataArray([r["lat"]], dims="z"),
            lon=xr.DataArray([r["lon"]], dims="z")
        ).values[0]
        return val
    return None

def chl_at_row(r):
    return env_at_row(r, feature_datasets["chl"], "chlor_a", chl_date_lookup)

def sst_at_row(r):
    return env_at_row(r, feature_datasets["sst"], "sst", sst_date_lookup)

def poc_at_row(r):
    return env_at_row(r, feature_datasets["poc"], "poc", poc_date_lookup)

def make_background_points(n_bg, p_df):
    # Get the lat/lon range from the presence data
    min_lat = np.deg2rad(p_df["lat"].min())
    max_lat = np.deg2rad(p_df["lat"].max())
    min_lon = np.deg2rad(p_df["lon"].min())
    max_lon = np.deg2rad(p_df["lon"].max())

    # Get the date range from the presence data
    min_date = pd.to_datetime(p_df["date"]).min()
    max_date = pd.to_datetime(p_df["date"]).max()

    # Convert to Cartesian for center calculation
    lats_rad = np.deg2rad(p_df["lat"].values)
    lons_rad = np.deg2rad(p_df["lon"].values)
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    x_mean = x.mean()
    y_mean = y.mean()
    z_mean = z.mean()
    center_lat = np.arctan2(z_mean, np.sqrt(x_mean**2 + y_mean**2))
    center_lon = np.arctan2(y_mean, x_mean)

    # Compute max angular distance from center to any point (for bounding circle)
    angular_distances = np.arccos(
        np.sin(center_lat) * np.sin(lats_rad) +
        np.cos(center_lat) * np.cos(lats_rad) * np.cos(lons_rad - center_lon)
    )
    max_ang_dist = angular_distances.max()

    # Generate random points in polar (spherical cap) coordinates
    u = np.random.uniform(0, 1, n_bg)
    v = np.random.uniform(0, 1, n_bg)
    # Uniformly sample cos(angular distance) for uniform area on sphere
    cos_c = 1 - u * (1 - np.cos(max_ang_dist))
    c = np.arccos(cos_c)
    theta = 2 * np.pi * v

    # Spherical law of cosines to get lat/lon
    sin_center_lat = np.sin(center_lat)
    cos_center_lat = np.cos(center_lat)
    lat_bg = np.arcsin(
        sin_center_lat * np.cos(c) +
        cos_center_lat * np.sin(c) * np.cos(theta)
    )
    lon_bg = center_lon + np.arctan2(
        np.sin(theta) * np.sin(c) * cos_center_lat,
        np.cos(c) - sin_center_lat * np.sin(lat_bg)
    )

    # Convert back to degrees
    random_lats = np.rad2deg(lat_bg)
    random_lons = np.rad2deg(lon_bg)
    # Normalize longitude to [-180, 180]
    random_lons = (random_lons + 180) % 360 - 180

    # Generate random dates
    random_dates = np.random.choice(p_df["date"].values, n_bg, replace=True)

    # Create background DataFrame
    bg_df = pd.DataFrame({
        'id': 'fake',
        'date': random_dates,
        'lc': 0,
        'lon': random_lons,
        'lat': random_lats
    })

    return bg_df

def preprocess_df_vectorized(df):
    """Vectorized preprocessing for much better performance"""
    # Convert dates once
    dates = pd.to_datetime(df["date"])
    df["month"] = dates.dt.month
    df["sin_m"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_m"] = np.cos(2*np.pi*df["month"]/12)
    
    # Initialize arrays
    chl_values = np.full(len(df), np.nan)
    sst_values = np.full(len(df), np.nan)
    poc_values = np.full(len(df), np.nan)

    # Process each dataset once for all matching dates
    for key, ds in feature_datasets["sst"].items():
        start_date, end_date = sst_date_lookup[key]
        mask = (dates >= start_date) & (dates <= end_date)
        
        if mask.any():
            # Get all coordinates for this date range
            lats = df.loc[mask, "lat"].values
            lons = df.loc[mask, "lon"].values
            
            # Vectorized interpolation
            sst_interp = ds["sst"].interp(
                lat=xr.DataArray(lats, dims="points"),
                lon=xr.DataArray(lons, dims="points")
            )
            sst_values[mask] = sst_interp.values
    
    for key, ds in feature_datasets["chl"].items():
        start_date, end_date = chl_date_lookup[key]
        mask = (dates >= start_date) & (dates <= end_date)
        
        if mask.any():
            # Get all coordinates for this date range
            lats = df.loc[mask, "lat"].values
            lons = df.loc[mask, "lon"].values
            chl_interp = ds["chlor_a"].interp(
                lat=xr.DataArray(lats, dims="points"),
                lon=xr.DataArray(lons, dims="points")
            )
            chl_values[mask] = chl_interp.values
    
    for key, ds in feature_datasets["poc"].items():
        start_date, end_date = poc_date_lookup[key]
        mask = (dates >= start_date) & (dates <= end_date)
        
        if mask.any():
            lats = df.loc[mask, "lat"].values
            lons = df.loc[mask, "lon"].values
            poc_interp = ds["poc"].interp(
                lat=xr.DataArray(lats, dims="points"),
                lon=xr.DataArray(lons, dims="points")
            )
            poc_values[mask] = poc_interp.values
    
    df['chl'] = chl_values
    df['sst'] = sst_values
    df['poc'] = poc_values
    return df

def preprocess_df(df):
    """Wrapper for backward compatibility - uses vectorized version"""
    return preprocess_df_vectorized(df)

def normalize_df(df):
    df['poc'] = df['poc'] ** 0.5
    return df

p_df = pd.DataFrame()
for folder in folders:
    for file in os.listdir(folder):
        if folder == "../data/blue-shark-gps":
            if not file.endswith(".csv") or not file.startswith("filtered_160424_"):
                continue
        df = pd.read_csv(os.path.join(folder, file))
        df["date"] = pd.to_datetime(df["date"], format='mixed')
        
        # Calculate speed for each point (skip first point as it has no previous point)
        df["speed"] = np.nan
        for i in range(1, len(df)):
            df.loc[i, "speed"] = calc_speed(
                df.loc[i-1, "lat"], df.loc[i-1, "lon"], df.loc[i-1, "date"],
                df.loc[i, "lat"], df.loc[i, "lon"], df.loc[i, "date"]
            )[2]
        
        # Filter out points with speed > threshold
        before_count = len(df)
        df = df[df["speed"].isna() | (df["speed"] <= SPEED_THRESHOLD)]
        after_count = len(df)
        print(f"File {file}: Removed {before_count - after_count} points with speed > {SPEED_THRESHOLD} km/h")
        
        p_df = pd.concat([p_df, df])

bg_df = make_background_points(n_bg=len(p_df)*5, p_df=p_df)
p_df = preprocess_df(p_df)
bg_df = preprocess_df(bg_df)

all_df = pd.concat([p_df, bg_df])
all_df = normalize_df(all_df)
all_df["date"] = pd.to_datetime(all_df["date"])
all_df.to_csv("./train_data/train_data.csv", index=False)