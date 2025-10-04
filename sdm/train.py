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
import xarray as xr
from rasterio.transform import from_origin
import shap

from utils import MONTHS_2016

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

###############################################################################
# 1) Load & prepare data
###############################################################################

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
    # Check for extreme values that might cause numerical issues
    if X[col].abs().max() > 1e6:
        print(f"    WARNING: {col} has very large values (max abs: {X[col].abs().max():.2e})")
    if X[col].std() > 1e3:
        print(f"    WARNING: {col} has high variance (std: {X[col].std():.2e})")
print(f"Target distribution: {y.value_counts().to_dict()}")

###############################################################################
# 2) Train model
###############################################################################

# Try different MaxEnt parameters to avoid numerical issues
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
print(f"  Inf count: {np.isinf(yp).sum()}")

# Check for NaN values before calculating AUC
if np.isnan(yp).any():
    print("WARNING: Predictions contain NaN values!")
    # Remove NaN predictions for AUC calculation
    valid_mask = ~np.isnan(yp)
    y_clean = y[valid_mask]
    yp_clean = yp[valid_mask]
    print(f"Using {len(yp_clean)} valid predictions out of {len(yp)} total")
    if len(yp_clean) > 0:
        print("Training AUC:", metrics.roc_auc_score(y_clean, yp_clean))
    else:
        print("ERROR: No valid predictions available for AUC calculation")
else:
    print("Training AUC:", metrics.roc_auc_score(y, yp))

###############################################################################
# 3) Feature Importance Analysis
###############################################################################
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# 3.1 Permutation Importance
print("\n1. Permutation Importance:")
print("-" * 30)

# Create a wrapper class for the MaxEnt model to work with sklearn's permutation_importance
class MaxEntWrapper:
    def __init__(self, model):
        self.model = model
        # Define classes for binary classification
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y):
        # The model is already fitted, so we just return self
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        # For binary classification, return probabilities for both classes
        predictions = self.model.predict(X)
        # Convert to 2D array with probabilities for class 0 and class 1
        proba = np.column_stack([1 - predictions, predictions])
        return proba
    
    def score(self, X, y):
        predictions = self.predict(X)
        # Handle NaN predictions
        if np.isnan(predictions).any():
            valid_mask = ~np.isnan(predictions)
            if valid_mask.sum() == 0:
                return 0.0
            return roc_auc_score(y[valid_mask], predictions[valid_mask])
        return roc_auc_score(y, predictions)

wrapped_model = MaxEntWrapper(model)

# Calculate permutation importance
perm_importance = permutation_importance(
    wrapped_model, X, y, 
    scoring='roc_auc', 
    n_repeats=10, 
    random_state=42,
    n_jobs=-1
)

# Display permutation importance results
perm_results = []
for i, feature in enumerate(FEATURES):
    importance = perm_importance.importances_mean[i]
    std = perm_importance.importances_std[i]
    perm_results.append((feature, importance, std))
    print(f"  {feature:>8}: {importance:8.6f} ± {std:8.6f}")

# Sort by importance
perm_results.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Ranked by importance:")
for i, (feature, importance, std) in enumerate(perm_results, 1):
    print(f"    {i}. {feature:>8}: {importance:8.6f} ± {std:8.6f}")

# 3.2 SHAP Values (if available)
print("\n2. SHAP Values:")
print("-" * 20)

try:
    # Create a small sample for SHAP analysis (to avoid memory issues)
    sample_size = min(1000, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]
    
    # Create a simple function wrapper for SHAP
    def model_predict(X_input):
        return model.predict(X_input)
    
    # Create SHAP explainer using the function
    explainer = shap.Explainer(model_predict, X_sample)
    shap_values = explainer(X_sample)
    
    # Calculate mean absolute SHAP values for feature importance
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    
    # Display SHAP importance results
    shap_results = []
    for i, feature in enumerate(FEATURES):
        importance = mean_shap_values[i]
        shap_results.append((feature, importance))
        print(f"  {feature:>8}: {importance:8.6f}")
    
    # Sort by importance
    shap_results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Ranked by importance:")
    for i, (feature, importance) in enumerate(shap_results, 1):
        print(f"    {i}. {feature:>8}: {importance:8.6f}")
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved SHAP summary plot → shap_feature_importance.png")
    
    # Create SHAP bar plot
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig("shap_feature_importance_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved SHAP bar plot → shap_feature_importance_bar.png")
    
except Exception as e:
    print(f"  Error calculating SHAP values: {e}")
    print("  SHAP analysis skipped. You may need to install shap: pip install shap")

# 3.3 Feature Importance Comparison Plot
print("\n3. Feature Importance Comparison:")
print("-" * 35)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Permutation importance plot
perm_features = [x[0] for x in perm_results]
perm_importances = [x[1] for x in perm_results]
perm_stds = [x[2] for x in perm_results]

ax1.barh(perm_features, perm_importances, xerr=perm_stds, capsize=5)
ax1.set_xlabel('Permutation Importance (AUC drop)')
ax1.set_title('Permutation Feature Importance')
ax1.grid(True, alpha=0.3)

# SHAP importance plot (if available)
if 'shap_results' in locals():
    shap_features = [x[0] for x in shap_results]
    shap_importances = [x[1] for x in shap_results]
    
    ax2.barh(shap_features, shap_importances)
    ax2.set_xlabel('Mean |SHAP value|')
    ax2.set_title('SHAP Feature Importance')
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'SHAP analysis\nnot available', 
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('SHAP Feature Importance')

plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved feature importance comparison → feature_importance_comparison.png")

# 3.4 Individual Feature Analysis
print("\n4. Individual Feature Analysis:")
print("-" * 30)

for feature in FEATURES:
    print(f"\n{feature.upper()}:")
    print(f"  Range: [{X[feature].min():.6f}, {X[feature].max():.6f}]")
    print(f"  Mean: {X[feature].mean():.6f}")
    print(f"  Std:  {X[feature].std():.6f}")
    
    # Correlation with target
    corr = X[feature].corr(y)
    print(f"  Correlation with target: {corr:.6f}")
    
    # Feature distribution by target class
    pos_values = X[y == 1][feature]
    neg_values = X[y == 0][feature]
    print(f"  Mean (positive class): {pos_values.mean():.6f}")
    print(f"  Mean (negative class): {neg_values.mean():.6f}")
    print(f"  Difference: {pos_values.mean() - neg_values.mean():.6f}")

print("\n" + "="*60)

###############################################################################
# 4) Calibration curve
###############################################################################
p_all = model.predict(X)
# Handle NaN values in predictions for calibration curve
if np.isnan(p_all).any():
    print("WARNING: Calibration curve predictions contain NaN values!")
    valid_mask = ~np.isnan(p_all)
    y_cal = y[valid_mask]
    p_cal = p_all[valid_mask]
    print(f"Using {len(p_cal)} valid predictions for calibration curve")
else:
    y_cal = y
    p_cal = p_all

frac_pos, mean_pred = calibration_curve(y_cal, p_cal, n_bins=10, strategy="quantile")

plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],"--",label="ideal")
plt.plot(mean_pred, frac_pos, marker="o", label="MaxEnt (cloglog)")
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.title("Calibration (quantile bins)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_CAL, dpi=120)
plt.close()
print(f"[OK] Saved calibration curve → {OUT_CAL}")

###############################################################################
# 5) Predict to raster grid
###############################################################################
def open_any(path: str) -> xr.Dataset:
    p = Path(path)
    return xr.open_dataset(p)

def interp_grid(ds: xr.Dataset, var: str, xs: np.ndarray, ys: np.ndarray, ts=None):
    da = ds[var]
    dai = da.interp(lat=("y", ys), lon=("x", xs))
    return np.asarray(dai.values)

for i in range(len(MONTHS_2016)):
    TS         = MONTHS_2016[i]["TS"]
    SST_FILE   = MONTHS_2016[i]["SST_FILE"]
    CHL_FILE   = MONTHS_2016[i]["CHL_FILE"]
    POC_FILE   = MONTHS_2016[i]["POC_FILE"]

    ds_sst = open_any(SST_FILE)
    ds_chl = open_any(CHL_FILE)
    ds_poc = open_any(POC_FILE)
    lat_min = 5
    lat_max = 50
    lon_min = -90
    lon_max = -30

    xs = np.arange(lon_min, lon_max + GRID_RES, GRID_RES)
    ys = np.arange(lat_min, lat_max + GRID_RES, GRID_RES)

    v_sst = VAR_SST if VAR_SST in ds_sst.data_vars else list(ds_sst.data_vars)[0]
    v_chl = VAR_CHL if VAR_CHL in ds_chl.data_vars else list(ds_chl.data_vars)[0]
    v_poc = VAR_POC if VAR_POC in ds_poc.data_vars else list(ds_poc.data_vars)[0]


    sst_grid = interp_grid(ds_sst, v_sst, xs, ys, TS)
    chl_grid = interp_grid(ds_chl, v_chl, xs, ys, TS)
    poc_grid = interp_grid(ds_poc, v_poc, xs, ys, TS)


    month = TS.month
    sin_m = np.sin(2*np.pi*month/12.0)
    cos_m = np.cos(2*np.pi*month/12.0)

    mesh_lon, mesh_lat = np.meshgrid(xs, ys)
    n = mesh_lon.size
    grid_df = pd.DataFrame({
        "sin_m": np.full(n, sin_m),
        "cos_m": np.full(n, cos_m),
        "sst": sst_grid.ravel().astype(float),
        "chl": chl_grid.ravel().astype(float),
        "poc": poc_grid.ravel().astype(float) ** 0.5,
    })

    valid = np.isfinite(grid_df["sst"].values) & np.isfinite(grid_df["chl"].values) & np.isfinite(grid_df["poc"].values)
    probs = np.full(n, np.nan, dtype=float)
    probs[valid] = model.predict(grid_df.loc[valid, FEATURES])
    print(grid_df.loc[valid, FEATURES])
    prob_grid = probs.reshape(mesh_lon.shape)

    print("Grid size:", len(xs), len(ys))

    # === Save heatmap PNG ===
    plt.figure(figsize=(7,6))
    im = plt.imshow(prob_grid, extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                    origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, label="P(foraging habitat)")
    plt.title(f"Foraging probability — {TS.date()}")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"foraging_prob_2016_{month:02d}.png", dpi=150)
    plt.close()
    print(f"[OK] Saved probability heatmap PNG → foraging_prob.png")
