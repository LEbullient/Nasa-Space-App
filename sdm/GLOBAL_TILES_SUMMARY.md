# Global Foraging Habitat Prediction Tiles - Summary

## Completed: October 5, 2025

Successfully generated **global foraging habitat prediction tiles** for blue sharks covering the entire world from July 2013 to January 2017.

## What Changed

### Previous Version (Regional)
- Coverage: North Atlantic only (5°N-50°N, 90°W-30°W)
- Grid size: 451 × 601 = 271,051 cells
- Total tiles: ~2,150
- Processing time: ~2 minutes

### Current Version (Global)
- Coverage: **Entire world** (90°S-90°N, 180°W-180°E)
- Grid size: 1,801 × 3,601 = 6,485,401 cells
- Total tiles: **30,849** tiles
- Processing time: **18 minutes**

## Verification

Tested tiles exist in multiple ocean regions:
- ✓ North Atlantic (40°N, 50°W)
- ✓ Pacific Ocean (30°N, 150°W)
- ✓ Indian Ocean (10°S, 80°E)
- ✓ South Atlantic (30°S, 20°W)
- ✓ Mediterranean Sea (35°N, 15°E)

## Sample Predictions (2016-01-01)

Average foraging habitat suitability by region:
- **North Atlantic**: 0.303 ± 0.221 (highest - matches training data region)
- **Pacific Ocean**: 0.190 ± 0.084 (moderate - blue sharks present)
- **Indian Ocean**: 0.003 ± 0.007 (lowest - less common habitat)

## Technical Details

### Model Performance
- Training AUC: **0.942** (excellent discrimination)
- Features: SST, Chlorophyll-a, POC, seasonal encoding
- Training samples: 47,242 observations
- Target: Blue shark foraging locations vs. pseudo-absences

### Tile Structure
```
tiles/predict/
├── 2013-07-01/ through 2017-01-01/
│   ├── 0/ (1 tile - world overview)
│   ├── 1/ (4 tiles)
│   ├── 2/ (12 tiles)
│   ├── 3/ (40 tiles)
│   ├── 4/ (135 tiles)
│   └── 5/ (492 tiles per month)
```

### Color Scheme
- 🟢 **Green** → Low foraging probability (0.0)
- 🟡 **Yellow** → Medium probability (0.5)
- 🔴 **Red** → High probability (1.0)

## Files Modified

1. **`train_predict_tiles.py`**
   - Changed grid extent from regional to global
   - Updated from: `lat_min=5, lat_max=50, lon_min=-90, lon_max=-30`
   - Updated to: `lat_min=-90, lat_max=90, lon_min=-180, lon_max=180`

2. **`PREDICTION_TILES_README.md`**
   - Updated coverage specifications
   - Updated grid size and tile counts
   - Updated processing time estimates

## Usage

Load in Leaflet with global coverage:

```javascript
const foragingLayer = L.tileLayer('tiles/predict/{date}/{z}/{x}/{y}.png', {
    opacity: 0.7,
    attribution: 'Blue shark foraging habitat predictions',
    minZoom: 0,
    maxZoom: 5
});

// Example: Load January 2016 predictions
foragingLayer.setUrl('tiles/predict/2016-01-01/{z}/{x}/{y}.png');
foragingLayer.addTo(map);
```

## Notes

- The model was trained on North Atlantic blue shark data, so predictions are most reliable in that region
- Predictions in other oceans extrapolate based on similar environmental conditions
- Transparent areas indicate land or missing environmental data
- Web Mercator projection properly handles global display in web maps
- Tiles are optimized with land masking to reduce file count and size
