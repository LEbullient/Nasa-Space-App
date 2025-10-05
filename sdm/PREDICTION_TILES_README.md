# Foraging Habitat Prediction Tiles

## Overview

This document describes the foraging habitat prediction tiles generated for blue sharks from July 2013 to January 2017.

## Script

The script `train_predict_tiles.py` generates:
1. A MaxEnt species distribution model (SDM) trained on blue shark tracking data
2. Monthly foraging habitat probability predictions
3. Map tiles in Web Mercator projection for visualization in Leaflet

## Output Structure

```
tiles/predict/
├── 2013-07-01/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
├── 2013-08-01/
│   └── ...
└── 2017-01-01/
    └── ...
```

Each month directory contains zoom levels 0-5, following the standard Leaflet tile structure:
- `{date}/{z}/{x}/{y}.png`
- Example: `2016-01-01/3/2/3.png`

## Tile Specifications

- **Format**: PNG with transparency (RGBA)
- **Size**: 256x256 pixels
- **Projection**: Web Mercator (EPSG:3857)
- **Zoom levels**: 0-5
- **Coverage**: Global (90°S-90°N, 180°W-180°E)

## Color Scheme

The tiles use a **green-to-red** color gradient:
- **Green** (#00ff00): Low foraging probability (0.0)
- **Yellow** (#ffff00): Medium foraging probability (0.5)
- **Red** (#ff0000): High foraging probability (1.0)
- **Transparent**: No data or land areas

## Model Details

### Training Data
- **Features**: 
  - `sin_m`, `cos_m`: Seasonal cyclical encoding (month)
  - `sst`: Sea surface temperature (°C)
  - `chl`: Chlorophyll-a concentration (mg/m³)
  - `poc`: Particulate organic carbon (mg/m³, square root transformed)
- **Training samples**: 47,242 (after NaN removal)
- **Training AUC**: 0.942

### Prediction Grid
- **Resolution**: 0.1° (~11 km at equator)
- **Extent**: 90°S-90°N, 180°W-180°E (Global)
- **Grid size**: 1,801 × 3,601 = 6,485,401 cells per month

## Time Coverage

**43 months** from July 2013 to January 2017:
- 2013: July - December (6 months)
- 2014: January - December (12 months)
- 2015: January - December (12 months)
- 2016: January - December (12 months)
- 2017: January (1 month)

## Usage in Leaflet

To use these tiles in a Leaflet map:

```javascript
// Define tile layer
const foragingLayer = L.tileLayer('tiles/predict/{date}/{z}/{x}/{y}.png', {
    tms: false,
    opacity: 0.7,
    attribution: 'Foraging habitat predictions',
    minZoom: 0,
    maxZoom: 5
});

// Add to map
foragingLayer.addTo(map);
```

To switch between different months, update the `{date}` parameter:
```javascript
function updateFoagingMonth(dateStr) {
    // dateStr format: "YYYY-MM-01" (e.g., "2016-01-01")
    const newUrl = 'tiles/predict/' + dateStr + '/{z}/{x}/{y}.png';
    foragingLayer.setUrl(newUrl);
}
```

## Data Sources

### Environmental Data
- **Source**: AQUA MODIS Level-3 Mapped Ocean Color Data
- **Variables**: SST, Chlorophyll-a, POC
- **Resolution**: 4km
- **Temporal**: Monthly composites

### Shark Tracking Data
- **Source**: Blue shark GPS tracking data (filtered)
- **Period**: 2013-2017
- **Files**: 17 filtered CSV files
- **Total observations**: 19,566 data points

## Processing Time

- **Model training**: ~2 seconds
- **Tile generation**: ~18 minutes for 43 months (global coverage)
- **Total tiles created**: ~30,850 tiles (variable by month due to land masking)

## Notes

- Tiles are only generated for ocean areas with valid environmental data
- Land areas and missing data regions are fully transparent
- The Web Mercator projection properly handles latitude distortion
- Maximum zoom level is limited to 5 to balance detail and file size
