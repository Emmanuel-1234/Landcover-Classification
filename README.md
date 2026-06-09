# Land Cover Classification Using Sentinel-2 Satellite Imagery

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-green)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive geospatial analysis project for automated land cover classification using Sentinel-2 satellite imagery and machine learning-optimized spectral indices.

## Project Overview

This project implements an advanced land cover classification system that leverages Google Earth Engine (GEE) and Sentinel-2 satellite data to identify and quantify different land cover types including vegetation, water bodies, built-up areas, and bare land. The study area focuses on the Vijayawada region in Andhra Pradesh, India, analyzing imagery from 2021.

The project demonstrates the application of remote sensing, geospatial analysis, and machine learning optimization techniques to solve real-world environmental monitoring challenges.

##  Key Features

- **Multi-spectral Analysis**: Utilizes all 13 Sentinel-2 bands for comprehensive land cover detection
- **Optimized Spectral Indices**: Machine learning-based optimization of BSI (Bare Soil Index) and NBI (New Built-up Index) using F1-score maximization
- **Multiple Vegetation Indices**: Implements NDVI and EVI for accurate vegetation mapping
- **Water Body Detection**: Combines NDVI, NDWI, and MNDWI for precise water body identification
- **Interactive Visualization**: Real-time band selection and reflectance sampling using geemap
- **Area Quantification**: Automated calculation of land cover areas in km²
- **Cloud Filtering**: Intelligent cloud coverage filtering (<5%) for optimal image quality

##  Spectral Indices Implemented

### Vegetation Indices
- **NDVI** (Normalized Difference Vegetation Index): `(NIR - Red) / (NIR + Red)`
  - Detects vegetation health and density
  - Threshold: > 0.3 for vegetation classification

- **EVI** (Enhanced Vegetation Index): `2.5 × ((NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1))`
  - Improved sensitivity in high biomass regions
  - Reduced atmospheric influence

### Water Indices
- **NDWI** (Normalized Difference Water Index): `(Green - NIR) / (Green + NIR)`
  - Delineates open water features
  - Threshold: > 0.2 for water classification

- **MNDWI** (Modified NDWI): `(Green - SWIR) / (Green + SWIR)`
  - Enhanced water detection with reduced urban confusion

### Built-up Area Indices
- **NDBI** (Normalized Difference Built-up Index): `(SWIR - NIR) / (SWIR + NIR)`
  - Standard built-up area detection

- **NBI** (New Built-up Index): Optimized formula with 5 machine-learned coefficients
  - Enhanced urban area detection through ML optimization

### Bare Soil Index
- **BSI** (Bare Soil Index): Optimized formula with 9 coefficients
  - F1-score maximized for bare land classification

##  Technologies Used

- **Python 3.8+**
- **Google Earth Engine (GEE)**: Cloud-based satellite imagery processing
- **geemap**: Interactive geospatial visualization and analysis
- **NumPy & Pandas**: Data manipulation and numerical operations
- **Matplotlib**: Visualization and plotting
- **scikit-learn**: Machine learning for index optimization
- **SciPy**: Numerical optimization (Nelder-Mead algorithm)
- **rasterio & GDAL**: Geospatial raster data I/O

##  Installation

### Prerequisites
- Python 3.8 or higher
- Google Colab account (recommended) or local Jupyter environment
- Google Earth Engine account ([Sign up here](https://earthengine.google.com/signup/))

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/land-cover-classification.git
cd land-cover-classification
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

3. **Authenticate Google Earth Engine**:
```python
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

For Google Colab users, authentication will prompt a browser-based login.

##  Usage

### Quick Start

#### 1. Define Region of Interest (ROI)
```python
# Vijayawada region coordinates
lon1, lat1 = 80.5856, 16.4726
lon2, lat2 = 80.6385, 16.5151
geometry = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])
```

#### 2. Load Sentinel-2 Imagery
```python
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate('2021-01-01', '2021-12-31')
    .filterBounds(geometry)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
)

# Select least cloudy image
image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
```

#### 3. Calculate Spectral Indices
```python
# Vegetation
ndvi = image.normalizedDifference(['B8', 'B4'])

# Water
ndwi = image.normalizedDifference(['B3', 'B8'])
mndwi = image.normalizedDifference(['B3', 'B11'])

# Built-up
ndbi = image.normalizedDifference(['B11', 'B8'])
```

#### 4. Classify and Quantify Land Cover
```python
# Create masks
veg_mask = ndvi.gt(0.3)
water_mask = ndvi.lt(0.2).And(ndwi.gt(0.2)).And(mndwi.gt(0.2))
buildup_mask = ndbi.gt(0.1)

# Calculate areas
veg_area = calculate_area(veg_mask, geometry)
water_area = calculate_area(water_mask, geometry)
```

### Interactive Map Visualization

```python
import geemap

# Create interactive map
Map = geemap.Map()

# Add layers
Map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'max': 3000}, 'True Color')
Map.addLayer(ndvi, {'min': -1, 'max': 1, 'palette': ['red', 'yellow', 'green']}, 'NDVI')
Map.addLayer(water_mask, {'palette': ['blue']}, 'Water Bodies')

# Center on ROI
Map.centerObject(geometry, 12)
Map
```

##  Project Structure

```
land-cover-classification/
│
├── notebooks/
│   ├── land_cover_analysis.ipynb      # Main analysis notebook
│   ├── index_optimization.ipynb        # ML optimization experiments
│   └── visualization.ipynb             # Results visualization
│
├── data/
│   ├── training_pixels.csv            # ESA WorldCover training data
│   └── README.md                       # Data description
│
├── results/
│   ├── figures/                        # Output visualizations
│   │   ├── ndvi_map.png
│   │   ├── water_bodies.png
│   │   └── classification_results.png
│   └── statistics/                     # Area calculation results
│       └── landcover_areas.csv
│
├── docs/
│   ├── METHODOLOGY.md                  # Detailed methodology
│   └── REFERENCES.md                   # Research references
│
├── src/                                # Source code (optional)
│   ├── __init__.py
│   ├── indices.py                      # Spectral index functions
│   └── utils.py                        # Utility functions
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
├── CITATION.cff                        # Citation metadata
├── CONTRIBUTING.md                     # Contribution guidelines
├── CHANGELOG.md                        # Version history
└── README.md                           # This file
```

##  Results

### Study Area: Vijayawada Region (2021)

| Land Cover Type | Area (km²) | Percentage | Method |
|----------------|-----------|-----------|---------|
| Vegetation     | 10.93     | 49.0%     | NDVI > 0.3 |
| Water Bodies   | 3.68      | 16.5%     | NDVI < 0.2, NDWI > 0.2, MNDWI > 0.2 |
| Built-up Area  | 18.78     | 84.2%     | Optimized NBI, NDBI > 0.1 |
| Bare Land      | 16.47     | 73.8%     | Optimized BSI > 0.15 |

*Note: Overlapping percentages indicate mixed pixel classifications due to 10m resolution*

### Key Findings

 **Optimization Success**: Machine learning-based coefficient optimization improved F1-scores for BSI and NBI indices

 **Water Detection**: Multi-index approach (NDVI + NDWI + MNDWI) provided robust water body delineation, minimizing shadow confusion

 **Urban Mapping**: 10m spatial resolution enabled detailed urban area mapping, capturing road networks and building patterns

 **Vegetation Health**: NDVI and EVI combination revealed varying vegetation density across the study area

##  Methodology

### Index Optimization Process

The project employs **Nelder-Mead optimization** to derive optimal coefficients for BSI and NBI:

1. **Training Data**: Load ESA WorldCover labeled Sentinel-2 pixels
2. **Normalization**: Scale reflectance values to 0-1 range
3. **Index Formula**: Define parameterized index formula
4. **Optimization**: Maximize F1-score using Nelder-Mead algorithm
5. **Validation**: Test on independent validation dataset

**Optimized BSI Coefficients**:
```python
[1.1381, 1.0771, 0.7362, 0.9990, 0.8962, 1.1294, 0.9680, 0.9962, 1.1258]
```

**Optimized NBI Coefficients**:
```python
[1.1533, 1.2060, 0.7210, 0.9611, 0.9549]
```

### Classification Workflow

```
Sentinel-2 Image → Cloud Filter → Band Selection → Index Calculation 
→ Threshold Application → Mask Creation → Area Quantification → Visualization
```

##  Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| **Cloud Coverage** | Strict filter (<5%) + temporal compositing |
| **Mixed Pixels in Urban Areas** | Multiple complementary indices (BSI + NBI + NDBI) |
| **Water-Shadow Confusion** | Combined NDWI + MNDWI thresholding |
| **Index Parameter Tuning** | ML-based optimization using F1-score |
| **Computational Load** | Google Earth Engine cloud processing |

##  Future Enhancements

- [ ] **Random Forest Classifier**: Implement supervised ML classification
- [ ] **Time-Series Analysis**: Multi-temporal change detection (2020-2025)
- [ ] **DEM Integration**: Topographic correction using digital elevation models
- [ ] **Web Application**: Deploy interactive Streamlit/Flask web app
- [ ] **Multi-Platform Support**: Integrate Landsat and MODIS imagery
- [ ] **Automated Reporting**: Generate PDF reports with statistics and maps
- [ ] **Accuracy Assessment**: Ground truth validation with confusion matrix
- [ ] **Deep Learning**: Implement U-Net for semantic segmentation

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add: brief description'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Google Earth Engine** for providing satellite imagery and cloud computing infrastructure
- **Copernicus Sentinel-2** mission for open-access multispectral data
- **ESA WorldCover** project for training data labels
- **geemap** community for excellent geospatial visualization tools

## 📞 Contact

**K. Emmanuel**

- 📧 Email: emmanuelkaza1234@gmail.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/emmanuel-kaza-4825a41a7/)
- 🐙 GitHub: [@emmanuel](https://github.com/Emmanuel-1234)

**Project Link**: [https://github.com/yourusername/land-cover-classification](https://github.com/Emmanuel-1234/land-cover-classification)

##  References

1. Zha, Y., Gao, J., & Ni, S. (2003). Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. *International Journal of Remote Sensing*, 24(3), 583-594.

2. McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425-1432.

3. Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing*, 27(14), 3025-3033.

4. Huete, A., et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment*, 83(1-2), 195-213.

5. Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18-27.

---

⭐ **Star this repository** if you found it helpful!

📢 **Share with others** interested in remote sensing and geospatial analysis!
