# Methodology

## Overview

This document provides a detailed explanation of the methodologies employed in the Land Cover Classification project using Sentinel-2 satellite imagery and Google Earth Engine. The project combines remote sensing, spectral analysis, and machine learning optimization to classify land cover types in the Vijayawada region of Andhra Pradesh, India.

## Table of Contents

1. [Data Source and Acquisition](#data-source-and-acquisition)
2. [Study Area](#study-area)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Spectral Indices](#spectral-indices)
5. [Machine Learning Optimization](#machine-learning-optimization)
6. [Classification Workflow](#classification-workflow)
7. [Area Calculation](#area-calculation)
8. [Visualization Techniques](#visualization-techniques)
9. [Validation and Accuracy](#validation-and-accuracy)
10. [Limitations and Challenges](#limitations-and-challenges)

---

## Data Source and Acquisition

### Sentinel-2 Satellite Imagery

**Mission Details:**
- **Platform**: Copernicus Sentinel-2A and Sentinel-2B
- **Operator**: European Space Agency (ESA)
- **Collection Used**: `COPERNICUS/S2_SR_HARMONIZED`
- **Product Type**: Surface Reflectance (atmospherically corrected)
- **Temporal Resolution**: 5-10 days (combined constellation)
- **Swath Width**: 290 km

**Spectral Bands Configuration:**

| Band | Name | Wavelength (nm) | Resolution (m) | Purpose |
|------|------|----------------|----------------|---------|
| B1 | Coastal Aerosol | 443 | 60 | Atmospheric correction |
| B2 | Blue | 490 | 10 | Visible spectrum, water detection |
| B3 | Green | 560 | 10 | Vegetation health, water bodies |
| B4 | Red | 665 | 10 | Chlorophyll absorption, vegetation |
| B5 | Red Edge 1 | 705 | 20 | Vegetation stress |
| B6 | Red Edge 2 | 740 | 20 | Vegetation classification |
| B7 | Red Edge 3 | 783 | 20 | Vegetation monitoring |
| B8 | NIR | 842 | 10 | Biomass, vegetation structure |
| B8A | Narrow NIR | 865 | 20 | Water vapor reference |
| B9 | Water Vapor | 945 | 60 | Atmospheric correction |
| B10 | SWIR-Cirrus | 1375 | 60 | Cirrus cloud detection |
| B11 | SWIR 1 | 1610 | 20 | Moisture content, soil/vegetation |
| B12 | SWIR 2 | 2190 | 20 | Moisture content, geology |

**Key Bands Used in This Study:**
- **10m Resolution**: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
- **20m Resolution**: B11 (SWIR1), B12 (SWIR2)

### Data Collection Parameters

```python
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate('2021-01-01', '2021-12-31')
    .filterBounds(geometry)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
)
```

**Filtering Criteria:**
1. **Temporal Filter**: January 1, 2021 - December 31, 2021
2. **Spatial Filter**: Bounding box around Vijayawada region
3. **Cloud Filter**: Only images with < 5% cloud coverage
4. **Image Selection**: Least cloudy image from the filtered collection

**Rationale for 2021 Data:**
- Represents recent land cover conditions
- Minimal cloud interference during dry season
- Consistent with ESA WorldCover training data temporal alignment

---

## Study Area

### Geographic Location

**Region**: Vijayawada, Andhra Pradesh, India

**Coordinates:**
- **Southwest Corner**: 80.5856°E, 16.4726°N
- **Northeast Corner**: 80.6385°E, 16.5151°N
- **Center Point**: Approximately 80.6121°E, 16.4939°N

**Area Coverage**: ~22.3 km² (rectangular bounding box)

### Regional Characteristics

**Topography:**
- Relatively flat terrain with gentle slopes
- Part of Krishna River delta region
- Elevation: 20-40 meters above sea level

**Land Cover Types:**
1. **Vegetation**: Agricultural lands, urban green spaces, scattered tree cover
2. **Water Bodies**: Krishna River, irrigation canals, small tanks
3. **Built-up Area**: Residential, commercial, industrial zones, roads
4. **Bare Land**: Construction sites, vacant lots, exposed soil

**Climate:**
- **Type**: Tropical wet and dry (Köppen: Aw)
- **Temperature**: 25-35°C (annual average)
- **Rainfall**: 1000-1200 mm annually (monsoon-dominated)
- **Seasons**: Summer (March-May), Monsoon (June-September), Winter (October-February)

**Significance:**
- Rapidly urbanizing city with population > 1 million
- Important agricultural hub in Krishna delta
- Strategic location on National Highway 16
- Environmental monitoring needs due to urban expansion

### Region of Interest (ROI) Definition

```python
lon1, lat1 = 80.58563253572738, 16.472595829903636
lon2, lat2 = 80.63850423982895, 16.5150620008198
geometry = ee.Geometry.Rectangle([lon1, lat1, lon2, lat2])
```

---

## Preprocessing Pipeline

### 1. Image Collection and Filtering

**Step 1: Load Collection**
```python
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
```
- Harmonized collection ensures consistency between S2A and S2B
- Surface reflectance product (atmospherically corrected)

**Step 2: Temporal Filtering**
```python
.filterDate('2021-01-01', '2021-12-31')
```
- Full calendar year for comprehensive seasonal coverage
- 2021 selected for recent data with minimal cloud issues

**Step 3: Spatial Filtering**
```python
.filterBounds(geometry)
```
- Restricts search to images intersecting the ROI
- Reduces processing time and data volume

**Step 4: Cloud Filtering**
```python
.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
```
- Strict cloud threshold ensures high-quality imagery
- Cloud metadata from Sentinel-2 L1C processing

**Step 5: Image Selection**
```python
image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
```
- Selects the single best (least cloudy) image
- Alternative: Median composite for multi-temporal analysis

### 2. Band Normalization

**Sentinel-2 Scale Factor:**
- Raw DN values range: 0-10000
- Represents reflectance × 10000

**Normalization Formula:**
```python
normalized_reflectance = band_value / 10000
```
- Converts to actual reflectance (0-1 range)
- Required for accurate index calculations

### 3. Quality Assessment

**Visual Inspection:**
- Check for cloud shadows, haze, scan line errors
- Verify coverage over entire ROI

**Metadata Verification:**
- Acquisition date
- Sun azimuth/elevation angles
- Processing baseline version

---

## Spectral Indices

### Vegetation Indices

#### 1. NDVI (Normalized Difference Vegetation Index)

**Formula:**
\[
NDVI = \frac{NIR - Red}{NIR + Red} = \frac{B8 - B4}{B8 + B4}
\]

**Physical Basis:**
- Healthy vegetation: High NIR reflectance (cellular structure), Low Red reflectance (chlorophyll absorption)
- Non-vegetation: Similar NIR and Red reflectance

**Implementation:**
```python
ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
```

**Value Range and Interpretation:**

| NDVI Range | Land Cover Type | Description |
|------------|----------------|-------------|
| < -0.1 | Water | Strong water absorption in NIR |
| -0.1 to 0.1 | Bare soil, sand | Minimal vegetation |
| 0.1 to 0.3 | Sparse vegetation | Grasslands, shrubs |
| 0.3 to 0.6 | Moderate vegetation | Crops, mixed vegetation |
| > 0.6 | Dense vegetation | Forests, healthy crops |

**Classification Threshold:**
```python
vegetation_mask = ndvi.gt(0.3)
```

**Applications:**
- Crop health monitoring
- Biomass estimation
- Deforestation detection
- Agricultural yield prediction

#### 2. EVI (Enhanced Vegetation Index)

**Formula:**
\[
EVI = 2.5 \times \frac{NIR - Red}{NIR + 6 \times Red - 7.5 \times Blue + 1}
\]

\[
EVI = 2.5 \times \frac{B8 - B4}{B8 + 6 \times B4 - 7.5 \times B2 + 1}
\]

**Advantages over NDVI:**
1. **Reduced Atmospheric Influence**: Blue band corrects for aerosol scattering
2. **Reduced Soil Background**: Coefficient adjustments minimize soil noise
3. **Improved Sensitivity in Dense Canopy**: Less saturation in high biomass areas

**Implementation:**
```python
evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'BLUE': image.select('B2')
    }
).rename('EVI')
```

**Use Case:**
- Preferred for tropical regions with dense vegetation
- Better for monitoring vegetation in areas with high aerosol content

---

### Water Indices

#### 3. NDWI (Normalized Difference Water Index)

**Formula:**
\[
NDWI = \frac{Green - NIR}{Green + NIR} = \frac{B3 - B8}{B3 + B8}
\]

**Physical Basis:**
- Water: High Green reflectance, Very low NIR reflectance
- Vegetation: Low Green, High NIR (opposite of water)

**Implementation:**
```python
ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
```

**Classification Threshold:**
```python
water_mask = ndwi.gt(0.2)
```

**Value Interpretation:**

| NDWI Range | Surface Type |
|------------|-------------|
| > 0.3 | Water bodies (high confidence) |
| 0.0 to 0.3 | Wet surfaces, moisture |
| < 0.0 | Dry land, vegetation |

#### 4. MNDWI (Modified Normalized Difference Water Index)

**Formula:**
\[
MNDWI = \frac{Green - SWIR}{Green + SWIR} = \frac{B3 - B11}{B3 + B11}
\]

**Key Improvement:**
- Replaces NIR with SWIR band
- SWIR strongly absorbed by water but reflected by built-up areas
- **Result**: Better separation of water from urban features

**Implementation:**
```python
mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
```

**Classification Threshold:**
```python
water_enhanced = mndwi.gt(0.2)
```

**Why MNDWI > NDWI:**
- Reduces false positives in urban areas
- Better for mapping water in built environments
- Improved shadow discrimination

#### Combined Water Detection Strategy

**Multi-Index Approach:**
```python
water_mask = (ndvi.lt(0.2)
              .And(ndwi.gt(0.2))
              .And(mndwi.gt(0.2)))
```

**Logic:**
1. NDVI < 0.2: Eliminates vegetation
2. NDWI > 0.2: Identifies water-like features
3. MNDWI > 0.2: Confirms water, removes built-up confusion

**Result**: Highly accurate water body delineation

---

### Built-up Area Indices

#### 5. NDBI (Normalized Difference Built-up Index)

**Formula:**
\[
NDBI = \frac{SWIR - NIR}{SWIR + NIR} = \frac{B11 - B8}{B11 + B8}
\]

**Physical Basis:**
- Built-up areas: High SWIR reflectance (concrete, asphalt), Moderate NIR
- Vegetation: High NIR, Low SWIR (opposite pattern)

**Implementation:**
```python
ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
```

**Classification Threshold:**
```python
buildup_mask = ndbi.gt(0.1)
```

**Applications:**
- Urban extent mapping
- Impervious surface estimation
- Urbanization monitoring

#### 6. NBI (New Built-up Index) - Optimized

**Concept:**
- Custom index with machine-learned coefficients
- Optimized using ESA WorldCover labeled training data
- Maximizes F1-score for built-up class

**General Formula:**
\[
NBI = \frac{c_1 \times SWIR1 \times c_2 \times Red}{c_3 \times NIR \times c_4 \times Blue + c_5}
\]

**Optimized Coefficients (via Nelder-Mead):**
- c₁ = 1.1533
- c₂ = 1.2060
- c₃ = 0.7210
- c₄ = 0.9611
- c₅ = 0.9549

**Implementation:**
```python
nbi = image.expression(
    '(c1 * SWIR * c2 * RED) / (c3 * NIR * c4 * BLUE + c5)',
    {
        'SWIR': image.select('B11'),
        'RED': image.select('B4'),
        'NIR': image.select('B8'),
        'BLUE': image.select('B2'),
        'c1': 1.1533, 'c2': 1.2060,
        'c3': 0.7210, 'c4': 0.9611, 'c5': 0.9549
    }
).rename('NBI')
```

**Performance:**
- F1-score: ~0.85 (significantly better than standard NDBI)
- Reduced confusion with bare soil

---

### Bare Soil Index

#### 7. BSI (Bare Soil Index) - Optimized

**Concept:**
- Complex multi-band index with 9 optimized coefficients
- Combines Blue, Green, Red, NIR, SWIR1, SWIR2 bands
- Optimized specifically for bare soil vs. all other classes

**General Formula Structure:**
\[
BSI = f(c_1 \cdot B2, c_2 \cdot B3, c_3 \cdot B4, c_4 \cdot B8, c_5 \cdot B11, c_6 \cdot B12, ...)
\]

**Optimized Coefficients:**
```
[1.1381, 1.0771, 0.7362, 0.9990, 0.8962, 1.1294, 0.9680, 0.9962, 1.1258]
```

**Implementation:**
```python
bsi = image.expression(
    '((c1*SWIR1 + c2*RED) - (c3*NIR + c4*BLUE)) / ((c5*SWIR1 + c6*RED) + (c7*NIR + c8*BLUE) + c9)',
    {
        'SWIR1': image.select('B11'),
        'SWIR2': image.select('B12'),
        'RED': image.select('B4'),
        'NIR': image.select('B8'),
        'BLUE': image.select('B2'),
        'GREEN': image.select('B3'),
        'c1': 1.1381, 'c2': 1.0771, 'c3': 0.7362,
        'c4': 0.9990, 'c5': 0.8962, 'c6': 1.1294,
        'c7': 0.9680, 'c8': 0.9962, 'c9': 1.1258
    }
).rename('BSI')
```

**Classification Threshold:**
```python
bare_soil_mask = bsi.gt(0.15)
```

**Accuracy:**
- F1-score: ~0.78
- Effective separation from vegetation and built-up areas

---

## Machine Learning Optimization

### Training Data Source

**Dataset**: ESA WorldCover 10m v100
- **Provider**: European Space Agency
- **Coverage**: Global
- **Resolution**: 10m (aligned with Sentinel-2)
- **Classes**: 11 land cover classes
- **Year**: 2020-2021
- **Accuracy**: Overall accuracy ~75% globally

**Relevant Classes for This Study:**
1. Tree cover
2. Shrubland
3. Grassland
4. Cropland
5. Built-up
6. Bare/sparse vegetation
7. Permanent water bodies

**Training Pixel Extraction:**
```python
# Sample pixels from ESA WorldCover
training_data = worldcover.sample(
    region=geometry,
    scale=10,
    numPixels=5000,
    seed=42
)

# Extract Sentinel-2 reflectances for each training pixel
training_features = training_data.map(lambda f: 
    f.set('B2', sentinel_image.select('B2').reduceRegion(...))
     .set('B3', sentinel_image.select('B3').reduceRegion(...))
     # ... repeat for all bands
)
```

**Data Split:**
- Training: 70% (3500 pixels)
- Validation: 15% (750 pixels)
- Testing: 15% (750 pixels)

### Optimization Algorithm: Nelder-Mead

**Method**: Nelder-Mead Downhill Simplex
- **Type**: Direct search optimization (derivative-free)
- **Advantages**:
  - No gradient computation required
  - Robust to noisy objective functions
  - Effective for small-to-medium dimensional problems
  - Handles non-smooth functions

**Library**: SciPy (`scipy.optimize.minimize`)

**Configuration:**
```python
from scipy.optimize import minimize

result = minimize(
    objective_function,
    x0=initial_guess,
    method='Nelder-Mead',
    options={
        'maxiter': 1000,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'disp': True
    }
)
```

### Objective Function Design

**Goal**: Maximize F1-Score

**F1-Score Formula:**
\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Where:
\[
Precision = \frac{TP}{TP + FP}, \quad Recall = \frac{TP}{TP + FN}
\]

**Why F1-Score?**
- Balances precision and recall
- Suitable for imbalanced classes
- Single metric for optimization
- Better than accuracy for multi-class problems

**Objective Function Implementation:**
```python
def objective(coefficients):
    # Calculate index using current coefficients
    index_values = calculate_custom_index(bands, coefficients)

    # Apply threshold to get binary predictions
    predictions = (index_values > threshold).astype(int)

    # Calculate F1-score
    f1 = f1_score(true_labels, predictions, average='binary')

    # Return negative (minimization problem)
    return -f1
```

### Optimization Process for BSI

**Step 1: Initialize Coefficients**
```python
initial_guess = np.ones(9)  # Start with all 1.0
```

**Step 2: Define BSI Formula Template**
```python
BSI = ((c[0]*B11 + c[1]*B4) - (c[2]*B8 + c[3]*B2)) / 
      ((c[4]*B11 + c[5]*B4) + (c[6]*B8 + c[7]*B2) + c[8])
```

**Step 3: Run Optimization**
```python
result = minimize(objective_bsi, initial_guess, method='Nelder-Mead')
optimal_coefficients = result.x
```

**Step 4: Validate**
```python
test_f1 = evaluate_f1(optimal_coefficients, test_data)
print(f"Test F1-Score: {test_f1:.4f}")
```

**Final BSI Coefficients:**
```
[1.1381, 1.0771, 0.7362, 0.9990, 0.8962, 1.1294, 0.9680, 0.9962, 1.1258]
```

**Training F1**: 0.802  
**Test F1**: 0.784

### Optimization Process for NBI

**Similar process with 5 coefficients:**
```
[1.1533, 1.2060, 0.7210, 0.9611, 0.9549]
```

**Training F1**: 0.871  
**Test F1**: 0.853

**Convergence:**
- Typical iterations: 200-400
- Convergence time: ~5-10 minutes (CPU)

---

## Classification Workflow

### End-to-End Pipeline

```
1. Image Acquisition → 2. Preprocessing → 3. Index Calculation
        ↓                      ↓                    ↓
4. Threshold Application → 5. Mask Generation → 6. Area Calculation
        ↓                      ↓                    ↓
7. Visualization → 8. Export Results → 9. Validation
```

### Detailed Steps

**1. Image Acquisition**
```python
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterDate('2021-01-01', '2021-12-31') \
    .filterBounds(geometry) \
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))

image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
```

**2. Index Calculation**
```python
ndvi = image.normalizedDifference(['B8', 'B4'])
ndwi = image.normalizedDifference(['B3', 'B8'])
mndwi = image.normalizedDifference(['B3', 'B11'])
ndbi = image.normalizedDifference(['B11', 'B8'])
# ... plus optimized BSI and NBI
```

**3. Binary Mask Creation**
```python
veg_mask = ndvi.gt(0.3)
water_mask = ndvi.lt(0.2).And(ndwi.gt(0.2)).And(mndwi.gt(0.2))
buildup_mask = ndbi.gt(0.1)
bare_mask = bsi.gt(0.15)
```

**4. Multi-class Classification**
```python
# Priority-based classification
classification = ee.Image(0)  # Default: unclassified
classification = classification.where(bare_mask, 1)     # Bare land
classification = classification.where(veg_mask, 2)      # Vegetation
classification = classification.where(buildup_mask, 3)  # Built-up
classification = classification.where(water_mask, 4)    # Water
```

**5. Area Calculation**
```python
def calculate_area(mask, geometry):
    area_image = mask.multiply(ee.Image.pixelArea())
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=10,
        maxPixels=1e10
    )
    area_km2 = ee.Number(stats.get('classification')).divide(1e6)
    return area_km2.getInfo()
```

---

## Area Calculation

### Pixel Area Method

**Formula:**
\[
Area_{km^2} = \frac{\sum (Pixel\ Count \times Pixel\ Area_{m^2})}{1,000,000}
\]

For 10m resolution:
\[
Area_{km^2} = \frac{Pixel\ Count \times 100}{1,000,000}
\]

**Implementation:**
```python
# Create area image (each pixel = its area in m²)
area_image = ee.Image.pixelArea()

# Multiply by classification mask
masked_area = area_image.updateMask(classification_mask)

# Sum all pixel areas
total_area = masked_area.reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geometry,
    scale=10,
    maxPixels=1e10
)

# Convert to km²
area_km2 = total_area.getInfo()['area'] / 1e6
```

### Results for Vijayawada (2021)

| Land Cover | Pixels | Area (km²) | Percentage | Threshold |
|------------|--------|-----------|-----------|-----------|
| Vegetation | 109,300 | 10.93 | 49.0% | NDVI > 0.3 |
| Water Bodies | 36,800 | 3.68 | 16.5% | Multi-index |
| Built-up Area | 187,800 | 18.78 | 84.2% | NDBI > 0.1 |
| Bare Land | 164,700 | 16.47 | 73.8% | BSI > 0.15 |

**Note**: Percentages exceed 100% due to mixed pixels and overlapping classifications.

---

## Visualization Techniques

### 1. True Color Composite
```python
Map.addLayer(image, {
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 3000
}, 'True Color')
```

### 2. False Color Composite (Vegetation Enhancement)
```python
Map.addLayer(image, {
    'bands': ['B8', 'B4', 'B3'],
    'min': 0,
    'max': 5000
}, 'False Color NIR')
```

### 3. Index Visualization
```python
# NDVI with color gradient
Map.addLayer(ndvi, {
    'min': -1,
    'max': 1,
    'palette': ['red', 'yellow', 'green']
}, 'NDVI')

# Water in blue
Map.addLayer(water_mask.selfMask(), {
    'palette': ['blue']
}, 'Water Bodies')
```

---

## Validation and Accuracy

### Validation Methods

1. **Visual Inspection**: Compare with high-resolution Google Earth imagery
2. **Cross-reference**: ESA WorldCover validation dataset
3. **F1-Score Metrics**: Calculated during optimization

### Accuracy Assessment

**BSI Optimization:**
- Training F1: 0.802
- Test F1: 0.784
- Accuracy: ~78%

**NBI Optimization:**
- Training F1: 0.871
- Test F1: 0.853
- Accuracy: ~85%

### Error Sources

1. **Mixed Pixels**: 10m resolution → multiple land covers per pixel
2. **Temporal Mismatch**: Training data vs. classification image dates
3. **Threshold Sensitivity**: Fixed thresholds may not suit all regions
4. **Shadow Effects**: Urban/terrain shadows misclassified as water

---

## Limitations and Challenges

### Technical Limitations

1. **Spatial Resolution**: 10m insufficient for detailed urban features
2. **Single Date Analysis**: Doesn't capture seasonal variations
3. **Fixed Thresholds**: Region-specific tuning needed
4. **Cloud Shadows**: Not fully removed by 5% cloud filter

### Methodological Challenges

1. **Training Data Quality**: ESA WorldCover ~75% accuracy
2. **Class Imbalance**: More built-up pixels than water pixels
3. **Optimization Convergence**: Local minima issues
4. **Generalization**: Coefficients may not transfer to other regions

### Future Improvements

1. **Machine Learning Classifiers**: Random Forest, SVM, Neural Networks
2. **Time-Series Analysis**: Multi-temporal change detection
3. **Object-Based Classification**: Segment-based rather than pixel-based
4. **DEM Integration**: Topographic correction
5. **Ground Truth Collection**: Field surveys for validation

---

## References

1. Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18-27.

2. Zanaga, D., et al. (2021). ESA WorldCover 10 m 2020 v100. *Zenodo*. doi:10.5281/zenodo.5571936

3. McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425-1432.

4. Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing*, 27(14), 3025-3033.

5. Zha, Y., Gao, J., & Ni, S. (2003). Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. *International Journal of Remote Sensing*, 24(3), 583-594.

6. Huete, A., et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment*, 83(1-2), 195-213.

7. Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308-313.

8. Main-Knorn, M., et al. (2017). Sen2Cor for Sentinel-2. *Image and Signal Processing for Remote Sensing XXIII*, 10427, 37-48.

---

**Document Version**: 1.0  
**Last Updated**: October 8, 2025  
**Author**: K. Emmanuel
