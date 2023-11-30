# AerialAutoSegment: GPS-Enabled Semantic Segmentation Annotation for UAVs  

1. ~~Check the sky masking code. thermal-41000.png looks odd.~~ 
2. ~~Check projections close to drone~~
3. Check grid formation for cliffs and other sharp changes in elevation. thermal-39000.png
4. ~~Enable CRF refinement -- cannot help 10m resolution~~
5. (saraswati) OpenGL 3D to 2D rendering -- should solve problem 3. 
6. (saraswati) Enable SegmentAnything refinement


## Environment Setup
### Setup anaconda environment. 
I recommend miniconda and [mamba](https://mamba.readthedocs.io/en/latest/installation.html) (faster version of anaconda). Once miniconda/mamba is installed, install the conda file using the yaml file provided.
```
conda env create -f environment.yml
```

### Install missing packages via pip
Install `rasterio`, `pyproj`, `geojson`, etc. 

Install `pydensecrf` from source via
```
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Run the code
Scripts and notebooks like `dynamicworld_export.py` and those in folders `earth_engine_download` and `microsoft_planetarycomputer_download` are used to download geospatial data like landcover labels and high-resolution orthoimagery from the Google Earth Engine and Microsoft Planetary Computer. 

To run the code, use the notebook `autoseg.ipynb`. 

1. Run the first cell, which should create a `pkl` file, if not available already. This file saves label and surface map interpolation functions on a UTM grid. Note: the notebook is useful here because loading the pickle file is still pretty slow, so try to avoid rerunning the first cell or restarting the kernel. 

2. Run the second cell to warp overhead labels into the drone forward-facing camera frame. Results will be stored in `outputs`. 

## Preprocessing Experiments (9/26 Updates)
Perform the following for both our collected datasets and a select few tiles from the 2018 Chesapeake Bay Land Cover dataset. The Chesapeake bay dataset will be used for validation of our tile preprocessing methods.  

### Download DynamicWorld Land Use Land Cover (LULC) sets
1. Run `bash/dynamicworld_export.sh` to export data to Google Cloud Storage. Change `dynamicworld_export.py` to use Google Drive if desired. 
2. Move downloaded tiles into `/data/microsoft_planetery_computer/dynamicworld/PLACE/tiles/` and `/data/chesapeake_bay_lulc/dynamicworld/PLACE/tiles/`

Notes: 
- Use the tag `--dryrun` to save a local preview of the LULC mosaic before committing to the full export. 
- Modify the `--start-date` and `--end-date` as needed. This range should just encompass the date the data trajectory was taken, but enlarged such that a full mosaic can be rendered (missing data may be present in some DynamicWorld tile rasters).

### Download NAIP/Planet/DSM/DEM tiles
1. Use scripts in `microsoft_planetarycomputer_download` to download DSM, DEM, NAIP from 2010-2021.
2. Use EarthExplorer to download NAIP after 2021. 
3. Place all downloaded tiles in their respective folders shown above.
4. Planet must be downloaded as single composite `.tif` files. More code would need to be written to handle tiles.  

Paths to directory on `lambda`:
- NAIP path: `/data/microsoft_planetary_computer/naip/PLACE/tiles`
- Planet path: `/data/microsoft_planetary_computer/planet/PLACE/[visual / 4band]` 
- Digital Surface Map (DSM) path: `/data/microsoft_planetary_computer/dsm/PLACE/tiles`
- Digital Elevation Map (DEM 10 meters) path: `/data/microsoft_planetary_computer/dem/PLACE/tiles`
- Digital Elevation Map (DEM 1 meter) path: `/data/microsoft_planetary_computer/dem_1m/PLACE/tiles`

| Dataset | NAIP (0.6/1.0m) visual quality | DSM (2m) Available | DEM (10m) Available | DEM (1m) Available | 
| --- | --- | --- | --- | --- |
| Duck | Raster borders are obvious; difference in water reflectance |  Yes | Yes | No |
| Kentucky River | Fine | Yes | Yes | Possibly (but not downloaded) | 
| Colorado River | Fine | No | Yes | Yes (looks odd) |
| Castaic Lake | Fine | Yes | Yes | Yes |
| Big Bear | Raster borders obvious; clouds over the lake |  Yes | Yes | Yes |

### Create and preprocess tiles into mosaics
1. Ensure DynamicWorld and other data are downloaded and in their respective folders. 
2. Create mosaics from the downloaded raster tiles using `bash/merge_original_tiles.sh /data/microsoft_planetary_computer` and `bash/merge_original_tiles.sh /data/chesapeake_bay_lulc`.
3. Combine planet RGB (3 band) with NIR from 4 band surface reflectance data using `bash bash/stack_planet_visual_nir.sh`. Note: this stores directly into the `mosaic` directories.   
4. Preprocess (reproject, resample, crop to bounds) using `bash/batch_preprocess.sh /data/microsoft_planetary_computer` and `bash/batch_preprocess.sh /data/chesapeake_bay_lulc`.

### LULC refinement via Conditional Random Fields
1. **Convert DynamicWorld and Chesapeake Bay LULC labels into a common set of labels** (see table below) via `bash bash/commonize_lulc_labels.sh`. This is necessary if training/evaluating CRF refinement using the 1-meter resolution labels from Chesapeake Bay as ground truth. This step can be skipped if just doing CRF refinement without training but is still recommended to merge some similar labels. 

    | New Index | Common Name | Dynamic World | Chesapeake Bay | 
    | --- | --- | --- | --- |
    | 0 | Water |Water | Water |
    | 1 | Trees |Trees | Tree Canopy<br/> Tree Canopy over Impervious Structures<br/> Tree Canopy over Other Impervious<br/> Tree Canopy over Impervious Roads |
    | 2 | Low Brush | Grass<br/> Crops | Low Vegetation |
    | 3 | Shrub and Scrub | Shrub and Scrub | Shrub and Scrub |
    | 4 | Wetlands | Flooded Vegetation | Emergent Wetlands |
    | 5 | Human-made Structures | Built Up Area | Impervious Structures<br/> Other Impervious<br/> Impervious Roads |
    | 6 | Bare Ground | Bare Ground | Barren |
    | 7 | N/A |Snow/Ice  |                          |
    | 8 | N/A |      | Aberdeen Proving Grounds |
2. 