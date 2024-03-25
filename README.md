# Semantics from Space: Satellite-Guided Thermal Semantic Segmentation Annotation for Aerial Field Robots

Welcome to the Semantics from Space repository! This project introduces a novel method for automatically generating semantic segmentation annotations for thermal imagery obtained from aerial vehicles. Our method leverages satellite-derived data products in conjunction with onboard global positioning and attitude estimates. This allows for precise and rapid annotation of thermal and non-thermal imagery at a massively-parallelizable scale. Please see our arxiv preprint for more details:

[1] [Lee, C., Soedarmadji, S., Anderson, M., Clark, A. J., & Chung, S. J. (2024). Semantics from Space: Satellite-Guided Thermal Semantic Segmentation Annotation for Aerial Field Robots. arXiv preprint arXiv:2403.14056.](https://arxiv.org/abs/2403.14056)


Key Features:

- 🤖 Automatic generation of semantic segmentation annotations for thermal imagery.
- 🛰️ Utilization of satellite-derived data products and onboard sensor data.
- 🔄 Incorporation of thermal-conditioned refinement step with visual foundation models.
- 📈 Achieves 98.5% performance compared to costly high-resolution options.
- 🚀 Demonstrates 70-160% improvement over existing zero-shot segmentation methods.

Feel free to explore the codebase and contribute to further advancements in thermal semantic perception algorithms!


## Environment Setup
### Setup anaconda environment.

```bash
# Create a new environment (using whatever environment manager you prefer)
conda create --name autoseg python=3.11.3
conda activate autoseg

# Install dependencies (using `python -m pip` to guard against using incorrect pip version)
python -m pip install --requirement requirements.txt
python -m pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Thermal Dataset
If you want to use the same data we use in our paper, please get the dataset from here: [https://github.com/aerorobotics/caltech-aerial-rgbt-dataset](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset). Follow the instructions in that repository to download and extract the thermal data and the `aligned.csv` files containing GPS, pose, and time-sync information.

## Satellite-Derived Data Download and Preprocessing
Scripts and notebooks like `dynamicworld_export.py` and those in folders `earth_engine_download` and `microsoft_planetarycomputer_download` are used to download geospatial data like landcover labels and high-resolution orthoimagery from the Google Earth Engine and Microsoft Planetary Computer. 

### Download DynamicWorld Land Use Land Cover (LULC) sets
1. Run `bash/dynamicworld_export.sh` to export data to Google Cloud Storage. Change `dynamicworld_export.py` to use Google Drive if desired but this is not recommended. 
2. Move downloaded tiles into `/data/microsoft_planetery_computer/dynamicworld/PLACE/tiles/` and `/data/chesapeake_bay_lulc/dynamicworld/PLACE/tiles/` or into your desired storage locations.

Notes: 
- Use the tag `--dryrun` to save a local preview of the LULC mosaic before committing to the full export. 
- Modify the `--start-date` and `--end-date` as needed. This range should just encompass the date the data trajectory was taken, but enlarged such that a full mosaic can be rendered (missing data may be present in some DynamicWorld tile rasters).

### Download NAIP/Planet/DSM/DEM tiles
1. Use scripts in `microsoft_planetarycomputer_download` to download DSM, DEM, NAIP from 2010-2021.
2. Use EarthExplorer to download NAIP after 2021. 
3. Place all downloaded tiles in their respective folders (use custom folder paths if desired).
4. Repeat all this for Chesapeake Bay Program tiles (or any other LULC dataset with labels and high-resolution imagery) if you're interested in tuning a Dense CRF for LULC refinement.

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

### Create and preprocess tiles into mosaics
Note: please update folder paths as needed. The example paths are customized for our own work.  
1. Ensure DynamicWorld and other data are downloaded and in their respective folders. 
2. Create mosaics from the downloaded raster tiles using `bash/merge_original_tiles.sh /data/microsoft_planetary_computer` and `bash/merge_original_tiles.sh /data/chesapeake_bay_lulc`. 
3. Combine planet RGB (3 band) with NIR from 4 band surface reflectance data using `bash bash/stack_planet_visual_nir.sh`. Note: this stores directly into the `mosaic` directories. Note: this step is necessary because NIR and RGB from Planet are provided separately.   
4. Preprocess (reproject, resample, crop to bounds) using `bash/batch_preprocess.sh /data/microsoft_planetary_computer` and `bash/batch_preprocess.sh /data/chesapeake_bay_lulc`.

### LULC refinement via Conditional Random Fields
First, change to the appropriate directory.
```bash
cd lulc_refinement
```
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

2. To train, run
```bash 
bash bash/train.sh
```
This will perform parameter optimization on the dense CRF using Optuna. See the bash script and accompanying code for more details on command line flags.

3. To run inference on aerial imagery, run
```bash
bash bash/refine.sh
```
or other similar scripts in the same directory following the pattern `*refine*.sh`.

## Semantic Segmentation Annotation Generation
### Coarse segmentation label rendering 
This step will likely require tinkering with the code to update to your robotic platforms coordinate frame and sensor measurement conventions.
```bash
cd autosegment
```
1. Please see the python files `autoseg_v2.py` and `gl_project.py`. The code assumes particular coordinate frames, altitude measurements, etc... Please update the files accordingly. 

2. Once updated, run:
```bash
bash bash/autoseg.sh
```
to generate the coarse segmentation labels.

### Rendered label refinement
This should be much more straightforward compared to the previous step. First, switch to the appropriate directory: 
```bash
cd autoseg_refinement
```
1. Use the Segment Anything repo to generate SAM segmentations. Combine the SAM segmentations (binary masks) into a single png with labels from `0-n` for `(n + 1)` segmentation instances.  
2. Use the files `bash/refine_seg.sh` and `bash/refine_helper.sh` as reference to run your Segment Anything-based refinement. 

## Issues and Contributing
If you find issues with this repo, or have code to contribute, please submit and issue and/or a PR above.

