# AerialAutoSegment: GPS-Enabled Semantic Segmentation Annotation for UAVs  

1. Check the sky masking code. thermal-41000.png looks odd. 
2. Check projections close to drone
3. Check grid formation for cliffs and other sharp changes in elevation. thermal-39000.png
4. Enable CRF refinement -- cannot help 10m resolution
5. Enable SegmentAnything refinement


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

