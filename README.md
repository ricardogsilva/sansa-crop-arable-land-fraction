# Crop Arable Land Fraction Algorithm (CALF)

This repository implements SANSA's _Crop Arable Land Fraction_ algorithm.


## Installation

- Install miniconda
  
- configure conda-forge
  
  ```
  conda config --prepend channels conda-forge
  conda config --set channel_priority strict
  ```
  
- Clone this repo
  
- Create and activate a conda env
  
  ```
  conda create --name sansa-calf --file spec-file.txt 
  conda activate sansa-calf
  ```
  
- Use pip to install it inside the conda env
  
  ```
  pip install --editable .
  ```


## Usage

This algorithm may be used either as a standalone shell command or inside Python code.


### Standalone command

After having installed the code, you should now have access to the `sansa-calf` command
in your terminal. Check it out with:

```
sansa-calf --help

# example execution (both region of interest and crop mask layers have a 
# `name` attribute as their unique identifier
sansa-calf \
    2015-01-01 \
    2021-12-31 \
    test_spot7_gauteng_old_eo3 \
    results/calf_cli.tif \
    name_1 \
    name_2 \
    --calf-aux-output-path=results/calf_aux_cli.tif \
    --calf-stats-output-path=results/calf_stats_cli.csv \
    --region-of-interest-path=test-data/auxiliary.gpkg \
    --region-of-interest-layer=region-of-interest \
    --crop-mask-path=test-data/auxiliary.gpkg \
    --crop-mask-layer=crop-mask \
    --output-resolution=3 \
    --datacube-configuration=test-data/docker/datacube/datacube.conf \
    --datacube-env=sandbox
```


### In Python

In order to use the calf algorithm together with other Python code, just import the `calf` package.

```
import calf

# compute calf
calf_result = calf.compute_calf(
    dc, 
    start_date="2015-01-01",
    end_date="2021-12-31",
    ard_product="test_spot7_gauteng_old_eo3",
    region_of_interest_gdf=roi_gdf,
    region_of_interest_unique_attribute="name_1",
    crop_mask_gdf=crop_gdf,
    crop_mask_unique_attribute="name_2",
    vegetation_threshold=0.2,
    red_band="red",
    nir_band="nir",
    qflags_band="spclass",
    output_crs="EPSG:32635",
    output_resolution=20,
)

# inspect and do further analysis with calf result

# optionally, save results to disk
calf.save_calf_result(calf_result, "mycalf.tif")
calf.save_aux_calf_result(calf_result, "myauxcalf.tif")
```


## CALF Introduction

CALF is based on the Normalised Difference vegetation Index (NDVI) derived
from SPOT Analysis-ready data (ARD) and ancillary field boundaries data. NDVI
is a dimensionless index that is indicative of vegetation vigor and density,
calculated as the normalised difference of the surface reflectance in the red
and near-infrared (NIR) regions of the electromagnetic spectrum. NDVI is a
proxy to quantify vegetation amount and vigor.

To exclude the non-vegetated surfaces, we apply a _vegetation mask_, defined
as NDVI values less than a user-specified threshold (defaulting to 0.2). These
pixel values are automatically assigned a Fallow class in the final product.
Depending on the known vegetation growth stage (based on the crop calendar), the
threshold used for masking non-vegetated surfaces may be increased (e.g., to
0.5 at the peak-of-the-season) to mask fallow fields covered with grass and
weeds during the growing season. Consequently, only pixels above the threshold
are considered for further analysis. Moreover, ancillary data, i.e., field
boundaries, are used to create a _crop mask_, which is, in turn, used to
limit the analysis to the boundaries of the known fields by the national
Department of Agriculture.


### Methodology


#### Normalised Difference Vegetation Index (NDVI)

Vegetation indices allow the delineation of the distribution of vegetation and
soil based on the characteristic reflectance patterns of green vegetation.
The NDVI has a dynamic range of between -1 and +1, with the negative values
representing non-vegetated surfaces such as bare soils and water bodies, and
positive values from 0.2 representing sparse to dense vegetation. NDVI is
calculated as a ratio between red and near-infrared bands of the electromagnetic
spectrum.

    NDVI = (NIR - Red) / (NIR + Red)

Where **NIR** denotes pixel-wise surface reflectance in the near-infrared band,
and **Red** denotes pixel-wise surface reflectance in the red band.


#### Crop and vegetation Masking

The cropland mask is developed to exclude the non-cropped areas using the crop
field boundaries.

The vegetation mask is created based on the computed NDVI. The threshold
of the mask is input by the user (typically ranging from 0.2 to 0.5, depending
on the crop growth stage). The resultant mask is a binary layer representing
planted and non-planted areas. The mask is applied to the NDVI layer before
computing the CALF product.


#### Crop Arable Land Fraction (CALF)

CALF is computed as:

    CALF = (Xt - Xmed) / Sigma

Where  _Xt_ denotes pixelwise NDVI at time t, _Xmed_ and _Sigma_ denote the
NDVI spatial (i.e. zonal) mean and standard deviation, respectively.


### CALF algorithm overview

This is an overview of the CALF algorithm, as implemented in the current repository:

1. Gather as inputs:

   - A timeseries of SPOT ARD according with user-supplied temporal range. The SPOT
     ARD datasets are retrieved from an opendatacube instance (user must supply the
     datacube configuration)

   - User-supplied administrative boundaries (_i.e._ the region of interest). This
     is supplied as a geospatial file with polygon geometries

   - User-supplied crop field boundaries (_i.e._ the crop mask). This is supplied
     as a geospatial file with polygon geometries

   - User-supplied vegetation threshold, used to discard pixels that do not represent
     actual vegetated areas

2. Compute the intersection of each region of interest polygon with each crop mask
   polygon in order to obtain the relevant patches where analysis will be made

3. For each patch:

   1. Retrieve relevant region of ARD datasets from the opendatacube

   2. Mask out invalid pixels according with the ARD quality flags

   3. Mask out pixels outside of the crop mask

   4. Compute NDVI for each SPOT ARD

   5. Aggregate each time step's NDVI using the method of _Greenest Pixel Compositing_.
      This results in the aggregated NDVI (seasonal NDVI) where each pixel is the
      greenest of the timeseries:

          NDVIseason = max(NDVIt1, NDVt2, ..., NDVItn)

   6. Discard pixels where aggregated NDVI is below the input vegetation mask

   7. Compute zonal NDVI statistics (mean and standard deviation)

   8. Compute CALF for each feature of the region of interest layer

   9. Reclassify the computed CALF dataset into a two-class result. Reclassification rules are:

      - calf <= 1: assign a value of `zero` - these are **fallow** pixels)

      - calf > 1: assign a value of `one` - these are **planted** pixels)

   10. Merge the patch's results into an overall matriz that spans the whole region of interest

4. After calculations are done, the algorithm returns:

   1. a dataset that includes:

      1. the reclassified calf - this is the main output
      2. the raw calf pixel values
      3. the aggregated NDVI

   2. the number of pixels of each class (planted, fallow) found in each patch


## Testing

The `test-data` directory contains some data that may be used for testing purposes

```
conda env create --name sansa-calf --file spec-file.txt
conda activate

cd test-data/docker
docker-compose -p calf-test up -d
cd -

# initialize the datacube DB
datacube --config test-data/datacube.conf --env sandbox system init

# add the products
datacube --config test-data/datacube.conf --env sandbox product add test-data/datacube-documents/product-definitions/*

# index datasets
datacube --config test-data/datacube.conf --env sandbox dataset add test-data/datacube-documents/dataset-documents/*

# sansa-calf --help

# or launch a jupyter notebook server and use the provided `calf-interactive-testing` notebook
DATACUBE_CONFIG_PATH=$PWD/test-data/datacube.conf jupyer lab
```