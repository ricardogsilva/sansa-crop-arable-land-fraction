import datetime as dt
import typing
import uuid
from pathlib import Path

import datacube
import typer
from datacube.model import Dataset as dc_Dataset

from . import (
    calf,
    utils,
)

app = typer.Typer()


@app.command()
def main(
        start_date: dt.datetime = typer.Argument(
            ...,
            help=(
                    "Start date for the CALF analysis. Together with the `end_date`, "
                    "this is used to query the DESA datacube for existing SPOT ARD "
                    "data."
            ),
            envvar="CALF__START_DATE"
        ),
        end_date: dt.datetime = typer.Argument(
            ...,
            help=(
                    "End date for the CALF analysis. Together with the `start_date`, "
                    "this is used to interrogate the DESA datacube for existing SPOT "
                    "ARD data."
            ),
            envvar="CALF__END_DATE"
        ),
        output_path: Path = typer.Argument(
            ...,
            help=("Where to save the generated CALF product"),
            envvar="CALF__OUTPUT_PATH"
        ),
        region_of_interest: typing.Optional[Path] = typer.Option(
            None,
            help=(
                    "Path to a spatial vector file that has the region of interest "
                    "to use for the CALF analysis. If not supplied, then the "
                    "analysis will fall back to using the bounding box of the "
                    "crop mask as the region of interest. This file is expected "
                    "to contain polygon geometries. Any file format supported by "
                    "fiona is valid (e.g. shapefile, geopackage, spatialite, etc.). "
                    "This shall likely be the administrative boundaries file."
            ),
            envvar="CALF__REGION_OF_INTEREST"
        ),
        region_of_interest_layer: typing.Optional[str] = typer.Option(
            "1",
            help=(
                    "Index (1-based) or name of the layer from the region of "
                    "interest file to use."
            ),
            envvar="CALF__REGION_OF_INTEREST_LAYER"
        ),
        crop_mask_path: typing.Optional[Path] = typer.Option(
            None,
            help=(
                "Path to a spatial vector file that has the crop mask to use for the "
                "CALF analysis. If not supplied the CALF analysis will be done for "
                "the full region of interest. Similarly to the `region_of_interest` "
                "parameter, file is expected to contain polygon geometries. Any file "
                "format supported by fiona is valid."
            ),
            envvar="CALF__CROP_MASK_PATH"
        ),
        crop_mask_layer: typing.Optional[str] = typer.Option(
            "1",
            help=(
                "Index (1-based) or name of the layer from the crop mask file to use."
            ),
            envvar="CALF__CROP_MASK_LAYER"
        ),
        vegetation_mask_threshold: typing.Optional[float] = typer.Option(
            0.2,
            help="Vegetation mask threshold.",
            envvar="CALF__VEGETATION_MASK_THRESHOLD",
        ),
        output_crs: typing.Optional[str] = typer.Option("EPSG:32635"),
        output_resolution: typing.Optional[int] = typer.Option(10),
        resampling_method: typing.Optional[str] = typer.Option("cubic"),
        datacube_configuration: typing.Optional[Path] = typer.Option(
            None,
            help=(
                    "Path to the datacube configuration file. Defaults to searching "
                    "in the usual datacube locations (`/etc/datacube.conf`, "
                    "`~/.datacube.conf`)."
            ),
            envvar="DATACUBE_CONFIG_PATH",
        ),
        datacube_env: typing.Optional[str] = typer.Option(
            "default",
            help="Name of the datacube environment to use for the CALF analysis",
            envvar="DATACUBE_ENVIRONMENT",
        ),
        ard_product: typing.Optional[str] = typer.Option(
            None,
            help=(
                "Name of the DESA ARD product to use when calculating CALF. If "
                "not supplied, it will be tentatively auto-detected from the "
                "input dates."
            ),
            envvar="CALF__ARD_PRODUCT"
        ),
        ard_product_red_band: typing.Optional[str] = typer.Option(
            "red",
            help="Name of the Red band in the ARD product - used for calculating NDVI.",
            envvar="CALF_RED_BAND"
        ),
        ard_product_nir_band: typing.Optional[str] = typer.Option(
            "nir",
            help=(
                    "Name of the Near Infrared band in the ARD product - used for "
                    "calculating NDVI."
            ),
            envvar="CALF_NIR_BAND"
        ),
        ard_product_q_flags_band: typing.Optional[str] = typer.Option(
            "spclass",
            help="Name of the Quality Flags band in the ARD product.",
            envvar="CALF_QFLAGS_BAND"
        ),
):
    """Calculate Crop Arable Land  Fraction (CALF).

    CALF is based on the Normalised Difference vegetation Index (NDVI) derived
    from SPOT Analysis-ready data (ARD) and ancillary field boundaries data. NDVI
    is a dimensionless index that is indicative of vegetation vigor and density,
    calculated as the normalised difference of the surface reflectance in the red
    and near-infrared (NIR) regions of the electromagnetic spectrum. NDVI is a
    proxy to quantify vegetation amount and vigor.

    To exclude the non-vegetated surfaces, we apply a **vegetation mask**, defined
    as NDVI values less than a user-specified threshold (defaulting to 0.2). These
    pixel values are automatically assigned a Fallow class in the final product.
    Depending on the known vegetation growth stage (based on the crop calendar), the
    threshold used for masking non-vegetated surfaces may be increased (e.g., to
    0.5 at the peak-of-the-season) to mask fallow fields covered with grass and
    weeds during the growing season. Consequently, only pixels above the threshold
    are considered for further analysis. Moreover, ancillary data, i.e., field
    boundaries, are used to create a **crop mask**, which is, in turn, used to
    limit the analysis to the boundaries of the known fields by the national
    Department of Agriculture.


    If a crop mask is not provided then calculations shall be done over all of the
    input region of interest.

    If a region of interest is not provided, then it is computed from the crop mask.
    This means that at least one of them must be provided.


    ## Methodology


    ### Normalised Difference Vegetation Index (NDVI)

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


    ### Crop and vegetation Masking

    The cropland mask is developed to exclude the non-cropped areas using the crop
    field boundaries.

    The vegetation mask is created based on the computed NDVI. The threshold
    of the mask is input by the user (typically ranging from 0.2 to 0.5, depending
    on the crop growth stage). The resultant mask is a binary layer representing
    planted and non-planted areas. The mask is applied to the NDVI layer before
    computing the CALF product.


    ### Crop Arable Land Fraction (CALF)

    CALF is computed as:

        CALF = (Xt - Xmed) / Sigma

    Where  **Xt** denotes pixelwise NDVI at time t, **Xmed** and **Sigma** denote the
    NDVI spatial (i.e. zonal) mean and standard deviation, respectively.


    #### CALF algorithm overview

    1. Gather as inputs:

       - A timeseries of SPOT ARD

       - Crop field boundaries

       - Administrative boundaries (i.e. the region of interest)

    2. Compute NDVI for each SPOT ARD

    3. Where multiple acquisitions exist (in the input temporal period), NDVI layers
       are composited using the method of Greenest Pixel Compositing - this means to
       build a composited NDVI (seasonal NDVI) where each pixel is the greenest of the
       timeseries: NDVIseason = max(NDVIt1, NDVt2, ..., NDVItn)

    4. Compute crop and vegetation masks. Areas outside of the crop mask are disregarded
       from further calculations. Areas below the vegetation threshold are set to zero

    5. Compute zonal statistics for NDVI each feature of the region of interest layer.

    6. Compute CALF for each feature of the region of interest layer.

    7. Write out output products


    ## Quality assessment

    The CALF code shall inherit SPOT ARD quality flags and shall also


    ## Outputs

    This calculation produces the following outputs:

    1. A GeoTiff file with two bands:

       - Computed CALF numeric values
       - Quality flags

    2. A reclassified GeoTiff file with with classes (planted and fallow).
       Reclassification is done according with the following rules:

       - CALF <= 1 : Fallow (i.e. not planted)

       - CALF > 1 : Planted

    3. A vector file with the region of interest used for the analysis where each
       individual feature of the region of interest has an estimate of the size of
       each CALF class in hectares an also the proportion of planted land to total
       arable land.


    """

    if region_of_interest is None and crop_mask_path is None:
        raise typer.Abort("Must provide either a region of interest, a crop mask, or both")
    elif region_of_interest is not None:
        roi = _get_bounding_box(region_of_interest, region_of_interest_layer)
    else:
        roi = _get_bounding_box(crop_mask_path, crop_mask_layer)

    try:
        dc = datacube.Datacube(
            app=f"calf-algorithm-{uuid.uuid4()}",
            config=(
                str(datacube_configuration) if datacube_configuration is not None
                else None
            ),
            env=datacube_env,
        )
    except ValueError:
        typer.Abort("Could not connect to datacube")

    calf_dataset = calf.main(
        datacube_connection=dc,
        start_date=start_date,
        end_date=end_date,
        ard_product=ard_product,
        red_band=ard_product_red_band,
        nir_band=ard_product_nir_band,
        qflags_band=ard_product_q_flags_band,
        region_of_interest=None,
        crop_mask=None,
        vegetation_threshold=vegetation_mask_threshold,
        output_crs=output_crs,
        output_resolution=output_resolution,
        resampling_method=resampling_method,
        use_dask=False
    )
    calf.write_result_to_disk(calf_dataset, output_path)


def _get_datasets(
        dc: datacube.Datacube,
        query_date: typing.Tuple[str, str],
        product_name: typing.Optional[str] = None,
) -> typing.List[dc_Dataset]:
    """Find relevant datasets."""
    result = []
    if product_name is not None:
        result.extend(dc.find_datasets(product=product_name, time=query_date))
    else:
        for name in dc.list_products().name:
            found = dc.find_datasets(
                product=name,
                time=query_date
            )
            if len(found) > 0:
                result.extend(found)
                break
    return result


if __name__ == "__main__":
    app()
