import datetime as dt
import typing
import uuid
from pathlib import Path

import datacube
import typer
from datacube.model import Dataset as dc_Dataset

from . import (
    compute_calf,
    save_calf_result,
    save_aux_calf_result,
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
    """Calculate Crop Arable Land  Fraction (CALF)."""

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

    calf_dataset = compute_calf(
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
    main.write_result_to_disk(calf_dataset, output_path)


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
