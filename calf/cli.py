import datetime as dt
import typing
import uuid
from pathlib import Path

import datacube
import fiona.errors
import geopandas
import typer
from jinja2 import (
    Environment,
    PackageLoader
)

import calf

app = typer.Typer()


@app.command()
def compute_calf(
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
        ard_product: str = typer.Argument(
            ...,
            help="Name of the DESA ARD product to use when calculating CALF",
            envvar="CALF__ARD_PRODUCT"
        ),
        calf_output_path: Path = typer.Argument(
            ...,
            help=("Where to save the generated CALF product"),
            envvar="CALF__OUTPUT_PATH"
        ),
        calf_aux_output_path: typing.Optional[Path] = typer.Option(
            None,
            help=(
                    "Where to save the generated calf auxiliary datasets. "
                    "If not provided, these auxiliary datasets are not "
                    "saved to disk."
            ),
            envvar="CALF__AUX_OUTPUT_PATH"
        ),
        calf_stats_output_path: typing.Optional[Path] = typer.Option(
            None,
            help=(
                    "Where to save the generated calf stats CSV file. "
                    "If not provided, calf stats CSV file is not "
                    "saved to disk."
            ),
            envvar="CALF__STATS_OUTPUT_PATH"
        ),
        region_of_interest_path: typing.Optional[Path] = typer.Option(
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
        region_of_interest_unique_attribute: typing.Optional[str] = typer.Option(
            None,
            help=(
                    "Name of the attribute that can be used to refer to individual "
                    "features in the region of interest polygon layer. Defaults to the "
                    "name of the first column in the layer's attribute table."
            ),
            envvar="CALF__REGION_OF_INTEREST_UNIQUE_ATTRIBUTE"
        ),
        crop_mask_unique_attribute: typing.Optional[str] = typer.Option(
            None,
            help=(
                    "Name of the attribute that can be used to refer to individual "
                    "features in the crop mask polygon layer. Defaults to the name of "
                    "the first column in the layer's attribute table."
            ),
            envvar="CALF__CROP_MASK_UNIQUE_ATTRIBUTE"
        ),
):
    """Calculate Crop Arable Land  Fraction (CALF)."""

    if region_of_interest_path is None and crop_mask_path is None:
        typer.secho(
            "Must provide either a region of interest, a crop mask, or both",
            fg=typer.colors.RED
        )
        raise typer.Abort()

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
        typer.secho("Could not connect to datacube", fg=typer.colors.RED)
        raise typer.Abort()

    try:
        region_of_interest_gdf = geopandas.read_file(
            region_of_interest_path, layer=region_of_interest_layer)
    except fiona.errors.DriverError as exc:
        typer.secho(
            f"Could not read region of interest file: {str(exc)}", fg=typer.colors.RED)
        raise typer.Abort()

    try:
        crop_mask_gdf = geopandas.read_file(
            crop_mask_path, layer=crop_mask_layer)
    except fiona.errors.DriverError as exc:
        typer.secho(f"Could not read crop mask file: {str(exc)}", fg=typer.colors.RED)
        raise typer.Abort()

    typer.secho("Computing CALF...", fg=typer.colors.MAGENTA)
    calf_result = calf.compute_calf(
        datacube_connection=dc,
        start_date=start_date,
        end_date=end_date,
        ard_product=ard_product,
        region_of_interest_gdf=region_of_interest_gdf,
        crop_mask_gdf=crop_mask_gdf,
        vegetation_threshold=vegetation_mask_threshold,
        red_band=ard_product_red_band,
        nir_band=ard_product_nir_band,
        qflags_band=ard_product_q_flags_band,
        output_crs=output_crs,
        output_resolution=output_resolution,
        resampling_method=resampling_method,
        return_patches=False,
        region_of_interest_unique_attribute=region_of_interest_unique_attribute,
        crop_mask_unique_attribute=crop_mask_unique_attribute,
    )
    typer.secho("calf stats", fg=typer.colors.GREEN)
    typer.secho(calf_result.calf_stats.to_markdown(), fg=typer.colors.GREEN)
    typer.secho("Saving results...", fg=typer.colors.MAGENTA)
    calf.save_calf_result(calf_result.calf_ds, calf_output_path)
    typer.secho(
        f"Saved main calf output to {str(calf_output_path)!r}",
        fg=typer.colors.GREEN
    )
    if calf_aux_output_path is not None:
        calf.save_aux_calf_result(calf_result.calf_ds, calf_aux_output_path)
        typer.secho(
            f"Saved aux calf output to {str(calf_aux_output_path)!r}",
            fg=typer.colors.GREEN
        )
    if calf_stats_output_path is not None:
        calf_stats_valid_output_path = calf.validate_output_path(
            calf_stats_output_path, [".csv"], ".csv")
        calf_stats_valid_output_path.parent.mkdir(parents=True, exist_ok=True)
        calf_result.calf_stats.to_csv(calf_stats_output_path, index=False)
        typer.secho(
            f"Saved calf stats output to {str(calf_stats_output_path)!r}",
            fg=typer.colors.GREEN
        )


@app.command()
def prepare_sample_data(
        base_path: typing.Optional[Path] = typer.Option(
            Path(__file__).parents[1] / "test-data",
            help="Base path for the sample data",
            envvar="CALF__SAMPLE_DATA_BASE_PATH"
        ),
        rendered_directory: typing.Optional[Path] = typer.Option(
            Path.cwd(),
            help="Where to store the rendered datacube dataset-document file",
            envvar="CALF__RENDERED_DIRECTORY"
        )
):
    env = Environment(
        loader=PackageLoader("calf", "templates")
    )
    template_name = "spot7-dataset-document.yml"
    template = env.get_template(f"datacube-documents/dataset-documents/{template_name}")
    rendered = template.render(base_path=base_path)
    output_path = rendered_directory / template_name
    with output_path.open(mode="w") as fh:
        fh.write(rendered)
    typer.secho(
        f"Dataset document file has been rendered at {str(output_path)!r}",
        fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    app()
