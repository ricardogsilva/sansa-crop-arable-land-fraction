import datetime as dt
import logging
import typing
from pathlib import Path

import datacube
import fiona
import geopandas
import pandas
import rasterio.features
import typer
import xarray as xr
from datacube.utils import geometry as dc_geometry
from datacube.model import Dataset as dc_Dataset

from . import utils

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def main(
        start_date: dt.datetime,
        end_date: dt.datetime,
        region_of_interest: typing.Optional[Path] = None,
        crop_mask_path: typing.Optional[Path] = None,
        crop_mask_layer: typing.Optional[str] = "1",
        output_crs: typing.Optional[str] = "EPSG:32635",
        output_resolution: typing.Optional[int] = 10,
        resampling_method: typing.Optional[str] = "cubic",
        datacube_configuration: typing.Optional[Path] = None,
        datacube_env: typing.Optional[str] = "default",
        # product: typing.Optional[str] = None
):
    """Calculate CALF.

    If a crop mask is not provided then calculations shall be done over all of the
    input region of interest

    If a region of interest is not provided, then it is computed from the crop mask.
    This means that at least one of them must be provided.

    """

    if region_of_interest is None and crop_mask_path is None:
        logger.critical("Must provide a region of interest or a crop mask")
        raise typer.Abort()
    elif region_of_interest is not None:
        roi = region_of_interest  # TODO: should probably get the geometry of the ROI
    else:
        roi = _get_roi_bounds(crop_mask_path, crop_mask_layer)  # FIXME: Be sure to use the same representation for the ROI as used above

    # create arrays for the final products

    base_query_args = {
        "measurements": [
            "red",
            "nir",
            "spclass",
        ],
        # "x": roi[0:3:2],
        # "y": roi[1:4:2],
        "time": (start_date, end_date),
        "output_crs": output_crs,
        "resolution": (-output_resolution, output_resolution),
        "resampling": resampling_method,
        # "dask_chunks": {
        #     "time": 1,
        #     "x": 3000,
        #     "y": 3000
        # },
    }
    logger.debug("Connecting to datacube...")
    dc = datacube.Datacube(
        app="calf-algorithm",
        config=(
            str(datacube_configuration) if datacube_configuration is not None
            else None
        ),
        env=datacube_env,
    )
    logger.debug("Finding input datasets...")
    found_datasets = _get_datasets(dc, base_query_args["time"])
    if len(found_datasets) == 0:
        logger.info(
            "The datacube does not contain any datasets for the specified spatial "
            "and temporal range."
        )
        raise typer.Abort()
    datasets_query = base_query_args.copy()
    datasets_query["datasets"] = found_datasets

    if crop_mask_path is not None:
        logger.debug("Processing crop mask...")
        crop_mask_gdf = geopandas.read_file(crop_mask_path, layer=crop_mask_layer)
        for index, feature_series in enumerate(crop_mask_gdf.iterrows()):
            logger.info(
                f"Processing crop mask feature ({index + 1}/{len(crop_mask_gdf)})...")
            query = datasets_query.copy()
            query["geopolygon"] = dc_geometry.Geometry(
                feature_series["geometry"].__geo_interface__,
                dc_geometry.CRS(f"EPSG:{crop_mask_gdf.crs.to_epsg()}")
            )
            logger.debug("Loading datasets...")
            ds = dc.load_data(**query)
            logger.debug("Applying cloud/shadow/water mask...")
            valid_ds = _apply_validity_mask(ds)
            logger.debug("Rasterizing crop mask feature...")
            crop_mask = _rasterize_feature(valid_ds, feature_series)
            logger.debug("Applying crop mask...")
            crop_ds = valid_ds.where(crop_mask)
            logger.debug("Calculating daily NDVI...")
            ndvi = (crop_ds.nir - crop_ds.red) / (crop_ds.nir + crop_ds.red)
            logger.debug("Calculating aggregated NDVI...")
            aggregated_ndvi = ndvi.max(dim="time")
            mean = aggregated_ndvi.mean()
            stddev = aggregated_ndvi.std()
            calf = (aggregated_ndvi - mean) / stddev
            # - write CALF to final array
        # - reclassify to planted/fallow
        # - write final arrays
        # - calculate stats
    else:
        logger.debug("Processing entire region of interest...")
        pass


def _rasterize_feature(
        dataset_template: xr.Dataset, feature: pandas.Series) -> xr.DataArray:
    crs = dataset_template.geobox.crs
    transform = dataset_template.geobox.transform
    dims = dataset_template.geobox.dims
    coords = dataset_template[dims[0]], dataset_template[dims[1]]
    y, x = dataset_template.geobox.shape
    reprojected = feature.to_crs(crs=crs)
    shapes = reprojected.geometry
    rasterized_array = rasterio.features.rasterize(
        shapes, out_shape=(y, x), transform=transform)
    rasterized_xarray = xr.DataArray(
        rasterized_array, coords=coords, dims=dims)
    return rasterized_xarray


def _apply_validity_mask(dataset: xr.Dataset):
    invalid_pixel_flags = [
        "cloud cirrus",
        "cloud thick",
        "cloud thin",
        "shadow or water",
        "shadow other",
        "snow",
        "water deep",
        "water green",
        "water turbid",
        "water shallow",
    ]
    raw_flags = dataset.spclass.attrs["flags_definition"]["sca"]["values"]
    mask = dataset == True
    for value, flag in raw_flags.items():
        if flag.lower() in invalid_pixel_flags:
            mask = mask & (dataset.spclass != value)
    return dataset.where(mask)


def _get_datasets(
        dc: datacube.Datacube,
        query_date: typing.Tuple[str, str],
) -> typing.List[dc_Dataset]:
    result = []
    for name in dc.list_products().name:
        result.append(
            dc.find_datasets(
                product=name,
                time=query_date
            )
        )
    return result


def _get_relevant_products(
        dc: datacube.Datacube, base_query: typing.Dict) -> typing.List[str]:
    result = []
    for name in dc.list_products().name:
        existing_datasets = dc.find_datasets(
            product=name,
            **base_query
        )
        if len(existing_datasets) > 0:
            result.append(name)
    return result


def _get_roi_bounds(
        crop_mask_path: Path,
        layer_identifier: typing.Optional[typing.Union[str, int]] = None
) -> typing.Tuple:
    with fiona.open(crop_mask_path, layer=layer_identifier) as src:
        return src.bounds


if __name__ == "__main__":
    typer_handler = utils.TyperLoggerHandler()
    logging.basicConfig(level=logging.DEBUG, handlers=(typer_handler,))
    app()