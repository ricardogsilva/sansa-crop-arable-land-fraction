import dataclasses
import datetime as dt
import enum
import logging
import typing

import datacube
import fiona
import geopandas
import numpy as np
import rasterio.features
import xarray as xr
from datacube.model import Dataset as dc_Dataset
from datacube.utils import geometry as dc_geometry
from shapely import geometry

logger = logging.getLogger(__name__)


class CalfClassification(enum.Enum):
    FALLOW = 0
    PLANTED = 1


@dataclasses.dataclass()
class CalfComputationResult:
    numeric_calf: xr.DataArray
    reclassified_calf: xr.DataArray
    num_planted_pixels: int
    num_fallow_pixels: int
    total_pixels: int


def main(
        datacube_connection: datacube.Datacube,
        start_date: dt.datetime,
        end_date: dt.datetime,
        ard_product: str,
        red_band: str,
        nir_band: str,
        qflags_band: str,
        region_of_interest_df: geopandas.GeoDataFrame,
        crop_mask_df: geopandas.GeoDataFrame,
        vegetation_threshold: float,
        output_crs: str = "EPSG:32635",
        output_resolution: float = 10,
        resampling_method: str = "cubic",
        use_dask: bool = True
):
    datacube_base_query = {
        "product": ard_product,
        "measurements": [
            red_band,
            nir_band,
            qflags_band,
        ],
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
    if use_dask:
        datacube_base_query["dask_chunks"] = {
            "time": 1,
            "x": 3000,
            "y": 3000
        }

    # TODO: create arrays for the final products

    logger.debug("Finding input datasets...")
    found_datasets = get_ard_datasets(
        datacube_connection, datacube_base_query["time"], product_name=ard_product)
    if len(found_datasets) == 0:
        raise RuntimeError(
            "The datacube does not contain any datasets for the specified spatial "
            "and temporal range."
        )

    region_of_interest_df = _maybe_reproject(region_of_interest_df, output_crs)
    crop_mask_df = _maybe_reproject(crop_mask_df, output_crs)

    intersected_df = geopandas.overlay(
        region_of_interest_df,
        crop_mask_df,
        how="intersection",
        keep_geom_type=True
    )
    roi_feature_stats = {}
    for series_index, feature_series in enumerate(intersected_df.iterrows()):
        series_stats = roi_feature_stats.setdefault(feature_series["name1"], [])
        logger.info(f"Processing area {series_index + 1} of {len(intersected_df)})...")
        calf_result = _compute_calf(
            datacube_connection, datacube_base_query.copy(), feature_series,
            qflags_band, intersected_df.crs.to_epsg()
        )
        if calf_result is None:
            logger.warning(f"Could not calculate CALF for feature {feature_series}")
        else:
            series_stats.append(
                (
                    calf_result.num_fallow_pixels,
                    calf_result.num_planted_pixels,
                    calf_result.total_pixels
                )
            )
            pass  # now we need to put both numeric and reclassified CALF products into their respective final result arrays
    # TODO: write a CSV with the stats

    # maybe we can write out all of these partial calfs to disk and then in the end concatenate them all - this will likely conserve memory, at the expense of IO


def _compute_calf(
        datacube_connection: datacube.Datacube,
        datacube_base_query: typing.Dict,
        feature_series: geopandas.GeoSeries,
        qflags_band: str,
        feature_crs: int,
        vegetation_threshold: float,
) -> typing.Optional[CalfComputationResult]:
    datacube_query = datacube_base_query.copy()
    datacube_query["geopolygon"] = dc_geometry.Geometry(
        feature_series["geometry"].__geo_interface__,
        dc_geometry.CRS(f"EPSG:{feature_crs}")
    )
    logger.debug("Loading ARD data from datacube...")
    ds = datacube_connection.load(**datacube_query)
    if len(ds) == 0:
        logger.warning(
            f"No data loaded from the datacube for "
            f"feature_series: {feature_series}, skipping..."
        )
        result = None
    else:
        logger.debug("Applying cloud/shadow/water mask...")
        valid_da = apply_validity_mask(ds, qflags_band)
        logger.debug("Rasterizing crop mask feature...")
        crop_mask = rasterize_feature(valid_da, feature_series)
        logger.debug("Applying crop mask...")
        crop_da = valid_da * crop_mask
        logger.debug("Calculating daily NDVI...")
        ndvi_da = (crop_da.nir - crop_da.red) / (crop_da.nir + crop_da.red)
        logger.debug("Calculating aggregated NDVI...")
        aggregated_ndvi_da = ndvi_da.max(dim="time")
        aggregated_vegetation_ndvi = aggregated_ndvi_da.where(
            aggregated_ndvi_da > vegetation_threshold)
        mean = aggregated_vegetation_ndvi.mean()
        std_dev = aggregated_vegetation_ndvi.std()
        calf_da = (aggregated_vegetation_ndvi - mean) / std_dev
        reclassified_calf_da = xr.where(
            calf_da <= 1,
            CalfClassification.PLANTED.value,
            CalfClassification.FALLOW.value
        ) * crop_mask
        # now we need to make pixels that fall outside of the crop mask be Nan
        counts, frequencies = np.unique(reclassified_calf_da, return_counts=True)
        result = CalfComputationResult(
            numeric_calf=calf_da,
            reclassified_calf=reclassified_calf_da,
            num_planted_pixels=frequencies[counts == CalfClassification.PLANTED.value],
            num_fallow_pixels=frequencies[counts == CalfClassification.FALLOW.value],
            total_pixels=(~np.isnan(reclassified_calf_da)).sum()
        )
    return result




def _maybe_reproject(
        geodataframe: geopandas.GeoDataFrame,
        target_crs: str
) -> geopandas.GeoDataFrame:
    if geodataframe.crs.to_epsg() != int(target_crs.upper().replace("EPSG:", "")):
        logger.debug(f"Reprojecting {geodataframe}...")
        result = geodataframe.to_crs(target_crs)
    else:
        result = geodataframe
    return result


def get_ard_datasets(
        datacube_connection: datacube.Datacube,
        query_date: typing.Tuple[typing.Union[str, dt.datetime], typing.Union[str, dt.datetime]],
        product_name: typing.Optional[str] = None,
) -> typing.List[dc_Dataset]:
    """Find relevant datasets."""
    result = []
    if product_name is not None:
        result.extend(
            datacube_connection.find_datasets(product=product_name, time=query_date))
    else:
        for name in datacube_connection.list_products().name:
            found = datacube_connection.find_datasets(
                product=name,
                time=query_date
            )
            if len(found) > 0:
                result.extend(found)
                break
    return result


def apply_validity_mask(dataset: xr.Dataset, quality_flags_band: str):
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
    raw_flags = dataset[quality_flags_band].attrs["flags_definition"]["sca"]["values"]
    mask = None
    for value, flag in raw_flags.items():
        if flag.lower() in invalid_pixel_flags:
            if mask is None:
                mask = dataset[quality_flags_band] != value
            else:
                mask = mask | dataset[quality_flags_band] != value
    return dataset.where(mask)


def rasterize_feature(
        dataset_template: xr.Dataset,
        feature: geopandas.GeoSeries,
        all_touched: bool = False,
        fill_value = np.nan
) -> xr.DataArray:
    # crs = dataset_template.geobox.crs
    # reprojected = feature.to_crs(crs=crs)
    # shapes = reprojected.geometry
    shape = feature.geometry.__geo_interface__
    transform = dataset_template.geobox.transform
    y, x = dataset_template.geobox.shape
    rasterized_array = rasterio.features.rasterize(
        [(shape, 1)],
        out_shape=(y, x),
        transform=transform,
        all_touched=all_touched,
        fill=fill_value
    )
    dims = dataset_template.geobox.dims
    coords = dataset_template[dims[0]], dataset_template[dims[1]]
    rasterized_xarray = xr.DataArray(
        rasterized_array, coords=coords, dims=dims)
    return rasterized_xarray