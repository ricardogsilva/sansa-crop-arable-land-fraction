import dataclasses
import datetime as dt
import enum
import logging
import typing
from pathlib import Path

import datacube
import fiona
import geopandas
import numpy as np
import rasterio.features
import rasterio.windows
import rioxarray
import xarray as xr
from datacube.model import Dataset as dc_Dataset
from datacube.utils import geometry as dc_geometry
from shapely import geometry

logger = logging.getLogger(__name__)


class CalfClassification(enum.Enum):
    FALLOW = 0
    PLANTED = 1


class CalfOutputName(enum.Enum):
    NUMERIC_CALF = "raw_calf"
    RECLASSIFIED_CALF = "calf"
    SEASONAL_NDVI = "seasonal_ndvi"


@dataclasses.dataclass()
class CalfComputationResult:
    numeric_calf: xr.DataArray
    reclassified_calf: xr.DataArray
    seasonal_ndvi: xr.DataArray
    num_planted_pixels: int
    num_fallow_pixels: int
    total_pixels: int


def main(
        datacube_connection: datacube.Datacube,
        start_date: dt.datetime,
        end_date: dt.datetime,
        ard_product: str,
        region_of_interest_gdf: geopandas.GeoDataFrame,
        crop_mask_gdf: geopandas.GeoDataFrame,
        vegetation_threshold: typing.Optional[float] = 0.2,
        red_band: typing.Optional[str] = "red",
        nir_band: typing.Optional[str] = "nir",
        qflags_band: typing.Optional[str] = "spclass",
        output_crs: str = "EPSG:32635",
        output_resolution: float = 10,
        resampling_method: str = "cubic",
        use_dask: bool = False
) -> xr.Dataset:
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
    }
    if use_dask:
        datacube_base_query["dask_chunks"] = {
            "time": 1,
            "x": 3000,
            "y": 3000
        }

    logger.debug("Finding input datasets...")
    found_datasets = get_ard_datasets(
        datacube_connection, datacube_base_query["time"], product_name=ard_product)
    if len(found_datasets) == 0:
        raise RuntimeError(
            "The datacube does not contain any datasets for the specified spatial "
            "and temporal range."
        )

    region_of_interest_gdf = _maybe_reproject(region_of_interest_gdf, output_crs)
    crop_mask_gdf = _maybe_reproject(crop_mask_gdf, output_crs)

    # this is where we will store the results
    output_ds = _generate_output_dataset(
        region_of_interest_gdf, output_resolution,
        numeric_calf_fill=np.nan,
        reclassified_calf_fill=np.nan,
        seasonal_ndvi_fill=np.nan,
        numeric_calf_name=CalfOutputName.NUMERIC_CALF.value,
        reclassified_calf_name=CalfOutputName.RECLASSIFIED_CALF.value,
        seasonal_ndvi_name=CalfOutputName.SEASONAL_NDVI.value
    )
    intersected_df = geopandas.overlay(
        region_of_interest_gdf,
        crop_mask_gdf,
        how="intersection",
        keep_geom_type=True
    )
    roi_feature_stats = {}
    feature_calf_results = []
    for series_index, feature_series in intersected_df.iterrows():
        logger.info(
            f"Processing area {series_index + 1} of {len(intersected_df)} "
            f"({feature_series['name_1']} - {feature_series['name_2']})..."
        )
        series_stats = roi_feature_stats.setdefault(feature_series["name_1"], [])
        calf_result = _compute_calf(
            datacube_connection,
            datacube_base_query.copy(),
            feature_series,
            qflags_band,
            intersected_df.crs.to_epsg(),
            vegetation_threshold
        )
        if calf_result is None:
            logger.warning(f"Could not calculate CALF for feature {feature_series}")
        else:
            series_stats.append(
                (
                    feature_series["name_2"],
                    calf_result.num_fallow_pixels,
                    calf_result.num_planted_pixels,
                    calf_result.total_pixels
                )
            )
            feature_calf_results.append(calf_result)
            logger.debug("Gathering seasonal ndvi onto the main output dataset...")
            _overlay_arrays(
                output_ds[CalfOutputName.SEASONAL_NDVI.value],
                calf_result.seasonal_ndvi
            )
            logger.debug("Gathering numeric CALF onto the main output dataset...")
            _overlay_arrays(
                output_ds[CalfOutputName.NUMERIC_CALF.value],
                calf_result.numeric_calf
            )
            logger.debug("Gathering reclassified CALF onto the main output dataset...")
            _overlay_arrays(
                output_ds[CalfOutputName.RECLASSIFIED_CALF.value],
                calf_result.reclassified_calf
            )
    # TODO: write a CSV with the stats
    return output_ds, feature_calf_results


def write_result_to_disk(calf_ds: xr.Dataset, output_path: Path):
    calf_ds.rio.to_raster(output_path)


def _generate_output_dataset(
        region_of_interest: geopandas.GeoDataFrame,
        output_resolution,
        numeric_calf_fill: typing.Optional[int] = 0,
        reclassified_calf_fill: typing.Optional[int] = 0,
        seasonal_ndvi_fill: typing.Optional[int] = 0,
        numeric_calf_name: typing.Optional[str] = "raw_calf",
        reclassified_calf_name: typing.Optional[str] = "calf",
        seasonal_ndvi_name: typing.Optional[str] = "seasonal_ndvi",
) -> xr.Dataset:
    output_seasonal_ndvi_da = rasterize_geodataframe(
        region_of_interest,
        output_resolution,
        burn_value=seasonal_ndvi_fill,
        all_touched=True,
        fill=numeric_calf_fill
    )
    output_calf_da = rasterize_geodataframe(
        region_of_interest,
        output_resolution,
        burn_value=numeric_calf_fill,
        all_touched=True,
        fill=numeric_calf_fill
    )
    output_reclassified_da = rasterize_geodataframe(
        region_of_interest,
        output_resolution,
        burn_value=reclassified_calf_fill,
        all_touched=True,
        fill=reclassified_calf_fill
    )
    return xr.Dataset(
        {
            seasonal_ndvi_name: output_seasonal_ndvi_da,
            numeric_calf_name: output_calf_da,
            reclassified_calf_name: output_reclassified_da
        }
    )


def _write_raster(
        data_: np.array,
        dtype: np.dtype,
        output_path: Path,
        resolution: int,
        width: int,
        height: int,
        geo_transform: rasterio.Affine,
        crs: str,
):
    """Write raster to disk."""

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "width": width,
        "height": height,
        "transform": geo_transform,
        "crs": crs,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data_, indexes=1)


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
        logger.debug("Applying vegetation threshold to aggregated NDVI...")
        aggregated_vegetation_ndvi = aggregated_ndvi_da.where(
            aggregated_ndvi_da > vegetation_threshold)
        mean = aggregated_vegetation_ndvi.mean()
        std_dev = aggregated_vegetation_ndvi.std()
        logger.debug(f"aggregated NDVI stats: mean: {float(mean)} std_dev: {float(std_dev)}")
        logger.debug("Computing numeric CALF...")
        calf_da = (aggregated_vegetation_ndvi - mean) / std_dev
        calf_da.rio.write_crs(feature_crs, inplace=True)
        logger.debug("Computing reclassified CALF...")
        reclassified_calf_da = xr.where(
            calf_da <= 1,
            CalfClassification.PLANTED.value,
            CalfClassification.FALLOW.value
        )
        reclassified_calf_da = reclassified_calf_da.astype(np.uint8) * crop_mask
        logger.debug("Counting number of fallow and planted pixels...")
        counts, frequencies = np.unique(reclassified_calf_da, return_counts=True)
        result = CalfComputationResult(
            numeric_calf=calf_da,
            reclassified_calf=reclassified_calf_da,
            seasonal_ndvi=aggregated_vegetation_ndvi,
            num_planted_pixels=int(frequencies[counts == CalfClassification.PLANTED.value]),
            num_fallow_pixels=int(frequencies[counts == CalfClassification.FALLOW.value]),
            total_pixels=int((~np.isnan(reclassified_calf_da)).sum())
        )
    return result


def _overlay_arrays(bottom: xr.DataArray, top: xr.DataArray):
    window = rasterio.windows.from_bounds(
        *top.rio.bounds(),
        transform=bottom.rio.transform()
    )
    start_row = int(window.row_off)
    end_row = int(window.row_off + window.height)
    start_col = int(window.col_off)
    end_col = int(window.col_off + window.width)
    bottom[start_row:end_row, start_col:end_col] = top.data



def _maybe_reproject(
        geodataframe: geopandas.GeoDataFrame,
        target_crs: str
) -> geopandas.GeoDataFrame:
    if geodataframe.crs.to_epsg() != int(target_crs.upper().replace("EPSG:", "")):
        logger.debug(f"Reprojecting geodataframe...")
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


def rasterize_geodataframe(
        gdf: geopandas.GeoDataFrame,
        target_resolution: int,
        burn_value: typing.Optional[int] = 1,
        all_touched: typing.Optional[bool] = False,
        fill: typing.Optional[int] = 0,
) -> xr.DataArray:
    left, bottom, right, top = gdf.total_bounds
    width = int(np.ceil((right - left) / target_resolution))
    height = int(np.ceil((top - bottom) / target_resolution))
    geo_transform = rasterio.Affine(
        target_resolution,
        0.0,
        left,
        0.0,
        -target_resolution,
        top
    )
    rasterized = rasterio.features.rasterize(
        ((geom, burn_value) for geom in gdf.geometry),
        out_shape=(height, width),
        transform=geo_transform,
        all_touched=all_touched,
        fill=fill,
    )
    coords = _get_coords(geo_transform, width, height)
    rasterized_da = xr.DataArray(
        rasterized,
        dims=("y", "x"),
        coords={
            "x": coords[1],
            "y": coords[0],
        }
    )
    rasterized_da.rio.write_crs(f"epsg:{gdf.crs.to_epsg()}", inplace=True)
    return rasterized_da


def _get_coords(geo_transform: rasterio.Affine, width: int, height: int):
    row_coords = [rasterio.transform.xy(geo_transform, r, 0)[1] for r in range(height)]
    col_coords = [rasterio.transform.xy(geo_transform, 0, c)[0] for c in range(width)]
    return row_coords, col_coords
