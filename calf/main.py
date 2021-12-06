import dataclasses
import datetime as dt
import enum
import logging
import typing
from pathlib import Path

import datacube
import geopandas
import numpy as np
import pandas
import rasterio.features
import rasterio.windows
import rioxarray  # NOTE: this import is needed in order to use rioxarray, don't remove!
import xarray as xr
from datacube.model import Dataset as dc_Dataset
from datacube.utils import geometry as dc_geometry
from rioxarray.exceptions import OneDimensionalRaster

logger = logging.getLogger(__name__)


class CalfClassification(enum.Enum):
    FALLOW = 0
    PLANTED = 1


class CalfOutputName(enum.Enum):
    NUMERIC_CALF = "raw_calf"
    RECLASSIFIED_CALF = "calf"
    SEASONAL_NDVI = "seasonal_ndvi"


class MissingValue(enum.Enum):
    NUMERIC_CALF = 100
    RECLASSIFIED_CALF = 100
    SEASONAL_NDVI = 100


@dataclasses.dataclass()
class CalfComputationResult:
    numeric_calf: xr.DataArray
    reclassified_calf: xr.DataArray
    seasonal_ndvi: xr.DataArray
    num_planted_pixels: int
    num_fallow_pixels: int
    total_pixels: int
    mean: float
    std_dev: float


@dataclasses.dataclass()
class CalfAlgorithmResult:
    calf_ds: xr.Dataset
    calf_stats: pandas.DataFrame
    patches: typing.List[CalfComputationResult]


def compute_calf(
        datacube_connection: datacube.Datacube,
        start_date: typing.Union[str, dt.datetime],
        end_date: typing.Union[str, dt.datetime],
        ard_product: str,
        region_of_interest_gdf: geopandas.GeoDataFrame,
        crop_mask_gdf: geopandas.GeoDataFrame,
        vegetation_threshold: typing.Optional[float] = 0.2,
        red_band: typing.Optional[str] = "red",
        nir_band: typing.Optional[str] = "nir",
        qflags_band: typing.Optional[str] = "spclass",
        output_crs: str = "EPSG:32635",
        output_resolution: int = 10,
        resampling_method: str = "cubic",
        return_patches: typing.Optional[bool] = False,
        region_of_interest_unique_attribute: typing.Optional[str] = None,
        crop_mask_unique_attribute: typing.Optional[str] = None,

) -> CalfAlgorithmResult:
    roi_attribute, crop_attribute = _validate_unique_attributes(
        region_of_interest_unique_attribute,
        region_of_interest_gdf,
        crop_mask_unique_attribute,
        crop_mask_gdf
    )
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
        reclassified_calf_fill=MissingValue.RECLASSIFIED_CALF.value,
        seasonal_ndvi_fill=np.nan,
        numeric_calf_name=CalfOutputName.NUMERIC_CALF.value,
        reclassified_calf_name=CalfOutputName.RECLASSIFIED_CALF.value,
        seasonal_ndvi_name=CalfOutputName.SEASONAL_NDVI.value,
        attrs={
            "calf_start_date": start_date,
            "calf_end_date": end_date,
            "calf_vegetation_threshold": vegetation_threshold,
            "calf_input_product": ard_product,
        }
    )
    intersected_df = geopandas.overlay(
        region_of_interest_gdf,
        crop_mask_gdf,
        how="intersection",
        keep_geom_type=True
    )
    exploded_intersected_gdf = intersected_df.explode()
    roi_feature_stats = []
    feature_calf_results = []
    for current, iterator_data in enumerate(exploded_intersected_gdf.iterrows()):
        series_index, feature_series = iterator_data
        logger.debug(
            f"Processing area { current + 1} of {len(exploded_intersected_gdf)} ({series_index}) "
            f"({feature_series[roi_attribute]} - {feature_series[crop_attribute]})..."
        )
        calf_result = _compute_patch_calf(
            datacube_connection,
            datacube_base_query.copy(),
            feature_series,
            qflags_band,
            exploded_intersected_gdf.crs.to_epsg(),
            vegetation_threshold,
            roi_attribute,
            crop_attribute
        )
        if calf_result is None:
            logger.warning(f"Could not calculate CALF for feature {feature_series}")
        else:
            roi_feature_stats.append(
                (
                    feature_series[roi_attribute],
                    feature_series[crop_attribute],
                    calf_result.num_fallow_pixels,
                    calf_result.num_planted_pixels,
                    calf_result.total_pixels
                )
            )
            if return_patches:
                feature_calf_results.append(calf_result)
            logger.debug("Gathering seasonal ndvi onto the main output dataset...")
            _overlay_arrays(
                output_ds[CalfOutputName.SEASONAL_NDVI.value],
                calf_result.seasonal_ndvi,
            )
            logger.debug("Gathering numeric CALF onto the main output dataset...")
            _overlay_arrays(
                output_ds[CalfOutputName.NUMERIC_CALF.value],
                calf_result.numeric_calf,
            )
            logger.debug("Gathering reclassified CALF onto the main output dataset...")
            _overlay_arrays(
                output_ds[CalfOutputName.RECLASSIFIED_CALF.value],
                calf_result.reclassified_calf,
                no_data_value=MissingValue.RECLASSIFIED_CALF.value
            )
    return CalfAlgorithmResult(
        calf_ds=output_ds,
        calf_stats=_consolidate_stats(roi_feature_stats, output_resolution),
        patches=feature_calf_results
    )


def save_calf_result(calf_ds: xr.Dataset, output_path: Path):
    valid_output_path = validate_output_path(output_path, [".tif", ".tiff"], ".tif")
    valid_output_path.parent.mkdir(parents=True, exist_ok=True)
    int_ds = calf_ds[[CalfOutputName.RECLASSIFIED_CALF.value]]
    int_ds.rio.to_raster(valid_output_path)


def save_aux_calf_result(calf_ds: xr.Dataset, output_path: Path):
    valid_output_path = validate_output_path(output_path, [".tif", ".tiff"], ".tif")
    valid_output_path.parent.mkdir(parents=True, exist_ok=True)
    float_ds = calf_ds[[CalfOutputName.NUMERIC_CALF.value, CalfOutputName.SEASONAL_NDVI.value]]
    float_ds.rio.to_raster(valid_output_path)


def validate_output_path(candidate: Path, valid_extensions: typing.List[str], default_extension: str) -> Path:
    if candidate.is_dir():
        raise IOError("Output path must not point to an existing directory.")
    if candidate.suffix.lower() not in (ext.lower() for ext in valid_extensions):
        new_name = f"{candidate.name}{default_extension}"
    else:
        new_name = candidate.name
    return Path(candidate.parent) / new_name


def _consolidate_stats(
        calf_stats: typing.List[typing.Tuple[str, str, int, int, int]],
        spatial_resolution: int
) -> pandas.DataFrame:
    """Consolidate calf stats in a pandas DataFrame.

    Note that this function assumes the input calf stats were obtained
    in a projected coordinate system.

    """

    calf_stats = pandas.DataFrame({
        "region_of_interest": pandas.Series([row[0] for row in calf_stats]),
        "crop_mask": pandas.Series([row[1] for row in calf_stats]),
        "fallow_pixels": pandas.Series([row[2] for row in calf_stats]),
        "planted_pixels": pandas.Series([row[3] for row in calf_stats]),
        "computed_pixels": pandas.Series([row[4] for row in calf_stats]),
    })
    calf_stats["pixel_resolution"] = spatial_resolution
    calf_stats["fallow_hectares"] = (calf_stats["fallow_pixels"] * spatial_resolution) / 10_000
    calf_stats["planted_hectares"] = (calf_stats["planted_pixels"] * spatial_resolution) / 10_000
    calf_stats["total_computed_hectares"] = (calf_stats["computed_pixels"] * spatial_resolution) / 10_000
    return calf_stats


def _generate_output_dataset(
        region_of_interest: geopandas.GeoDataFrame,
        output_resolution,
        numeric_calf_fill: typing.Optional[int] = 0,
        reclassified_calf_fill: typing.Optional[int] = 0,
        seasonal_ndvi_fill: typing.Optional[int] = 0,
        numeric_calf_name: typing.Optional[str] = "raw_calf",
        reclassified_calf_name: typing.Optional[str] = "calf",
        seasonal_ndvi_name: typing.Optional[str] = "seasonal_ndvi",
        attrs: typing.Optional[typing.Dict] = None
) -> xr.Dataset:
    output_seasonal_ndvi_da = _rasterize_geodataframe(
        region_of_interest,
        output_resolution,
        burn_value=seasonal_ndvi_fill,
        all_touched=True,
        fill=seasonal_ndvi_fill,
        dtype=np.float64
    )
    output_calf_da = _rasterize_geodataframe(
        region_of_interest,
        output_resolution,
        burn_value=numeric_calf_fill,
        all_touched=True,
        fill=numeric_calf_fill,
        dtype=np.float64
    )
    output_reclassified_da = _rasterize_geodataframe(
        region_of_interest,
        output_resolution,
        burn_value=reclassified_calf_fill,
        all_touched=True,
        fill=reclassified_calf_fill,
        dtype=np.uint8
    )
    output_reclassified_da.rio.write_nodata(reclassified_calf_fill, inplace=True)
    result = xr.Dataset(
        {
            seasonal_ndvi_name: output_seasonal_ndvi_da,
            numeric_calf_name: output_calf_da,
            reclassified_calf_name: output_reclassified_da
        }
    )
    for attr_name, attr_value in (attrs or {}).items():
        result.attrs[attr_name] = attr_value
    return result


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


def _compute_patch_calf(
        datacube_connection: datacube.Datacube,
        datacube_base_query: typing.Dict,
        feature_series: geopandas.GeoSeries,
        qflags_band: str,
        feature_crs: int,
        vegetation_threshold: float,
        region_of_interest_unique_attribute: str,
        crop_mask_unique_attribute: str,
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
        base_attrs = {
            "region_of_interest_feature": feature_series[
                region_of_interest_unique_attribute],
            "crop_mask_feature": feature_series[crop_mask_unique_attribute],
        }
        logger.debug("Applying cloud/shadow/water mask...")
        valid_da = apply_validity_mask(ds, qflags_band)
        logger.debug("Rasterizing crop mask feature...")
        crop_mask = _rasterize_feature(valid_da, feature_series)
        logger.debug("Applying crop mask...")
        crop_da = valid_da * crop_mask
        logger.debug("Calculating daily NDVI...")
        ndvi_da = (crop_da.nir - crop_da.red) / (crop_da.nir + crop_da.red)
        logger.debug("Calculating aggregated NDVI...")
        aggregated_ndvi_da = ndvi_da.max(dim="time")
        aggregated_ndvi_da.attrs.update(base_attrs)
        logger.debug("Applying vegetation threshold to aggregated NDVI...")
        aggregated_vegetation_ndvi = aggregated_ndvi_da.where(
            aggregated_ndvi_da > vegetation_threshold)
        mean = float(aggregated_vegetation_ndvi.mean())
        std_dev = float(aggregated_vegetation_ndvi.std())
        logger.debug(f"aggregated NDVI stats: mean: {mean} std_dev: {std_dev}")
        logger.debug("Computing numeric CALF...")
        calf_da = (aggregated_vegetation_ndvi - mean) / std_dev
        calf_da.attrs.update(base_attrs)
        calf_da.rio.write_crs(feature_crs, inplace=True)
        logger.debug("Computing reclassified CALF...")
        reclassified_calf_da = xr.where(
            calf_da <= 1,
            CalfClassification.FALLOW.value,
            CalfClassification.PLANTED.value
        ) * crop_mask
        reclassified_calf_da.attrs.update(base_attrs)

        # setting the missing value for reclassified CALF - it cannot be `np.nan`
        # because we are using a dtype of uint8 and np.nan is considered a float value
        int_reclassified_calf_da = reclassified_calf_da.fillna(
            MissingValue.RECLASSIFIED_CALF.value).astype(np.uint8)

        logger.debug("Counting number of fallow and planted pixels...")
        fallow, planted, total = _count_calf_pixels(int_reclassified_calf_da)
        result = CalfComputationResult(
            numeric_calf=calf_da,
            reclassified_calf=int_reclassified_calf_da,
            seasonal_ndvi=aggregated_vegetation_ndvi,
            num_planted_pixels=planted,
            num_fallow_pixels=fallow,
            total_pixels=total,
            mean=mean,
            std_dev=std_dev
        )
    return result


def _count_calf_pixels(reclassified_calf: xr.DataArray) -> typing.Tuple[int, int, int]:
    logger.debug("Counting number of fallow and planted pixels...")
    counts, frequencies = np.unique(reclassified_calf, return_counts=True)
    try:
        num_planted = int(frequencies[counts == CalfClassification.PLANTED.value])
    except TypeError:
        num_planted = 0
    try:
        num_fallow = int(frequencies[counts == CalfClassification.FALLOW.value])
    except TypeError:
        num_fallow = 0
    total = num_planted + num_fallow
    return num_fallow, num_planted, total


def _overlay_arrays(
        bottom: xr.DataArray,
        top: xr.DataArray,
        no_data_value: typing.Optional[typing.Any] = None
) -> None:
    try:
        top_bounds = top.rio.bounds()
    except OneDimensionalRaster:
        # rioxarray has trouble getting the bounds of 1-d rasters, so we get them manually
        x_offset, x_resolution, _, y_offset, _, y_resolution = top.affine.to_gdal()
        left_bound = x_offset
        right_bound = x_offset + top.rio.width * x_resolution
        top_bound = y_offset
        bottom_bound = y_offset + top.rio.height * y_resolution
        top_bounds = (left_bound, bottom_bound, right_bound, top_bound)

    window = rasterio.windows.from_bounds(
        *top_bounds,
        transform=bottom.rio.transform()
    )
    start_row = int(window.row_off)
    end_row = int(window.row_off + window.height)
    start_col = int(window.col_off)
    end_col = int(window.col_off + window.width)

    overlay_region = bottom[start_row:end_row, start_col:end_col]
    region = overlay_region.data
    top_arr = top.data
    if no_data_value is None:
        region = np.where(~np.isnan(top_arr), top_arr, region)
    else:
        region = np.where(top_arr != no_data_value, top_arr, region)
    bottom[start_row:end_row, start_col:end_col] = region


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


def _rasterize_feature(
        dataset_template: xr.Dataset,
        feature: geopandas.GeoSeries,
        all_touched: bool = False,
        burn_value: typing.Optional[int] = 1,
        fill_value=np.nan
) -> xr.DataArray:
    # crs = dataset_template.geobox.crs
    # reprojected = feature.to_crs(crs=crs)
    # shapes = reprojected.geometry
    shape = feature.geometry.__geo_interface__
    transform = dataset_template.geobox.transform
    y, x = dataset_template.geobox.shape
    rasterized_array = rasterio.features.rasterize(
        [(shape, burn_value)],
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


def _rasterize_geodataframe(
        gdf: geopandas.GeoDataFrame,
        target_resolution: int,
        burn_value: typing.Optional[int] = 1,
        all_touched: typing.Optional[bool] = False,
        fill: typing.Optional[int] = 0,
        dtype: typing.Optional[np.dtype] = None
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
    ).astype(dtype)
    rasterized_da.rio.write_crs(f"epsg:{gdf.crs.to_epsg()}", inplace=True)
    return rasterized_da


def _get_coords(geo_transform: rasterio.Affine, width: int, height: int):
    row_coords = [rasterio.transform.xy(geo_transform, r, 0)[1] for r in range(height)]
    col_coords = [rasterio.transform.xy(geo_transform, 0, c)[0] for c in range(width)]
    return row_coords, col_coords


def _validate_unique_attributes(
        region_of_interest_attribute: typing.Optional[str],
        region_of_interest_gdf: geopandas.GeoDataFrame,
        crop_mask_attribute: typing.Optional[str],
        crop_mask_gdf: geopandas.GeoDataFrame
) -> typing.Tuple[str, str]:
    if region_of_interest_attribute is None:
        region_of_interest_attribute = region_of_interest_gdf.columns[0]
        roi_attr_exists = True
    else:
        roi_attr_exists = region_of_interest_attribute in region_of_interest_gdf.columns

    if crop_mask_attribute is None:
        crop_mask_attribute = crop_mask_gdf.columns[0]
        crop_attr_exists = True
    else:
        crop_attr_exists = crop_mask_attribute in crop_mask_gdf.columns

    if roi_attr_exists and crop_attr_exists:
        if region_of_interest_attribute == crop_mask_attribute:
            roi_attr = f"{region_of_interest_attribute}_1"
            crop_attr = f"{crop_mask_attribute}_2"
        else:
            roi_attr = region_of_interest_attribute
            crop_attr = crop_mask_attribute
    elif not roi_attr_exists:
        raise RuntimeError(
            f"Invalid region of interest attribute: {region_of_interest_attribute!r}")
    elif not crop_attr_exists:
        raise RuntimeError(f"Invalid crop mask attribute: {crop_mask_attribute!r}")
    else:
        raise RuntimeError("Reached undefined state when validating attribute names")
    return roi_attr, crop_attr