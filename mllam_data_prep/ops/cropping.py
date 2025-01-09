from typing import Tuple, Union

import numpy as np
import spherical_geometry as sg
import xarray as xr
from spherical_geometry.polygon import SphericalPolygon


def _get_latlon_coords(da: xr.DataArray) -> tuple:
    """
    Get the latlon coordinates of a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The data.

    Returns
    -------
    tuple (xarray.DataArray, xarray.DataArray)
        The latitude and longitude coordinates.
    """
    if "latitude" in da.coords and "longitude" in da.coords:
        return (da.latitude, da.longitude)
    elif "lat" in da.coords and "lon" in da.coords:
        return (da.lat, da.lon)
    else:
        raise Exception("Could not find lat/lon coordinates in DataArray.")


def create_convex_hull_mask(ds: xr.Dataset, ds_reference: xr.Dataset) -> xr.DataArray:
    """
    Create a grid-point mask for lat/lon coordinates in `da` indicating which
    points are interior to the convex hull of the lat/lon coordinates of
    `da_ref`.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset for which to create the mask.
    ds_reference : xarray.Dataset
        The reference dataset from which to create the convex hull of the coordinates.

    Returns
    -------
    xarray.DataArray
        A boolean mask indicating which points are interior to the convex hull.
    """
    da_lat, da_lon = _get_latlon_coords(ds)
    da_lat_ref, da_lon_ref = _get_latlon_coords(ds_reference)

    assert da_lat.dims == da_lon.dims
    assert da_lat_ref.dims == da_lon_ref.dims

    # latlon to (x, y, z) on unit sphere
    da_ref_xyz = _latlon_to_unit_sphere_xyz(da_lat=da_lat_ref, da_lon=da_lon_ref)

    chull_lam = SphericalPolygon.convex_hull(da_ref_xyz.values)

    # call .load() to avoid using dask arrays in the following apply_ufunc
    da_interior_mask = xr.apply_ufunc(
        chull_lam.contains_lonlat, da_lon.load(), da_lat.load(), vectorize=True
    ).astype(bool)
    da_interior_mask.attrs[
        "long_name"
    ] = "contained in convex hull of source dataset (da_ref)"

    return da_interior_mask


def _latlon_to_unit_sphere_xyz(
    da_lat: xr.DataArray, da_lon: xr.DataArray
) -> xr.DataArray:
    """
    Convert lat/lon coordinates to (x, y, z) on the unit sphere.

    Parameters
    ----------
    da_lat : xarray.DataArray
        Latitude coordinates.
    da_lon : xarray.DataArray
        Longitude coordinates.

    Returns
    -------
    xr.DataArray
        The (x, y, z) coordinates on the unit sphere as an xarray.DataArray
        with dimensions (grid_index, component).
    """
    pts_xyz = np.array(sg.vector.lonlat_to_vector(da_lon, da_lat)).T
    da_xyz = xr.DataArray(
        pts_xyz, coords=da_lat.coords, dims=list(da_lat.dims) + ["xyz"]
    )
    return da_xyz


def distance_to_convex_hull_boundary(
    ds: xr.Dataset,
    ds_reference: xr.Dataset,
    grid_index_dim: str = "grid_index",
    include_convex_hull_mask: bool = False,
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    For all points in `ds` that are external to the convex hull of the points in
    `ds_reference`, calculate the minimum distance to the convex hull boundary.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset for which to calculate the distance.
    ds_reference : xarray.Dataset
        The reference dataset from which to calculate the convex hull boundary.
    grid_index_dim : str, optional
        The name of the grid index dimension in `ds` and `ds_reference`.
    include_convex_hull_mask : bool, optional
        Whether to include the convex hull mask in the output.

    Returns
    -------
    da_mindist_to_ref : xarray.DataArray
        The minimum distance to the convex hull boundary.
    da_ch_mask : xarray.DataArray
        The convex hull mask (only if `include_convex_hull_mask` is True).

    """
    # rename the grid index dimension in ds_reference to avoid conflicts (since
    # the grid index dimension otherwise has the same name in both datasets,
    # and later we will want to find the minimum distance for each point in ds
    # to all points in ds_reference)
    ds_reference_separate_gridindex = ds_reference.rename(
        {grid_index_dim: "grid_index_ref"}
    )

    # create a mask from the convex hull of ds_reference for the grid points in ds
    da_ch_mask = create_convex_hull_mask(
        ds=ds, ds_reference=ds_reference_separate_gridindex
    )

    # only consider points that are external to the convex hull
    ds_exterior = ds.where(~da_ch_mask, drop=True)

    da_xyz = _latlon_to_unit_sphere_xyz(*_get_latlon_coords(ds_exterior))
    da_xyz_ref = _latlon_to_unit_sphere_xyz(
        *_get_latlon_coords(ds_reference_separate_gridindex)
    )

    def calc_dist(da_pt_xyz):
        dotproduct = np.dot(da_xyz_ref, da_pt_xyz.T)
        val = np.min(np.arccos(dotproduct))
        return val

    da_mindist_to_ref = xr.apply_ufunc(
        calc_dist,
        da_xyz,
        input_core_dims=[["xyz"]],
        output_core_dims=[[]],
        vectorize=True,
    )
    da_mindist_to_ref.attrs[
        "long_name"
    ] = "minimum distance to convex hull boundary of reference dataset"
    da_mindist_to_ref.attrs["units"] = "radians"

    if include_convex_hull_mask:
        return da_mindist_to_ref, da_ch_mask

    return da_mindist_to_ref


def crop_with_convex_hull(
    ds: xr.Dataset,
    ds_reference: xr.Dataset,
    grid_index_dim: str = "grid_index",
    margin_thickness: float = 2.0,
    include_interior_points: bool = True,
    return_mask=False,
) -> xr.Dataset:
    """
    Crop grid points (with coordinates given in lat/lon) in `ds` that are
    within a certain distance (within the margin of a given width) of the
    convex hull boundary of the points in `ds_reference`. The margin is
    measured in degrees.

    ┌──────────────────────────────────────┐
    │ Margin                               │
    │     ┌────── Convex hull ───────┐     │
    │     │                          │     │
    │     │ included if              │     │
    │     │ include_interior == True │     │
    │     │                          │     │
    │<--->│                          │     │
    │  :  └──────────────────────────┘     │
    │  :... margin width                   │
    └──────────────────────────────────────┘

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to crop.
    ds_reference : xarray.Dataset
        The reference dataset from which to calculate the convex hull boundary.
    grid_index_dim : str, optional
        The name of the grid index dimension in `ds` and `ds_reference`.
    margin_thickness : float, optional
        The thickness of the margin to apply to the convex hull boundary in
        degrees. Points within this margin will be included in the output.
    """
    if margin_thickness == 0.0:
        if not include_interior_points:
            raise Exception(
                "With no margin and exclude_interior=False, all points would be excluded."
            )
        da_mask = create_convex_hull_mask(ds=ds, ds_reference=ds_reference)
    else:
        da_min_dist_to_ref, da_ch_mask = distance_to_convex_hull_boundary(
            ds,
            ds_reference,
            grid_index_dim=grid_index_dim,
            include_convex_hull_mask=True,
        )

        max_dist_radians = margin_thickness * np.pi / 180.0
        da_boundary_mask = da_min_dist_to_ref < max_dist_radians

        if not include_interior_points:
            da_mask = da_boundary_mask
        else:
            da_interior_points = da_ch_mask.where(da_ch_mask, drop=True)
            da_mask = xr.concat(
                [da_interior_points, da_boundary_mask], dim="grid_index"
            )

    ds_cropped = ds.where(da_mask, drop=True)

    if return_mask:
        return ds_cropped, da_mask

    return ds_cropped
