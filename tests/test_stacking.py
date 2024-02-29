import numpy as np
import xarray as xr

import mllam_data_prep.main as mdp


def test_stack_variables_along_coord():
    """
    Test the stacking of variables along a coordinate

    i.e. from variables [var1, var2] with levels [1, 2, 3]
    to a single variable with levels [var1_l1, var1_l2, var1_l3, var2_l1, var2_l2, var2_l3]
    """
    name_format = "{var_name}_l{level}"
    nx, ny, nz = 10, 6, 3
    ds = xr.Dataset(
        {
            "var1": xr.DataArray(
                np.random.random((nx, ny, nz)),
                dims=(
                    "x",
                    "y",
                    "level",
                ),
            ),
            "var2": xr.DataArray(
                np.random.random((nx, ny, nz)),
                dims=(
                    "x",
                    "y",
                    "level",
                ),
            ),
        },
        coords={"level": np.arange(nz)},
    )

    combined_dim_name = "feature"
    da_stacked = mdp._stack_variables_by_coord_values(
        ds=ds,
        coord="level",
        name_format=name_format,
        combined_dim_name=combined_dim_name,
    )
    expected_coord_values = [
        name_format.format(var_name=v, level=level)
        for v in ["var1", "var2"]
        for level in range(nz)
    ]

    assert da_stacked.dims == ("x", "y", "feature")
    assert da_stacked.coords[combined_dim_name].values.tolist() == expected_coord_values
    for v in expected_coord_values:
        assert da_stacked.sel({combined_dim_name: v}).shape == (nx, ny)

    # check that the values are the same
    for v in ["var1", "var2"]:
        for level in [1, 2]:
            expected_values = ds[v].sel(level=level).values
            actual_values = da_stacked.sel(
                {combined_dim_name: name_format.format(var_name=v, level=level)}
            ).values
            assert np.all(expected_values == actual_values)


def test_stack_xy_coords():
    """
    Test stacking two (or more) coordinates to create a single coordinate, for
    example (x, y) grid coordinates to a single grid_index coordinate
    """
    nx, ny, nz = 10, 6, 3

    ds = xr.Dataset(
        {
            "var1": xr.DataArray(
                np.random.random((nx, ny, nz)),
                dims=(
                    "x",
                    "y",
                    "level",
                ),
            ),
            "var2": xr.DataArray(
                np.random.random((nx, ny, nz)),
                dims=(
                    "x",
                    "y",
                    "level",
                ),
            ),
        },
        coords={"level": np.arange(nz)},
    )
    dim_mapping = dict(
        grid_index=["x", "y"],
        feature=dict(
            dims=["level"], stack_variables_by_var_name=True, name="{level}_{var_name}"
        ),
    )

    da_stacked = mdp.map_dims_and_variables(ds=ds, dim_mapping=dim_mapping)

    assert set(da_stacked.dims) == set(("grid_index", "feature"))
    assert da_stacked.coords["grid_index"].shape == (nx * ny,)