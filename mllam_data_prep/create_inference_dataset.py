import tempfile
from pathlib import Path

import isodate
import rich
from loguru import logger

from . import config as mdp_config
from .cli import create_argparser as create_base_argparser
from .create_dataset import create_dataset_zarr


def parse_key_value_arg(arg: str) -> tuple[str, str]:
    """
    Parse a single key=value argument into a dictionary.

    This function is intended for use with argparse's `type=` argument when parsing
    command-line arguments like: --overwrite-source-paths key1=value1 key2=value2

    Args:
        arg (str): A string in the format key=value.

    Returns:
        Dict[str, str]: A dictionary with one key-value pair parsed from the input string.

    Raises:
        argparse.ArgumentTypeError: If the argument is not in key=value format.
    """
    import argparse

    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{arg}'. Expected key=value."
        )

    key, value = arg.split("=", 1)
    return (key, value)


def main():
    parser = create_base_argparser()

    parser.add_argument(
        "--overwrite-input-paths",
        nargs="*",
        type=parse_key_value_arg,
        help=(
            "List of key=value pairs used to overwrite input paths of named inputs in the config file. "
            "For example: --overwrite-input-paths "
            "danra_surface=s3://mybucket/2025-05-01T1200Z/danra_surface.zarr "
            "danra_height_levels=s3://mybucket/2025-05-01T1200Z/danra_height_levels.zarr"
        ),
    )
    parser.add_argument(
        "--analysis_time",
        required=True,
        help="Analysis time to use for the dataset. This is used to select the correct time slice from the input data.",
        type=isodate.parse_datetime,
    )

    args = parser.parse_args()

    # the new sampling dimension is `analysis_time`
    old_sampling_dim = "time"
    sampling_dim = "analysis_time"
    # instead of only having `time` as dimension, the input forecast datasets
    # have two dimensions that describe the time value [analysis_time,
    # elapsed_forecast_duration]
    dim_replacements = dict(
        time=["analysis_time", "elapsed_forecast_duration"],
    )
    # there will be a single split called "test"
    split_name = "test"
    # which will have a single time slice, given by the analysis time argument
    # to the script
    sampling_coord_range = dict(
        start=args.analysis_time,
        end=args.analysis_time,
    )

    # load and modify the original config file
    fp_config = args.config
    config = mdp_config.Config.from_yaml_file(file=fp_config)

    if len(args.overwrite_input_paths) > 0:
        # Convert the list of tuples to a dictionary
        overwrite_input_paths = dict(args.overwrite_input_paths)
    else:
        overwrite_input_paths = None

    if overwrite_input_paths is not None:
        for key, value in overwrite_input_paths.items():
            if key not in config.inputs:
                raise ValueError(
                    f"Key {key} not found in config inputs. Available keys are: {list(config.inputs.keys())}"
                )
            logger.info(
                f"Overwriting input path for {key} with {value} previously {config.inputs[key].path}"
            )
            config.inputs[key].path = value

    # setup the split (test) for the dataset with a coordinate range along the
    # sampling dimension (analysis_time) of length 1
    config.output.splitting = mdp_config.Splitting(
        dim=sampling_dim, splits={split_name: mdp_config.Split(**sampling_coord_range)}
    )

    # ensure the output data is sampled along the sampling dimension
    # (analysis_time) too
    config.output.coord_ranges = {
        sampling_dim: mdp_config.Range(**sampling_coord_range)
    }

    config.output.chunking = {sampling_dim: 1}

    # replace old sampling_dimension (time) dimension in outputs with [`analysis_time`, `elapsed_forecast_time`]
    for variable, dims in config.output.variables.items():
        if old_sampling_dim in dims:
            orig_sampling_dim_index = dims.index(old_sampling_dim)
            dims.remove(old_sampling_dim)
            for dim in dim_replacements[old_sampling_dim][::-1]:
                dims.insert(orig_sampling_dim_index, dim)
            config.output.variables[variable] = dims
            logger.info(
                f"Replaced {old_sampling_dim} dimension with {dim_replacements[old_sampling_dim]} for {variable}"
            )

    # these dimensions should also be "renamed" from the input datasets
    for input_name in config.inputs.keys():
        if "time" in config.inputs[input_name].dim_mapping:
            dims = config.inputs[input_name].dims
            orig_sampling_dim_index = dims.index(old_sampling_dim)
            dims.remove(old_sampling_dim)
            for dim in dim_replacements[old_sampling_dim][::-1]:
                dims.insert(orig_sampling_dim_index, dim)
            config.inputs[input_name].dims = dims

            del config.inputs[input_name].dim_mapping[old_sampling_dim]

            # add new "rename" dim-mappins for `analysis_time` and `elapsed_forecast_duration`
            for dim in dim_replacements[old_sampling_dim]:
                config.inputs[input_name].dim_mapping[dim] = mdp_config.DimMapping(
                    method="rename", dim=dim
                )

    # save config to temporary filepath
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    fp_config_temp = Path(tmpfile.name)
    config.to_yaml_file(fp_config_temp)
    logger.info(f"Temporary config file created at {fp_config_temp}")

    rich.print(config)

    use_stats_from_path = args.use_stats_from_path

    create_dataset_zarr(
        fp_config=fp_config_temp,
        fp_zarr=args.output,
        overwrite=args.overwrite,
        use_stats_from_path=use_stats_from_path,
    )


if __name__ == "__main__":
    main()
