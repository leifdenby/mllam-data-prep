from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import dataclass_wizard
from dataclass_wizard import JSONWizard


class InvalidConfigException(Exception):
    pass


class GlobalJSONMeta(JSONWizard.Meta):
    """
    Global settings for the JSON load/dump process, that should apply to
    *all* subclasses of `JSONWizard`.

    Note: it does not matter where this class is defined, as long as it's
    declared before any methods in `JSONWizard` are called.
    """

    raise_on_unknown_json_key = True


@dataclass
class Range:
    """
    Defines a range for a variable to be used for selection, i.e.
    `xarray.Dataset.sel({var_name}: slice({start}, {end}, {step}))`, the variable
    name is the key in the dictionary and the slice object is created from the
    `start`, `end`, and `step` attributes.

    Attributes
    ----------
    start: str
        The start of the range, e.g. "1990-09-03T00:00", 0, or 0.0.
    end: str
        The end of the range, e.g. "1990-09-04T00:00", 1, or 1.0.
    step: str
        The step size for the range, e.g. "PT3H", 1, or 1.0. If not given
        then the entire range will be selected.
    """

    start: Union[str, int, float]
    end: Union[str, int, float]
    step: Union[str, int, float] = None


@dataclass
class ValueSelection:
    """
    Defines a selection on the coordinate values of a variable, the
    `values` attribute can either be a list of values to select or a
    `Range` object to select a range of values. This is used to create
    a slice object for the selection. Optionally, the `units` attribute can be
    used to specify the units of the values which will used to ensure that
    the `units` attribute of the variable has the same value.

    Attributes:
        values: The values to select.
        units: The units of the values.
    """

    values: Union[List[Union[float, int]], Range]
    units: str = None


@dataclass
class DimMapping:
    """
    Defines the process for mapping dimensions and variables from an input
    dataset to a single new dimension (as in dimension in the
    output dataset of the dataset generation).

    There are three methods implemented for mapping:
    - "rename":
        Renames a dimension in the dataset to a new name.

        E.g. adding a dim-mapping as `{"time": {"method": "rename", "dim": "analysis_time"}}`
        will rename the "analysis_time" dimension in the input dataset to "time" dimension in the output.

    - "stack_variables_by_var_name":
        Stacks all variables along a new dimension that is mapped to the output dimensions name given.

        E.g. adding a dim-mapping as
        `{"state_feature": {"method": "stack_variables_by_var_name", "name_format": "{var_name}{altitude}m", dims: [altitude]}}`
        will stack all variables in the input dataset along the "state_feature" dimension in the output
        and the coordinate values will be given as f"{var_name}{altitude}m" where `var_name` is the name
        of the variable and `altitude` is the value of the "altitude" coordinate.
        If any dimensions are specified in the `dims` attribute, then the these dimensions will
        also be stacked into this new dimension, and the `name_format` attribute can be used to
        use the coordinate values from the stacked dimensions in the new coordinate values.

    - "stack":
        Stacks the provided coordinates and maps the result to the output dimension.

        E.g. `{"grid_index": {"method": "stack", "dims": ["x", "y"]}}` will stack the "x" and "y"
        dimensions in the input dataset into a new "grid_index" dimension in the output.

    Attributes:
        method: The method used for mapping.
        dims: The dimensions to be mapped.
        name_format: The format for naming the mapped dimensions.

    Attributes
    ----------
    method: str
        The method used for mapping. The options are:
        - "rename": Renames a dimension in the dataset to a new name.
        - "stack_variables_by_var_name": Stacks all variables along a new dimension that is mapped to the output dimensions name given.
        - "stack": Stacks the provided coordinates and maps the result to the output dimension.
    dims: List[str]
        The dimensions to be mapped when using the "stack" or "stack_variables_by_var_name" methods.
    dim: str
        The dimension to be renamed when using the "rename" method.
    name_format: str
        The format for naming the mapped dimensions when using the "stack_variables_by_var_name" method.
    """

    method: str
    dims: Optional[List[str]] = None
    dim: Optional[str] = None
    name_format: str = field(default=None)


@dataclass
class InputDataset:
    """
    Definition of a single input dataset which will be mapped to one the
    variables that have been defined as input variables for model architecture
    being targeted by the dataset.
    The definition for a input dataset includes setting
        1) the path to the dataset,
        2) the expected dimensions of the dataset,
        3) the variables to select from the dataset (and optionally subsection
           along the coordinates for each variable) and finally
        4) the method by which the dimensions and variables of the dataset are
           mapped to the dimensions of the architecture's input variables (this
           includes stacking of all the selected variables into a new single
           variable along a new coordinate, and may include renaming and
           stacking dimensions existing dimensions).

    Attributes
    ----------
    path: str
        Path to the dataset, e.g. the path to a zarr dataset or netCDF file.
        This can be anything that can be passed to `xarray.open_dataset`
    dims: List[str]
        List of the expected dimensions of the dataset. E.g. `["time", "x", "y"]`.
        These will be checked to ensure consistency of the dataset being read.
    variables: Union[List[str], Dict[str, Dict[str, ValueSelection]]]
        List of the variables to select from the dataset. E.g. `["temperature", "precipitation"]`
        or a dictionary where the keys are the variable names and the values are dictionaries
        defining the selection for each variable. E.g. `{"temperature": levels: {"values": [1000, 950, 900]}}`
        would select the "temperature" variable and only the levels 1000, 950, and 900.
    dim_mapping: Dict[str, DimMapping]
        Mapping of the variables and dimensions in the input dataset to the dimensions of the
        architecture's input variables. The key is the name of the architecture dimension to map to
        and the ´DimMapping´ describes how to map the dimensions and variables of the input dataset
        to this input dimension for the architecture.
    target_architecture_variable: str
        The name of the variable in the architecture that this dataset is intended to map to. If multiple
        datasets map to the same variable, then the data from all datasets will be concatenated along the
        dimension that isn't shared (e.g. two datasets that coincide in space and time will only differ
        in the feature dimension, so the two will be combined by concatenating along the feature dimension).
        If a single shared coordinate cannot be found then an exception will be raised.
    """

    path: str
    dims: List[str]
    variables: Union[List[str], Dict[str, Dict[str, ValueSelection]]]
    dim_mapping: Dict[str, DimMapping]
    target_architecture_variable: str
    attributes: Dict[str, Any] = None
    target_architecture_variable: str


@dataclass
class Architecture:
    """
    Information about the model architecture this dataset is intended for. This
    covers defining what input variables the architecture expects (and the dimensions of each),
    the expected value range for each coordinate, and the chunking information for each dimension.

    Attributes
    ----------
    input_variables: Dict[str, List[str]]
        Defines the input variables for model architecture. The keys are the
        variable names to create and the values are lists of the dimensions. E.g.
        `{"static": ["grid_index", "feature"], "state": ["time", "grid_index", "state_feature"]}`.
        would define that the architecture expects a variable named "static" with
        dimensions "grid_index" and "feature" and a variable named "state" with
        dimensions "time", "grid_index", and "state_feature".

    input_coord_ranges: Dict[str, Range]
        Defines the expected value range for each coordinate. The keys are the
        name of the coordinate and the values are the range, e.g.
        `{"time": {"start": "1990-09-03T00:00", "end": "1990-09-04T00:00", "step": "PT3H"}}`
        would define that the "time" coordinate should have values between
        "1990-09-03T00:00" and "1990-09-04T00:00" with a step size of 3 hours.
        These range definitions are both used to ensure that the input dataset
        has the expected range and to select the correct values from the input
        dataset. If not given then the entire range will be selected.

    chunking: Dict[str, int]
        Defines the chunking information for each dimension. The keys are the
        names of the dimensions and the values are the chunk size for that dimension.
        If chunking is not specified for a dimension, then the entire dimension
        will be a single chunk.
    """

    input_variables: Dict[str, List[str]]
    input_coord_ranges: Dict[str, Range] = None
    chunking: Dict[str, int] = None


@dataclass
class Config(dataclass_wizard.YAMLWizard):
    """Configuration for the model.

    Attributes:
        schema_version: Version of the config file schema.
        dataset_version: Version of the dataset itself.
        architecture: Information about the model architecture this dataset is intended for.
        inputs: Input datasets for the model.

    Attributes
    ----------
    architecture: Architecture
        Information about the model architecture this dataset is intended for. This
        covers defining what input variables the architecture expects (and the dimensions of each),
        the expected value range for each coordinate, and the chunking information for each dimension.
    inputs: Dict[str, InputDataset]
        Input datasets for the model. The keys are the names of the datasets and the values are
        the input dataset configurations.
    schema_version: str
        Version string for the config file schema.
    dataset_version: str
        Version string for the dataset itself.
    """

    architecture: Architecture
    inputs: Dict[str, InputDataset]
    schema_version: str
    dataset_version: str


if __name__ == "__main__":
    config = Config.from_yaml_file("example.danra.yaml")
    import rich

    rich.print(config)
