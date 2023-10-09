"""Provides the user with initial distribution functions to be used in the 
trackings, and provides help in saving the configurations chosen in a GenericWriter
object. The initial distributions can then be re-constructed from the saved
configuration parameters in the GenericWriter object.
"""
import os
from math import floor

# from scipy.constants import c as clight
import numpy as np
from numpy.random import default_rng

from .generic_writer import GenericWriter


def grid_distribution(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_num: float = None,
    y_num: float = None,
    x_step: float = None,
    y_step: float = None,
    out: GenericWriter = None,
    out_header: str = "dist_params",
):
    """Generate the initial conditions in a 2D X-Y grid.

    Parameters
    ----------
    x_min : float
        Minimum value of the horizontal coordinate.
    x_max : float
        Maximum value of the horizontal coordinate.
    y_min : float
        Minimum value of the vertical coordinate.
    y_max : float
        Maximum value of the vertical coordinate.
    x_num : int, optional
        Number of points in the horizontal coordinate, by default None, either
        this or x_step must be specified.
    y_num : int, optional
        Number of points in the vertical coordinate, by default None, either
        this or y_step must be specified.
    x_step : float, optional
        Step in the horizontal coordinate, by default None, either this or
        x_num must be specified.
    y_step : float, optional
        Step in the vertical coordinate, by default None, either this or
        y_num must be specified.
    out : GenericWriter, optional
        Writer object to store the distribution parameters, by default None.
        If None, the parameters are not stored.
    out_header : str, optional
        Header to be used inside the writer object for the distribution parameters,
        by default "dist_params".

    Returns
    -------
    x : np.ndarray
        Array of the horizontal coordinates. Flattened.
    y : np.ndarray
        Array of the vertical coordinates. Flattened.
    """

    # Make the grid in xy
    def check_options(coord_min, coord_max, coord_step, coord_num, plane):
        if coord_step is None and coord_num is None:
            raise ValueError(f"Specify at least '{plane}_step' or '{plane}_num'.")
        elif coord_step is not None and coord_num is not None:
            raise ValueError(
                f"Use only one of '{plane}_step' and '{plane}_num', not both."
            )
        elif coord_step is not None:
            coord_num = floor((coord_max - coord_min) / coord_step) + 1
            coord_max = coord_min + (coord_num - 1) * coord_step
        return coord_min, coord_max, coord_num

    x_min, x_max, x_num = check_options(x_min, x_max, x_step, x_num, "x")
    y_min, y_max, y_num = check_options(y_min, y_max, y_step, y_num, "y")

    x_space = np.linspace(x_min, x_max, x_num)
    y_space = np.linspace(y_min, y_max, y_num)

    # Make all combinations
    x, y = np.array(np.meshgrid(x_space, y_space)).reshape(2, -1)

    # Save data if requested
    if out is not None:
        out.write_data(os.path.join(out_header, "dist_type"), "grid")
        out.write_data(os.path.join(out_header, "x_min"), x_min)
        out.write_data(os.path.join(out_header, "x_max"), x_max)
        out.write_data(os.path.join(out_header, "x_num"), x_num)
        out.write_data(os.path.join(out_header, "y_min"), y_min)
        out.write_data(os.path.join(out_header, "y_max"), y_max)
        out.write_data(os.path.join(out_header, "y_num"), y_num)
        out.write_data(os.path.join(out_header, "x_space"), x_space)
        out.write_data(os.path.join(out_header, "y_space"), y_space)

    return x, y


def radial_distribution(
    num_angles,
    r_min,
    r_max,
    r_num=None,
    r_step=None,
    ang_min=0.0,
    ang_max=np.pi / 4,
    open_border=True,
    out: GenericWriter = None,
    out_header: str = "dist_params",
    return_r_ang=False,
):
    """Generate the initial conditions in a 2D polar grid.

    Parameters
    ----------
    num_angles : int
        Number of angles to be used in the radial grid.
    r_min : float
        Minimum value of the radial coordinate.
    r_max : float
        Maximum value of the radial coordinate.
    r_num : int, optional
        Number of points in the radial coordinate, by default None, either
        this or r_step must be specified.
    r_step : float, optional
        Step in the radial coordinate, by default None, either this or
        r_num must be specified.
    ang_min : float, optional
        Minimum value of the angle coordinate, by default 0.0. (in radians)
    ang_max : float, optional
        Maximum value of the angle coordinate, by default np.pi/4. (in radians)
    open_border : bool, optional
        If True, the 1st and last angles will be ang_min+ang_step and ang_max-ang_step respectively, by default True.
    out : GenericWriter, optional
        Writer object to store the distribution parameters, by default None.
        If None, the parameters are not stored.
    out_header : str, optional
        Header to be used inside the writer object for the distribution parameters,
        by default "dist_params".
    return_r_ang : bool, optional
        If True, the r and ang coordinates are returned after x and y, by default
        False.

    Returns
    -------
    x : np.ndarray
        Array of the horizontal coordinates. Flattened.
    y : np.ndarray
        Array of the vertical coordinates. Flattened.
    r : np.ndarray
        Array of the radial coordinates. Flattened. Only returned if return_r_ang is True.
    ang : np.ndarray
        Array of the angular coordinates. Flattened. Only returned if return_r_ang is True.
    """

    # Make the grid in r
    if r_step is None and r_num is None:
        raise ValueError("Specify at least 'r_step' or 'r_num'.")
    elif r_step is not None and r_num is not None:
        raise ValueError("Use only one of 'r_step' and 'r_num', not both.")
    elif r_step is not None:
        r_num = floor((r_max - r_min) / r_step) + 1
        r_max = r_min + (r_num - 1) * r_step
    r_space = np.linspace(r_min, r_max, r_num)
    # Make the grid in angles
    ang_step = (ang_max - ang_min) / (num_angles + 1)
    ang_space = np.linspace(
        ang_min + ang_step * open_border, ang_max - ang_step * open_border, num_angles
    )

    r, ang = np.array(np.meshgrid(r_space, ang_space)).reshape(2, -1)

    # Get the normalised coordinates
    x = r * np.cos(ang)
    y = r * np.sin(ang)

    # Save data if requested
    if out is not None:
        out.write_data(os.path.join(out_header, "dist_type"), "radial")
        out.write_data(os.path.join(out_header, "r_min"), r_min)
        out.write_data(os.path.join(out_header, "r_max"), r_max)
        out.write_data(os.path.join(out_header, "r_num"), r_num)
        out.write_data(os.path.join(out_header, "ang_min"), ang_min)
        out.write_data(os.path.join(out_header, "ang_max"), ang_max)
        out.write_data(os.path.join(out_header, "num_angles"), num_angles)
        out.write_data(os.path.join(out_header, "r_space"), r_space)
        out.write_data(os.path.join(out_header, "ang_space"), ang_space)

    if return_r_ang:
        return x, y, r, ang
    return x, y


def random_distribution(
    num_part=1000,
    r_min=0.0,
    r_max=25.0,
    ang_min=0.0,
    ang_max=np.pi / 4,
    out: GenericWriter = None,
    out_header: str = "dist_params",
    rng=None,
):
    """Generate the initial conditions in a 2D random grid.

    Parameters
    ----------
    num_part : int, optional
        Number of particles to be generated, by default 1000.
    r_min : float, optional
        Minimum value of the radial coordinate, by default 0.0.
    r_max : float, optional
        Maximum value of the radial coordinate, by default 25.
    ang_min : float, optional
        Minimum value of the angle coordinate, by default 0.0. (in radians)
    ang_max : float, optional
        Maximum value of the angle coordinate, by default np.pi/4. (in radians)
    out : GenericWriter, optional
        Writer object to store the distribution parameters, by default None.
        If None, the parameters are not stored.
    out_header : str, optional
        Header to be used inside the writer object for the distribution parameters,
        by default "dist_params".
    rng : np.random.Generator, optional
        Random number generator object, by default None. If None, a new one is created.

    Returns
    -------
    x : np.ndarray
        Array of the horizontal coordinates. Flattened.
    y : np.ndarray
        Array of the vertical coordinates. Flattened.
    """

    # Make the data
    if rng is None:
        rng = default_rng()

    r = rng.uniform(low=r_min**2, high=r_max**2, size=num_part)
    th = rng.uniform(low=ang_min, high=ang_max, size=num_part)
    r = np.sqrt(r)
    x = r * np.cos(th)
    y = r * np.sin(th)

    # Save data if requested
    if out is not None:
        out.write_data(os.path.join(out_header, "dist_type"), "random")
        out.write_data(os.path.join(out_header, "num_part"), num_part)
        out.write_data(os.path.join(out_header, "r_min"), r_min)
        out.write_data(os.path.join(out_header, "r_max"), r_max)
        out.write_data(os.path.join(out_header, "ang_min"), ang_min)
        out.write_data(os.path.join(out_header, "ang_max"), ang_max)

    return x, y
