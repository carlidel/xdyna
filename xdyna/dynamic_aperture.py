import time

import cv2
import numpy as np
from scipy.special import lambertw as W

from .ml import MLBorder


def find_largest_conglomerate(matrix):
    """Find the largest conglomerate of True values in a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be analyzed.

    Returns
    -------
    np.ndarray
        Matrix with only the largest conglomerate of True values.
    """

    def dfs(i, j, conglomerate):
        if 0 <= i < len(matrix) and 0 <= j < len(matrix[0]) and matrix[i][j]:
            matrix[i][j] = False
            conglomerate.append((i, j))
            dfs(i - 1, j, conglomerate)  # Up
            dfs(i + 1, j, conglomerate)  # Down
            dfs(i, j - 1, conglomerate)  # Left
            dfs(i, j + 1, conglomerate)  # Right

    if not matrix or not matrix[0]:
        return []

    max_conglomerate_size = 0
    largest_conglomerate = []

    for i, col in enumerate(matrix):
        for j, val in enumerate(col):
            if val:
                conglomerate = []
                dfs(i, j, conglomerate)
                if len(conglomerate) > max_conglomerate_size:
                    max_conglomerate_size = len(conglomerate)
                    largest_conglomerate = conglomerate

    if not largest_conglomerate:
        return []

    # Create a new matrix with only the largest conglomerate
    new_matrix = [[False] * len(matrix[0]) for _ in range(len(matrix))]
    for i, j in largest_conglomerate:
        new_matrix[i][j] = True

    return np.asarray(new_matrix)


def find_border_of_conglomerate(grid):
    """Finds the border of a conglomerate of True values in a grid. Uses the
    morphological methods in opencv.

    Parameters
    ----------
    grid : np.ndarray
        Grid to be analyzed.

    Returns
    -------
    np.ndarray
        Grid with only the border of the conglomerate.
    """
    # temporarely convert to uint8
    u8_grid = np.asarray(grid, dtype=np.uint8)

    # Perform dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated_matrix = cv2.dilate(u8_grid, kernel, iterations=1)

    # Perform erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded_matrix = cv2.erode(u8_grid, kernel, iterations=1)

    # Get the border by subtracting the eroded matrix from the dilated matrix
    border = np.asarray(np.logical_and(u8_grid, dilated_matrix - eroded_matrix))

    return border


# just formulas to evaluate DA for now...


def border_from_radial_samples(r_space, ang_space, survival_data, return_x_y=False):
    """Compute the dynamic aperture borderfrom radial samples. The radial samples
    are assumed to be in the format (r, theta) and flattened, as in the output
    one would obtain from initial_distributions.radial_distribution.

    Parameters
    ----------
    r_space : np.ndarray
        Unique radial samples.
    ang_space : np.ndarray
        Unique angular samples.
    survival_data : np.ndarray
        Boolean-like array with the survival information for each particle.
        True means that the particle survived.
    return_x_y : bool, optional
        Whether to return the x and y coordinates of the border, by default False

    Returns
    -------
    r_border : np.ndarray
        Radial coordinates of the DA border.
    x_border : np.ndarray
        X coordinates of the DA border. Only returned if return_x_y is True.
    y_border : np.ndarray
        Y coordinates of the DA border. Only returned if return_x_y is True.
    """
    idx_border = np.empty(ang_space.size, dtype=np.int64)
    for i in range(ang_space.size):
        idx = np.argmin(survival_data[:, i])
        if idx > 0:
            idx_border[i] = idx - 1
        else:
            if survival_data[0, i]:
                idx_border[i] = len(r_space) - 1

    r_border = np.array([r_space[idx] for idx in idx_border])
    x_border = r_border * np.cos(ang_space)
    y_border = r_border * np.sin(ang_space)

    if return_x_y:
        return r_border, x_border, y_border
    return r_border


def border_from_grid_samples(x_space, y_space, survival_data, return_x_y=False):
    """Compute the dynamic aperture border from grid samples. The grid samples
    are assumed to be in the format (x, y) and flattened, as in the output
    one would obtain from initial_distributions.grid_distribution.

    Parameters
    ----------
    x_space : np.ndarray
        Unique x samples.
    y_space : np.ndarray
        Unique y samples.
    survival_data : np.ndarray
        Boolean-like array with the survival information for each particle.
        True means that the particle survived.
    return_x_y : bool, optional
        Whether to return the x and y coordinates of the border, by default False

    Returns
    -------
    ang_border : np.ndarray
        Angular coordinates of the DA border.
    r_border : np.ndarray
        Radial coordinates of the DA border.
    x_border : np.ndarray
        X coordinates of the DA border. Only returned if return_x_y is True.
    y_border : np.ndarray
        Y coordinates of the DA border. Only returned if return_x_y is True.
    """
    border = find_border_of_conglomerate(find_largest_conglomerate(survival_data))

    # get_idx list of indices of the border
    x_idx, y_idx = np.where(border)
    # get the corresponding x and y values
    x_border = np.array([x_space[idx] for idx in x_idx])
    y_border = np.array([y_space[idx] for idx in y_idx])
    # get the corresponding r and ang values
    r_border = np.sqrt(x_border**2 + y_border**2)
    ang_border = np.arctan2(y_border, x_border)

    # zip and sort the values by angle
    border = sorted(zip(ang_border, r_border, x_border, y_border), key=lambda x: x[0])
    # unzip the values
    ang_border, r_border, x_border, y_border = zip(*border)

    if return_x_y:
        return ang_border, r_border, x_border, y_border
    return ang_border, r_border


def border_from_random_samples(
    x_data,
    y_data,
    survival_data,
    return_x_y=False,
    mirror_data=False,
    memory_threshold=1e9,
):
    """Compute the dynamic aperture border from random samples. The random samples
    are passed to the MLBorder class, which uses a Support Vector Machine to
    classify the samples as inside or outside the DA and draw the border.

    Note that currently the MLBorder class expects the samples to be taken at
    all angles, including angles above np.pi/2 and below 0. If your data is
    taken only in the range [0, np.pi/2] (i.e. only in the first quadrant), you
    can use mirror_data=True to mirror the data in the other quadrants.

    Parameters
    ----------
    x_data : np.ndarray
        x samples.
    y_data : np.ndarray
        y samples.
    survival_data : np.ndarray
        Boolean-like array with the survival information for each particle.
        True means that the particle survived.
    return_x_y : bool, optional
        Whether to return the x and y coordinates of the border, by default False
    mirror_data : bool, optional
        Whether to mirror the data in the other quadrants, by default False
    memory_threshold : float, optional
        Threshold to be used for the DA evaluation, by default 1e9

    Returns
    -------
    ang_border : np.ndarray
        Angular coordinates of the DA border.
    r_border : np.ndarray
        Radial coordinates of the DA border.
    x_border : np.ndarray
        X coordinates of the DA border. Only returned if return_x_y is True.
    y_border : np.ndarray
        Y coordinates of the DA border. Only returned if return_x_y is True.
    """
    if mirror_data:
        x_data = np.concatenate((x_data, -x_data, x_data, -x_data))
        y_data = np.concatenate((y_data, y_data, -y_data, -y_data))
        survival_data = np.concatenate(
            (survival_data, survival_data, survival_data, survival_data)
        )

    prev = time.process_time()
    print(f"Start SVM evaluation...", end="")
    ML = MLBorder(x_data, y_data, survival_data, memory_threshold=memory_threshold)
    ML.fit(50)
    ML.evaluate(0.5)
    min_border_x, min_border_y = ML.border
    print(f"done (in {round(time.process_time()-prev,2)} seconds).")

    # get the corresponding r and ang values
    r_border = np.sqrt(min_border_x**2 + min_border_y**2)
    ang_border = np.arctan2(min_border_y, min_border_x)

    # zip and sort the values by angle
    border = sorted(
        zip(ang_border, r_border, min_border_x, min_border_y), key=lambda x: x[0]
    )
    # unzip the values
    ang_border, r_border, min_border_x, min_border_y = zip(*border)

    if return_x_y:
        return ang_border, r_border, min_border_x, min_border_y
    return ang_border, r_border


# Open border interpolation
# --------------------------------------------------------
def trapz(x, y, xrange):
    """
    Return the integral using the trapezoidal rule for open border.
    Works for not constant step too.

    Parameters
    ----------
    x : np.ndarray
        Array of the x values.
    y : np.ndarray
        Array of the y values.
    xrange : tuple
        Tuple with the lower and upper limits of the integration range.

    Returns
    -------
    float
        Integral using the trapezoidal rule for open border.
    """
    x = np.array(x)
    y = np.array(y)
    sort = np.argsort(x)
    x = x[sort]
    y = y[sort]
    #     D=integrate.trapezoid(x=x*np.pi/180, y=np.ones(x.size))
    #     return np.sqrt( 2/np.pi*integrate.trapezoid(x=x*np.pi/180, y=y**2) )
    #     return np.sqrt( integrate.trapezoid(x=x*np.pi/180, y=y**2)/D )
    #     return integrate.trapezoid(x=x*np.pi/180, y=y)/D

    res = y[0] * (x[0] - xrange[0]) + y[-1] * (
        xrange[1] - x[-1]
    )  # Lower and upper open border schema
    res += (0.5) * ((y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum()  # Close border schema
    return res


def simpson(x, y, xrange):
    """
    Return the integral using the simpson's 1/3 rule for open border.
    Works for not constant step too.

    Parameters
    ----------
    x : np.ndarray
        Array of the x values.
    y : np.ndarray
        Array of the y values.
    xrange : tuple
        Tuple with the lower and upper limits of the integration range.

    Returns
    -------
    float
        Integral using the trapezoidal rule for open border.
    """
    if len(y) >= 3 and (len(y) % 2) == 1:
        x = np.array(x)
        y = np.array(y)
        sort = np.argsort(x)
        x = x[sort]
        y = y[sort]

        res = (
            (23 * y[0] - 16 * y[1] + 5 * y[2]) * (x[0] - xrange[0]) / 12
        )  # Lower open border schema
        res += (
            (23 * y[-1] - 16 * y[-2] + 5 * y[-3]) * (xrange[1] - x[-1]) / 12
        )  # Upper open border schema

        # Constant stepsize
        #         res+= ( (y[0:-1:2]+4*y[1::2]+y[2::2])*(x[2::2] - x[0:-1:2]) ).sum()/6     # Close border schema

        # Different stepsize
        h1 = x[1::2] - x[0:-1:2]
        h2 = x[2::2] - x[1::2]
        res += (
            (
                y[0:-1:2] * h2 * (2 * h1**2 - h2 * (h2 - h1))
                + y[1::2] * (h1 + h2) ** 3
                + y[2::2] * h1 * (2 * h2**2 + h1 * (h2 - h1))
            )
            / (6 * h1 * h2)
        ).sum()

        return res
    else:
        return 0


def alter_simpson(x, y, xrange):  # used to be called compute_da
    """
    Return the integral using the alternative simpson rule for open border.
    Does not works for not constant step.

    Parameters
    ----------
    x : np.ndarray
        Array of the x values.
    y : np.ndarray
        Array of the y values.
    xrange : tuple
        Tuple with the lower and upper limits of the integration range.

    Returns
    -------
    float
        Integral using the trapezoidal rule for open border.
    """
    if len(y) > 6:
        x = np.array(x)
        y = np.array(y)
        sort = np.argsort(x)
        x = x[sort]
        y = y[sort]

        res = (
            (23 * y[0] - 16 * y[1] + 5 * y[2]) * (x[0] - xrange[0]) / 12
        )  # Lower open border schema
        res += (
            (23 * y[-1] - 16 * y[-2] + 5 * y[-3]) * (xrange[1] - x[-1]) / 12
        )  # Upper open border schema
        wght = np.ones(len(y))
        wght[0] = wght[-1] = 3 / 8
        wght[1] = wght[-2] = -16 / 12
        wght[2] = wght[-3] = 5 / 12
        res += (y * wght).sum() * (x[1] - x[0])  # Close border schema
        return res
    else:
        return 0


# Compute average DA
# --------------------------------------------------------
def compute_da_1d(x, y, xrange, interp=trapz):  # used to be called compute_da
    """
    Return the arithmetic average. Default interpolator: trapz.

    Parameters
    ----------
    x : np.ndarray
        Array of the x values.
    y : np.ndarray
        Array of the y values.
    xrange : tuple
        Tuple with the lower and upper limits of the integration range.
    interp : function, optional
        Interpolator to be used for the integration, by default trapz.

    Returns
    -------
    float
        Arithmetic average.
    """
    return interp(x, y, xrange) / (xrange[1] - xrange[0])


def compute_da_2d(x, y, xrange, interp=trapz):
    """
    Return the quadratic average. Default interpolator: trapz.

    Parameters
    ----------
    x : np.ndarray
        Array of the x values.
    y : np.ndarray
        Array of the y values.
    xrange : tuple
        Tuple with the lower and upper limits of the integration range.
    interp : function, optional
        Interpolator to be used for the integration, by default trapz.

    Returns
    -------
    float
        Quadratic average.
    """
    return np.sqrt(interp(x, y**2, xrange) / (xrange[1] - xrange[0]))


def compute_da_4d(x, y, xrange, interp=trapz):
    """
    Return the 4D average. Default interpolator: trapz.

    Parameters
    ----------
    x : np.ndarray
        Array of the x values.
    y : np.ndarray
        Array of the y values.
    xrange : tuple
        Tuple with the lower and upper limits of the integration range.
    interp : function, optional
        Interpolator to be used for the integration, by default trapz.

    Returns
    -------
    float
        4D average.
    """
    return interp(x, (y**4) * np.sin(2 * x), xrange) ** (1 / 4)


# --------------------------------------------------------
def is_list_of_lists_or_ndarrays(variable):
    if isinstance(variable, list):
        if all(isinstance(item, (list, np.ndarray)) for item in variable):
            return True
    return False


def da_from_border(
    ang_border,
    r_border,
    interp_order="1D",
    interp_method="trapz",
    min_angle=None,
    max_angle=None,
    return_dict=False,
):
    """Compute the dynamic aperture from border coordinates.

    Parameters
    ----------
    ang_border : np.ndarray or list of np.ndarray
        Angular coordinates of the DA border.
    r_border : np.ndarray or list of np.ndarray
        Radial coordinates of the DA border.
    interp_order : str, optional
        Interpolation order to be used for the DA evaluation, by default "1D".
        Available options: "1D", "2D", "4D".
    interp_method : str, optional
        Interpolation method to be used for the DA evaluation, by default "trapz".
        Available options: "trapz", "simpson", "alter_simpson".
    min_angle : float, optional
        Minimum angle to be used for the DA evaluation, by default None.
        If None, the minimum angle is the minimum angle in the border.
    max_angle : float, optional
        Maximum angle to be used for the DA evaluation, by default None.
        If None, the maximum angle is the maximum angle in the border.
    return_dict : bool, optional
        If True, many DA values are returned as a dictionary, by default False.
        If False, only the mean DA is returned.

    Returns
    -------
    if return_dict is False:
        float
            Mean dynamic aperture. If ang_border and r_border are lists, a ndarray
            of mean dynamic apertures is returned.

    if return_dict is True:
        dict
            mean: Mean dynamic aperture. Or ndarray of mean dynamic apertures.
            min: Minimum dynamic aperture. Or ndarray of minimum dynamic apertures.
            max: Maximum dynamic aperture. Or ndarray of maximum dynamic apertures.
    """

    if interp_order == "1D":
        compute_da = compute_da_1d
    elif interp_order == "2D":
        compute_da = compute_da_2d
    elif interp_order == "4D":
        compute_da = compute_da_4d
    else:
        raise ValueError("Invalid interpolation order.")

    if interp_method == "trapz":
        interp = trapz
    elif interp_method == "simpson":
        interp = simpson
    elif interp_method == "alter_simpson":
        interp = alter_simpson
    else:
        raise ValueError("Invalid interpolation method.")

    if min_angle is None:
        min_angle = ang_border.min()
    if max_angle is None:
        max_angle = ang_border.max()

    if is_list_of_lists_or_ndarrays(ang_border):
        if not return_dict:
            mean_da = [
                da_from_border(
                    ang_border[i],
                    r_border[i],
                    interp_order=interp_order,
                    interp_method=interp_method,
                    min_angle=min_angle,
                    max_angle=max_angle,
                    return_dict=return_dict,
                )
                for i in range(len(ang_border))
            ]
            return np.array(mean_da)
        else:
            dict_list = [
                da_from_border(
                    ang_border[i],
                    r_border[i],
                    interp_order=interp_order,
                    interp_method=interp_method,
                    min_angle=min_angle,
                    max_angle=max_angle,
                    return_dict=return_dict,
                )
                for i in range(len(ang_border))
            ]
            merged_dict = {}
            for key in dict_list[0].keys():
                merged_dict[key] = np.array([d[key] for d in dict_list])
            return merged_dict

    argsort = np.argsort(ang_border)
    ang_border = ang_border[argsort]
    r_border = r_border[argsort]

    mean_da = compute_da(
        ang_border[(ang_border >= min_angle) & (ang_border <= max_angle)],
        r_border[(ang_border >= min_angle) & (ang_border <= max_angle)],
        (min_angle, max_angle),
        interp=interp,
    )

    if return_dict:
        dict(mean=mean_da, min=r_border.min(), max=r_border.max())
    return mean_da


# DA vs Turns models
# --------------------------------------------------------
# Taken from https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.22.104003
def model_2(N, rho=1, K=1, N0=1):  # Eq. 20
    return rho * (K / (2 * np.exp(1) * np.log(N / N0))) ** K


model_2_default = {"rho": 1, "K": 1, "N0": 1}
model_2_boundary = {"rho": [1e-10, np.inf], "K": [0.01, 2], "N0": [1, np.inf]}


def model_2b(N, btilde=1, K=1, N0=1, B=1):  # Eq. 35a
    return btilde / (B * np.log(N / N0)) ** K


model_2b_default = {"btilde": 1, "K": 1, "N0": 1, "B": 1}
model_2b_boundary = {
    "btilde": [1e-10, np.inf],
    "K": [0.01, 2],
    "N0": [1, np.inf],
    "B": [1e-10, 1e5],
}


def model_2n(N, b=1, K=1, N0=1):  # Eq. 2 from Frederik
    return b / (np.log(N / N0)) ** K


model_2n_default = {"b": 1, "K": 1, "N0": 1}
model_2n_boundary = {"b": [1e-10, np.inf], "K": [0.01, 2], "N0": [1, np.inf]}


def model_4(N, rho=1, K=1, lmbd=0.5):  # Eq. 23
    return (
        rho
        / (
            -(2 * np.exp(1) * lmbd)
            * np.real(
                W(
                    (-1 / (2 * np.exp(1) * lmbd))
                    * (rho / 6) ** (1 / K)
                    * (8 * N / 7) ** (-1 / (lmbd * K)),
                    k=-1,
                )
            )
        )
        ** K
    )


model_4_default = {"rho": 1, "K": 1, "lmbd": 0.5}
model_4_boundary = {"rho": [1e-10, 1e10], "K": [0.01, 2], "lmbd": [1e-10, 1e10]}


def model_4b(N, btilde=1, K=1, N0=1, B=1):  # Eq. 35c
    return (
        btilde
        / (-(0.5 * K * B) * np.real(W((-2 / (K * B)) * (N / N0) ** (-2 / K), k=-1)))
        ** K
    )


model_4b_default = {"btilde": 1, "K": 1, "N0": 1, "B": 1}
model_4b_boundary = {
    "btilde": [1e-10, np.inf],
    "K": [0.01, 2],
    "N0": [1, np.inf],
    "B": [1e-10, 1e10],
}


def model_4n(N, rho=1, K=1, mu=1):  # Eq. 4 from Frederik
    return rho / (-np.real(W((mu * N) ** (-2 / K), k=-1))) ** K


model_4n_default = {"rho": 1, "K": 1, "mu": 1}
model_4n_boundary = {"rho": [1e-10, np.inf], "K": [0.01, 2], "mu": [1e-10, 1e10]}
# --------------------------------------------------------
