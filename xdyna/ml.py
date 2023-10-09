from time import process_time

import numpy as np
import pandas as pd
import scipy as sp
from skimage import measure
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# try:
#     # Rapids is scikit-learn on gpu's
#     # download via conda; see https://rapids.ai/start.html#get-rapids for info
#     from cuml.svm import SVC
# except ImportError:
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

from .geometry import in_polygon_2D

# pip install scikit-learn-intelex
# try:
#     from sklearnex import patch_sklearn #, unpatch_sklearn
#     patch_sklearn()
# except ImportError:
#     pass


# TODO: catch if border does not have enough points (i.e. goes out of range) if minimum is taken
# too small. Solution is to increase Nmin, or add a wider range of points


class MLBorder:
    def __init__(
        self,
        x,
        y,
        survival_data,
        memory_threshold=1e9,
        cv=None,
        cv_matrix=None,
        margin=0.2,
        min_samples_per_label=100,
        try_to_balance_input=True,
    ):
        """Initialize the MLBorder object.

        Parameters
        ----------
        x : np.ndarray
            Horizontal coordinates of the input data.
        y : np.ndarray
            Vertical coordinates of the input data.
        survival_data : np.ndarray
            Boolean array indicating which particles survived.
        memory_threshold : int, optional
            Maximum memory to be used by the ML model, in bytes. Default is 1e9.
        """
        # self._turn = at_turn
        self._memory_threshold = memory_threshold
        self._input_time = 0
        self._fit_time = 0
        self._evaluation_time = 0
        # self._cv_matrix = None  # Needed to make set_input_data work correctly

        self.set_input_data(
            x, y, survival_data, margin, min_samples_per_label, try_to_balance_input
        )

        # If a cv_matrix is given, it sets the number of CV splits and constructs a model
        self._cv_matrix = cv_matrix
        if cv_matrix is not None:
            self._cv = self._get_cv_from_matrix()
            if cv is not None and self._cv != cv:
                print(
                    f"Warning: argument {cv=} in MLBorder constructor does not match cv from 'cv_matrix' "
                    + f"({self._cv}). Ignored the former."
                )
        elif cv is None:
            self._cv = 10
        else:
            self._cv = cv
        self._update_ml()

    # TODO: improve scoring function, because now for some splits it clips at 1.
    def fit(self, iterations=50, *, cv=None):
        if not self.ml_possible:
            return
        start_time = process_time()
        previous_best = self.best_estimator_
        if cv is not None:
            if self.cv_matrix is None:
                self._cv = cv
            elif self._cv != cv:
                print(
                    f"Warning: argument {cv=} in 'fit()' does not match cv from 'cv_matrix' ({self._cv}). "
                    + "Ignored the former."
                )
        svc = SVC(
            kernel="rbf",
            decision_function_shape="ovr",
            class_weight="balanced",
            cache_size=self.memory_threshold / 1e6,
        )
        svc_pipe = make_pipeline(StandardScaler(), svc)
        svc_param_grid = {
            "svc__C": loguniform(1e0, 1e5),
            "svc__gamma": loguniform(1e-2, 1e3),
        }
        clf = RandomizedSearchCV(
            svc_pipe, svc_param_grid, n_iter=iterations, n_jobs=-1, cv=self._cv
        )
        clf.fit(self._input_data, self._labels)
        self._update_cv_matrix(cv_result=clf.cv_results_, n_iter=iterations)
        self._fit_time += process_time() - start_time

        # Fitting potentially invalidates a previous border
        if previous_best != self.best_estimator_:
            self._border_x = None
            self._border_y = None
            self._border_res = None
            self._volume = None

    def _get_cv_from_matrix(self):
        return sum([1 if "score_" in str else 0 for str in self.cv_matrix.index]) - 4

    def _update_cv_matrix(self, cv_result, n_iter):
        # Number of CV splits in the results
        cv = sum([1 if "split" in str else 0 for str in cv_result.keys()])

        # If we already have results from a previous scan, prepend them
        if self.cv_matrix is None:
            scores = ["score_" + str(i) for i in range(cv)]
            cv_matrix = pd.DataFrame(
                index=[
                    "C",
                    "gamma",
                    "score_min",
                    "score_mean",
                    "score_max",
                    "score_std",
                    *scores,
                    "fit_time",
                ]
            )
        else:
            # Check if previous runs were done with the same number of CV splits
            if cv != self._get_cv_from_matrix():
                raise ValueError(
                    f"Current result used {cv} CV splits, while previous run was done "
                    + f"with {previous_cv} CV splits. Cannot combine!"
                )
            cv_matrix = self.cv_matrix

        last_id = len(cv_matrix.columns) - 1
        this_cv_matrix = {}
        for this_id in range(n_iter):
            cv_scores = {
                "score_" + str(i): cv_result["split" + str(i) + "_test_score"][this_id]
                for i in range(cv)
            }
            this_cv_matrix[last_id + this_id + 1] = pd.Series(
                {
                    "C": cv_result["params"][this_id]["svc__C"],
                    "gamma": cv_result["params"][this_id]["svc__gamma"],
                    "score_min": min(cv_scores.values()),
                    "score_mean": cv_result["mean_test_score"][this_id],
                    "score_max": max(cv_scores.values()),
                    "score_std": cv_result["std_test_score"][this_id],
                    **cv_scores,
                    "fit_time": cv_result["mean_fit_time"][this_id] * cv,
                }
            )

        self._cv_matrix = pd.concat([cv_matrix, pd.DataFrame(this_cv_matrix)], axis=1)
        self._update_ml()

    def _find_optimal_parameters(self):
        if self._cv_matrix is None:
            return
        # Best score is chosen by the best mean
        best = self._cv_matrix.loc["score_mean"].idxmax()
        self._best_estimator_ = best
        self._C = self._cv_matrix[best]["C"]
        self._gamma = self._cv_matrix[best]["gamma"]

    def _update_ml(self):
        if self._cv_matrix is None or not self.ml_possible:
            self._svc = None
            self._svc_pipe = None
            self._predict = None
            self._C = None
            self._gamma = None
            self._best_estimator_ = None
            self._border_x = None
            self._border_y = None
            self._border_res = None
            self._volume = None
        else:
            self._find_optimal_parameters()
            self._svc = SVC(
                kernel="rbf", decision_function_shape="ovr", class_weight="balanced"
            )
            self._svc.C = self.C
            self._svc.gamma = self.gamma
            self._svc.cache_size = self.memory_threshold / 1e6
            self._svc_pipe = make_pipeline(StandardScaler(), self._svc)
            self._svc_pipe.fit(self._input_data, self.labels)

    def set_input_data(
        self,
        x,
        y,
        boolean_mask,
        margin=0.2,
        min_samples_per_label=100,
        try_to_balance_input=True,
    ):
        """Set the input data for the ML model.

        Parameters
        ----------
        x : np.ndarray
            Horizontal coordinates of the input data.
        y : np.ndarray
            Vertical coordinates of the input data.
        boolean_mask : np.ndarray
            Boolean array indicating which particles survived.
        margin : float, optional
            Margin to be added to the extra data, in units of the input data.
            Default is 0.2.
        min_samples_per_label : int, optional
            Minimum number of samples per label. Default is 100.
        try_to_balance_input : bool, optional
            If True, the input data will be balanced by removing samples from the
            category with more samples. Default is True.
        """
        start_time = process_time()
        # Setting new input data undoes previous fits
        if "_cv_matrix" in self.__dict__:  # Check if attribute exists
            print("Warning: Removed previous existing cv_matrix!")
        self._cv_matrix = None
        self._update_ml()

        # Make sure the data is in the correct format
        labels = np.zeros(boolean_mask.shape)
        labels[boolean_mask] = 1
        labels[boolean_mask] = 0

        # Define the two categories
        n_1 = np.count_nonzero(labels == 1)
        n_0 = np.count_nonzero(labels == 0)
        r_0 = np.sqrt(x[labels == 0] ** 2 + y[labels == 0] ** 2)
        r_1 = np.sqrt(x[labels == 1] ** 2 + y[labels == 1] ** 2)

        # Check if we have enough samples in each category
        self._ml_possible = True
        self._ml_impossible_reason = None
        self._extra_sample_r_region = []
        if min_samples_per_label >= len(labels) / 2:
            self._ml_possible = False
            self._ml_impossible_reason = -1
            self._extra_sample_r_region = [0, np.concatenate([r_0, r_1]).max()]
        if n_0 < min_samples_per_label:
            self._ml_possible = False
            self._ml_impossible_reason = 0
            self._extra_sample_r_region = [
                r_0.min() * (1 - margin),
                r_0.max() * (1 + margin),
            ]
        if n_1 < min_samples_per_label:
            self._ml_possible = False
            self._ml_impossible_reason = 1
            self._extra_sample_r_region = [0, r_1.max() * (1 + margin)]
        if not self.ml_possible:
            self._input_data = None
            self._labels = None
            return

        # Try to balance the samples:
        mask = np.full_like(labels, True, dtype=bool)
        if try_to_balance_input:
            step = 0.01
            r_upper = r_0.max()
            r_lower = r_1.min()

            # if n_0 > n_1:  by shrinking the 0-particles region radially,
            # stepwise from outside inwards
            while (
                n_0 > n_1 * (1 + margin)
                and r_upper >= r_1.max() * (1 + margin)
                and n_0 > min_samples_per_label
            ):
                mask = np.sqrt(x**2 + y**2) <= r_upper
                n_0 = len(labels[mask][labels[mask] == 0])
                # The following needs to be last, to avoid the (unlikely) case where
                # step would be larger
                # than the margin, and particles from the other category would be
                # accidentally removed
                r_upper -= step

            # if n_0 < n_1:  by shrinking the 1-particles region radially,
            # stepwise from inside outwards
            while (
                n_1 > n_0 * (1 + margin)
                and r_lower <= r_0.min() * (1 + margin)
                and n_1 > min_samples_per_label
            ):
                mask = np.sqrt(x**2 + y**2) >= r_lower
                n_1 = len(labels[mask][labels[mask] == 1])
                # Same as above
                r_lower += step

        # Mask the data and labels, and store it
        data = np.array([x, y]).T
        self._input_data = data[mask]
        self._labels = labels[mask]
        self._xmin = x[mask].min()
        self._xmax = x[mask].max()
        self._ymin = y[mask].min()
        self._ymax = y[mask].max()
        self._input_time += process_time() - start_time

    #         # To limit a bit the number of samples, and
    #         # to balance them a bit,
    #         # select a square region around the surviving points,
    #         # with a (by default) 20% margin on each side
    #         region_x = x[labels==1]
    #         region_y = y[labels==1]
    #         xmin, xmax = region_x.min(), region_x.max()
    #         ymin, ymax = region_y.min(), region_y.max()
    #         dx = xmax - xmin
    #         dy = ymax - ymin
    #         xmin -= margin*dx
    #         xmax += margin*dx
    #         ymin -= margin*dy
    #         ymax += margin*dy

    #         # Mask the data and labels, and store it
    #         data = np.array([x,y]).T
    #         mask = np.array([ xmin<=x<=xmax and ymin<=y<=ymax for x,y in data ])
    #         self._input_data = data[mask]
    #         self._labels = labels[mask]
    #         self._ymin = ymin
    #         self._ymax = ymax
    #         self._xmin = xmin
    #         self._xmax = xmax

    def evaluate(self, step_resolution=0.01):
        """Evaluate the border of the surviving particles. Creates a grid of points
        with the given resolution, and finds the contour predicted by the ML model.

        Parameters
        ----------
        step_resolution : float, optional
            Resolution of the grid to be used, in units of the input data. Default is 0.01.
        """

        if self._svc is None:
            raise ValueError("ML model not yet fitted. Do this first.")
        # No need to evaluate if border exists and resolution is the same
        if (
            self._border_x is not None
            and self._border_y is not None
            and self._border_res == step_resolution
        ):
            return
        start_time = process_time()
        x_grid = np.arange(self._xmin, self._xmax, step_resolution)
        y_grid = np.arange(self._ymin, self._ymax, step_resolution)
        actual_x_max = x_grid[-1]
        actual_y_max = y_grid[-1]
        len_x = len(x_grid)
        len_y = len(y_grid)
        xx, yy = np.meshgrid(x_grid, y_grid)
        predicted_data = self.predict(np.c_[xx.ravel(), yy.ravel()])
        predicted_data = predicted_data.reshape(xx.shape)

        # Then we find the contours of the data
        #
        # TODO: This whole hack to find the curve by using contour image analysis is not ideal.
        # We should look for smarter alternatives, or at least do the step_resolution iteratively
        contours = measure.find_contours(predicted_data, 0.5)

        # The contours are not in coordinates but in list indices;
        # we need to retransform them into coordinates
        def x_converter(x):
            return self._xmin + (x / (len_x - 1) * (actual_x_max - self._xmin))

        def y_converter(y):
            return self._ymin + (y / (len_y - 1) * (actual_y_max - self._ymin))

        contours_converted = [
            np.array([[x_converter(x), y_converter(y)] for y, x in contour])
            for contour in contours
        ]

        # If several contours are present (i.e. islands), we choose the one containing the origin
        contour_found = False
        for contour in contours_converted:
            contour = contour.T
            if in_polygon_2D(0, 0, contour[0], contour[1]):
                if contour_found:
                    raise RuntimeError(
                        "Several contours found around the origin! Please investigate."
                    )
                else:
                    border_x = contour[0]
                    border_y = contour[1]
                    contour_found = True
        if not contour_found:
            raise RuntimeError(
                "No contour found around the origin! Please investigate."
            )

        self._border_x = border_x
        self._border_y = border_y
        self._evaluation_time += process_time() - start_time
        self._volume = None

    def _calculate_volume(self, int_points=None):
        if self._border_x is None or self._border_y is None:
            return
        border, _ = sp.interpolate.splprep(self.border, s=0, per=1)
        if int_points is None:
            int_points = 50 * len(self._border_x)
        int_points = int(int_points)
        u = np.linspace(0, 1 - 1 / int_points, int_points) + 1 / 2 / int_points
        x_curv, _ = sp.interpolate.splev(u, border, der=0, ext=2)
        _, dy_curv = sp.interpolate.splev(u, border, der=1, ext=2)
        self._volume = (1 / int_points * x_curv * dy_curv).sum()

    @property
    def volume(self):
        if self._volume is None:
            self._calculate_volume()
        return self._volume

    @property
    def border(self):
        if self._border_x is None or self._border_y is None:
            raise ValueError("Border not yet evaluated. Do this first.")
        return np.array([self._border_x, self._border_y])

    @property
    def border_resolution(self):
        return self._border_res

    @property
    def memory_threshold(self):
        return self._memory_threshold

    @memory_threshold.setter
    def memory_threshold(self, val):
        self._memory_threshold = val
        if self._svc is not None:
            self._svc.cache_size = val / 1e6

    @property
    def time(self):
        return {
            "total": self._input_time + self._fit_time + self._evaluation_time,
            "input": self._input_time,
            "fit": self._fit_time,
            "evaluation": self._evaluation_time,
        }

    @property
    def input_data(self):
        return self._input_data

    @property
    def labels(self):
        return self._labels

    @property
    def n_samples(self):
        return None if self._labels is None else len(self._labels)

    @property
    def ml_possible(self):
        return self._ml_possible

    @property
    def ml_impossible_reason(self):
        return self._ml_impossible_reason

    @property
    def extra_sample_r_region(self):
        return self._extra_sample_r_region

    @property
    def cv(self):
        return self._cv

    @property
    def cv_matrix(self):
        return self._cv_matrix

    @property
    def predict(self):
        if self._svc is None:
            raise ValueError("ML model not yet fitted. Do this first.")
        else:
            return self._svc_pipe.predict

    @property
    def ml_iterations(self):
        return len(self.cv_matrix.columns)

    @property
    def C(self):
        return self._C

    @property
    def gamma(self):
        return self._gamma

    @property
    def best_estimator_(self):
        return self._best_estimator_
