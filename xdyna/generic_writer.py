import warnings
from abc import ABC, abstractmethod

import h5py
import numpy as np


class GenericWriter(ABC):
    @abstractmethod
    def __init__(self, filename, **kwargs):
        pass

    @abstractmethod
    def write_data(self, dataset_name: str, data: np.ndarray, overwrite=False):
        pass

    @abstractmethod
    def get_data(self, dataset_name: str):
        pass

    @abstractmethod
    def dataset_exists(self, dataset_name: str):
        pass

    @abstractmethod
    def get_storage_element(self):
        pass


class H5pyWriter(GenericWriter):
    """Class to write data to an HDF5 file."""

    def __init__(self, filename, compression=None):
        self.filename = filename
        self.compression = compression

    def _explore_h5py_group(self, group, prefix=""):
        for key, value in group.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            if isinstance(value, h5py.Group):
                yield from self._explore_h5py_group(value, new_prefix)
            else:
                yield new_prefix, value[()]

    def write_data(self, dataset_name: str, data: np.ndarray, overwrite=False):
        """Write data to an HDF5 file.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        data : np.ndarray
            Data to write
        overwrite : bool, optional
            If True, overwrite the dataset if it already exists, by default False, if set to "raise" raise an error if the dataset already exists
        """
        with h5py.File(self.filename, mode="a") as f:
            # check if dataset already exists
            if dataset_name in f:
                if overwrite:
                    del f[dataset_name]
                elif overwrite == "raise":
                    raise ValueError(
                        f"Dataset {dataset_name} already exists in file {self.filename}"
                    )
                else:
                    # just raise a warning and continue
                    warnings.warn(
                        f"Dataset {dataset_name} already exists in file {self.filename}"
                    )
                    return
            if self.compression is None:
                f.create_dataset(dataset_name, data=data)
            else:
                f.create_dataset(dataset_name, data=data, compression=self.compression)

    def get_data(self, dataset_name: str):
        """Get data from an HDF5 file.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        np.ndarray
            Data
        """
        with h5py.File(self.filename, mode="r") as f:
            # check if dataset exists
            if dataset_name not in f:
                raise ValueError(
                    f"Dataset {dataset_name} does not exist in file {self.filename}"
                )

            return f[dataset_name][:]

    def dataset_exists(self, dataset_name: str):
        """Check if a dataset exists in the HDF5 file.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        bool
            True if the dataset exists, False otherwise
        """
        with h5py.File(self.filename, mode="r") as f:
            return dataset_name in f

    def get_storage_element(self):
        warnings.warn(
            "Warning: you are getting a h5py.File object, remember to close it after use!"
        )
        return h5py.File(self.filename, mode="a")

    def convert_to_localwriter(self, filename=None):
        """Converts the H5pyWriter object to a LocalWriter object."""
        if filename is None:
            filename = self.filename

        # create a new LocalWriter object
        localwriter = LocalWriter(filename)

        for path, data in self._explore_h5py_group(h5py.File(self.filename, mode="r")):
            localwriter.write_data(path, data)

        return localwriter


class LocalWriter(GenericWriter):
    """Class to write data to a local python dictionary."""

    def _create_nested_dict(self, d, keys, data, overwrite=False):
        if len(keys) == 1:
            if keys[0] in d and not overwrite:
                raise ValueError(f"Dataset {keys[0]} already exists in dictionary {d}")
            else:
                d[keys[0]] = data
        else:
            if keys[0] not in d:
                d[keys[0]] = {}
            self._create_nested_dict(d[keys[0]], keys[1:], data, overwrite=overwrite)

    def _explore_nested_dict(self, d, prefix=""):
        for key, value in d.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                yield from self._explore_nested_dict(value, new_prefix)
            else:
                yield new_prefix, value

    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.data = {}

    def write_data(self, dataset_name: str, data: np.ndarray, overwrite=False):
        # convert dataset_name path to a list
        dataset_name_list = dataset_name.split("/")
        # check if dataset already exists in nested dictionary
        self._create_nested_dict(
            self.data, dataset_name_list, data, overwrite=overwrite
        )

    def get_data(self, dataset_name: str):
        # convert dataset_name path to a list
        dataset_name_list = dataset_name.split("/")
        # check if dataset exists in nested dictionary
        d = self.data
        for key in dataset_name_list:
            if key not in d:
                raise ValueError(
                    f"Dataset {dataset_name} does not exist in dictionary {self.data}"
                )
            d = d[key]
        return d

    def dataset_exists(self, dataset_name: str):
        # convert dataset_name path to a list
        dataset_name_list = dataset_name.split("/")
        # check if dataset exists in nested dictionary
        d = self.data
        for key in dataset_name_list:
            if key not in d:
                return False
            d = d[key]
        return True

    def get_storage_element(self):
        return self.data

    def convert_to_h5pywriter(self, filename=None, compression=None):
        """Converts the LocalWriter object to a H5pyWriter object."""
        if filename is None:
            filename = self.filename

        # create a new H5pyWriter object
        h5pywriter = H5pyWriter(filename, compression=compression)

        for path, data in self._explore_nested_dict(self.data):
            h5pywriter.write_data(path, data)

        return h5pywriter