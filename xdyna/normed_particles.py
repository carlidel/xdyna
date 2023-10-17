import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import xtrack.twiss as xtw
from scipy.constants import c as c_light

EW_GIVEN = 2.185575985356659 * (c_light / (4 * np.pi)) * 1e3 / 1e12


class NormedParticles:
    """Class to store particles in normalized coordinates."""

    def __len__(self):
        return self._normed_part.shape[1]

    def __init__(
        self,
        twiss: xtw.TwissTable,
        nemitt_x: float,
        nemitt_y: float,
        nemitt_z=None,
        _context=xo.ContextCpu(),
        idx_pos=0,
        part=None,
        n_part=1000,
    ):
        """Initialize the NormedParticles object.

        Parameters
        ----------
        twiss : xtw.TwissTable
            Twiss table of the line to be used for the normalization.
        nemitt_x : float
            Normalized emittance in the horizontal plane.
        nemitt_y : float
            Normalized emittance in the vertical plane.
        nemitt_z : float, optional
            Normalized emittance in the longitudinal plane, by default None
            If None, a unitary emittance is assumed.
        _context : _type_
            xobjects context to be used.
        idx_pos : int, optional
            Index to the element wanted for the normalization, by default 0
        part : _type_, optional
            Particle object to be used as base, by default None
        n_part : int, optional
            If no part is given, a storage of size n_part is allocated, by
            default 1000

        Raises
        ------
        ValueError
            If neither part nor n_part is given.
        """
        self._context = _context
        self._twiss_data, self._w, self._w_inv = NormedParticles.get_twiss_data(
            twiss, nemitt_x, nemitt_y, _context, nemitt_z, idx_pos
        )

        if part is None:
            if n_part is None:
                raise ValueError("Either part or n_part must be given")
            self._normed_part = _context.nplike_array_type([6, n_part])

        else:
            self._create_normed_placeholder(part)

    @classmethod
    def get_twiss_data(
        cls,
        twiss: xtw.TwissTable,
        nemitt_x: float,
        nemitt_y: float,
        _context,
        nemitt_z=None,
        idx_pos=0,
    ):
        """Get the twiss data for the given twiss object and the given
        normalized emittance values.

        Parameters
        ----------
        twiss : xtrack.Twiss
            Twiss object
        nemitt_x : float
            Normalized emittance in x
        nemitt_y : float
            Normalized emittance in y
        _context : xobjects.Context
            Context to use
        nemitt_z : float, optional
            Normalized emittance in z, by default None
            If None, a unitary emittance is assumed.
        idx_pos : int, optional
            Index of the position to use, by default 0

        Returns
        -------
        twiss_data : xobjects.Array
            Twiss data with the following structure:
            [nemitt_x, nemitt_y, twiss.x[idx_pos], twiss.px[idx_pos], twiss.y
            [idx_pos], twiss.py[idx_pos], twiss.zeta[idx_pos], twiss.ptau
            [idx_pos]]
        w : xobjects.Array
            Twiss W matrix
        w_inv : xobjects.Array
            Twiss W inverse matrix
        """
        twiss_data = _context.nplike_array_type(9)

        twiss_data[0] = nemitt_x
        twiss_data[1] = nemitt_y

        twiss_data[2] = twiss.x[idx_pos]
        twiss_data[3] = twiss.px[idx_pos]
        twiss_data[4] = twiss.y[idx_pos]
        twiss_data[5] = twiss.py[idx_pos]
        twiss_data[6] = twiss.zeta[idx_pos]
        twiss_data[7] = twiss.ptau[idx_pos]

        twiss_data[8] = nemitt_z if nemitt_z is not None else np.nan

        w = _context.nparray_to_context_array(twiss.W_matrix[idx_pos])
        w_inv = _context.nparray_to_context_array(
            np.linalg.inv(twiss.W_matrix[idx_pos])
        )

        return twiss_data, w, w_inv

    def _create_normed_placeholder(self, part: xp.Particles):
        """Create a placeholder for the normalized part. Updates the
        normed_part attribute.

        Parameters
        ----------
        part : xp.Particles
            Particles object
        """

        self._normed_part = self._context.nplike_array_type([6, len(part.x)])
        self.phys_to_norm(part)

    def phys_to_norm(self, part: xp.Particles):
        """Transform the physical coordinates to normalized coordinates.
        Updates the normed_part attribute.

        Parameters
        ----------
        part : xp.Particles
            Particles object
        """
        mask = part.state <= 0
        gemitt_x = (
            self._twiss_data[0] / part._xobject.beta0[0] / part._xobject.gamma0[0]
        )
        gemitt_y = (
            self._twiss_data[1] / part._xobject.beta0[0] / part._xobject.gamma0[0]
        )
        gemitt_z = (
            self._twiss_data[8] / (part._xobject.beta0[0] / part._xobject.gamma0[0])
            if not np.isnan(self._twiss_data[8])
            else 1.0
        )

        self._normed_part[0] = part.x - self._twiss_data[2]
        self._normed_part[1] = part.px - self._twiss_data[3]
        self._normed_part[2] = part.y - self._twiss_data[4]
        self._normed_part[3] = part.py - self._twiss_data[5]
        self._normed_part[4] = part.zeta - self._twiss_data[6]
        self._normed_part[5] = (part.ptau - self._twiss_data[7]) / part._xobject.beta0[
            0
        ]

        self._normed_part = np.dot(self._w_inv, self._normed_part)

        self._normed_part[0] /= np.sqrt(gemitt_x)
        self._normed_part[1] /= np.sqrt(gemitt_x)
        self._normed_part[2] /= np.sqrt(gemitt_y)
        self._normed_part[3] /= np.sqrt(gemitt_y)
        self._normed_part[4] /= np.sqrt(gemitt_z)
        self._normed_part[5] /= np.sqrt(gemitt_z)

        self._normed_part[:, mask] = np.nan

    def norm_to_phys(self, part: xp.Particles):
        """Transform the normalized coordinates to physical coordinates.
        Updates the given Particles object.

        Parameters
        ----------
        part : xp.Particles
            Target Particles object

        Returns
        -------
        part : xp.Particles
            Particles object
        """
        # mask = part.state <= 0
        gemitt_x = (
            self._twiss_data[0] / part._xobject.beta0[0] / part._xobject.gamma0[0]
        )
        gemitt_y = (
            self._twiss_data[1] / part._xobject.beta0[0] / part._xobject.gamma0[0]
        )
        gemitt_z = (
            self._twiss_data[8] / (part._xobject.beta0[0] / part._xobject.gamma0[0])
            if not np.isnan(self._twiss_data[8])
            else 1.0
        )

        normed = self._normed_part.copy()
        normed[0] *= np.sqrt(gemitt_x)
        normed[1] *= np.sqrt(gemitt_x)
        normed[2] *= np.sqrt(gemitt_y)
        normed[3] *= np.sqrt(gemitt_y)
        normed[4] *= np.sqrt(gemitt_z)
        normed[5] *= np.sqrt(gemitt_z)

        normed = np.dot(self._w, normed)

        part.zeta = normed[4] + self._twiss_data[6]
        part.ptau = normed[5] * part._xobject.beta0[0] + self._twiss_data[7]

        part.x = normed[0] + self._twiss_data[2]
        part.px = normed[1] + self._twiss_data[3]
        part.y = normed[2] + self._twiss_data[4]
        part.py = normed[3] + self._twiss_data[5]

        return part

    @property
    def x_norm(self):
        """Normalized x coordinates (returns a context array)"""
        return self._normed_part[0]

    @x_norm.setter
    def x_norm(self, value):
        self._normed_part[0] = value

    @property
    def px_norm(self):
        """Normalized px coordinates (returns a context array)"""
        return self._normed_part[1]

    @px_norm.setter
    def px_norm(self, value):
        self._normed_part[1] = value

    @property
    def y_norm(self):
        """Normalized y coordinates (returns a context array)"""
        return self._normed_part[2]

    @y_norm.setter
    def y_norm(self, value):
        self._normed_part[2] = value

    @property
    def py_norm(self):
        """Normalized py coordinates (returns a context array)"""
        return self._normed_part[3]

    @py_norm.setter
    def py_norm(self, value):
        self._normed_part[3] = value

    @property
    def zeta_norm(self):
        """Normalized zeta coordinates (returns a context array)"""
        return self._normed_part[4]

    @zeta_norm.setter
    def zeta_norm(self, value):
        self._normed_part[4] = value

    @property
    def pzeta_norm(self):
        """Normalized pzeta coordinates (returns a context array)"""
        return self._normed_part[5]

    @pzeta_norm.setter
    def pzeta_norm(self, value):
        self._normed_part[5] = value
