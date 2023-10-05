import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import xtrack.twiss as xtw
from tqdm.autonotebook import tqdm

from .generic_writer import GenericWriter
from .normed_particles import NormedParticles


def birkhoff_weights(n):
    """Get the Birkhoff weights for a given number of samples.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Array of Birkhoff weights.
    """
    weights = np.arange(n, dtype=np.float64)
    weights /= n
    weights = np.exp(-1 / (weights * (1 - weights)))
    return weights / np.sum(weights)


def get_part_displacement_and_direction(
    ref_part: xp.Particles, part: xp.Particles, _context=xo.ContextCpu()
):
    """Get the displacement and direction of a particle with respect to a reference
    particle.

    Parameters
    ----------
    ref_part : xp.Particles
        Reference particle.
    part : xp.Particles
        Particle to be compared with the reference particle.

    Returns
    -------
    tuple
        displacement, direction
    """
    argsort_ref = np.argsort(ref_part.particle_id)
    argsort = np.argsort(part.particle_id)

    direction = _context.nplike_array_type([6, len(part.particle_id)])
    direction[0, :] = part.x[argsort] - ref_part.x[argsort_ref]
    direction[1, :] = part.px[argsort] - ref_part.px[argsort_ref]
    direction[2, :] = part.y[argsort] - ref_part.y[argsort_ref]
    direction[3, :] = part.py[argsort] - ref_part.py[argsort_ref]
    direction[4, :] = part.zeta[argsort] - ref_part.zeta[argsort_ref]
    direction[5, :] = part.ptau[argsort] - ref_part.ptau[argsort_ref]

    displacement = np.sum((direction) ** 2, axis=0) ** 0.5

    direction /= displacement
    return displacement, direction


def get_normed_part_displacement_and_direction(
    ref_norm_part: NormedParticles,
    norm_part: NormedParticles,
    argsort_ref,
    argsort,
    _context=xo.ContextCpu(),
):
    """Get the displacement and direction of a particle with respect to a reference
    particle. Normalized coordinates are used.

    Parameters
    ----------
    ref_norm_part : NormedParticles
        Reference particles.
    norm_part : NormedParticles
        Particles to be compared with the reference particles.
    argsort_ref : np.ndarray
        Array of indices that sorts the reference particles.
    argsort : np.ndarray
        Array of indices that sorts the particles.
    _context : xo.Context, optional
        Context to be used, by default xo.ContextCpu()

    Returns
    -------
    tuple
        displacement, direction
    """
    direction = _context.nplike_array_type([6, len(norm_part.x_norm)])
    direction[0, :] = norm_part.x_norm[argsort] - ref_norm_part.x_norm[argsort_ref]
    direction[1, :] = norm_part.px_norm[argsort] - ref_norm_part.px_norm[argsort_ref]
    direction[2, :] = norm_part.y_norm[argsort] - ref_norm_part.y_norm[argsort_ref]
    direction[3, :] = norm_part.py_norm[argsort] - ref_norm_part.py_norm[argsort_ref]
    direction[4, :] = (
        norm_part.zeta_norm[argsort] - ref_norm_part.zeta_norm[argsort_ref]
    )
    direction[5, :] = (
        norm_part.pzeta_norm[argsort] - ref_norm_part.pzeta_norm[argsort_ref]
    )

    displacement = np.sum((direction) ** 2, axis=0) ** 0.5

    direction /= displacement
    return displacement, direction


class GhostParticleManager:
    """Class to manage ghost particles and track them."""

    def __init__(
        self,
        part: xp.Particles,
        _context=xo.ContextCpu(),
        use_norm_coord=True,
        twiss=None,
        nemitt_x=None,
        nemitt_y=None,
        idx_pos=0,
    ):
        """Initialize the GhostParticleManager.

        Parameters
        ----------
        part : xp.Particles
            Particles to be tracked.
        _context : xo.Context, optional
            Context to be used, by default xo.ContextCpu()
        use_norm_coord : bool, optional
            If True, normalized coordinates are used, by default True
        twiss : xtw.Twiss, optional
            Twiss object, by default None, required if use_norm_coord is True
        nemitt_x : float, optional
            Normalized emittance in x, by default None, required if use_norm_coord
            is True
        nemitt_y : float, optional
            Normalized emittance in y, by default None, required if use_norm_coord
            is True
        idx_pos : int, optional
            Index of the position in the particle array, by default 0, required if
            use_norm_coord is True
        """
        self._part = part
        self._context = _context
        self._use_norm_coord = use_norm_coord

        if self._use_norm_coord:
            if twiss is None:
                raise ValueError("If norm_coord is True, twiss must be given")
            if nemitt_x is None:
                raise ValueError("If norm_coord is True, nemitt_x must be given")
            if nemitt_y is None:
                raise ValueError("If norm_coord is True, nemitt_y must be given")

            self._twiss = twiss
            self._nemitt_x = nemitt_x
            self._nemitt_y = nemitt_y
            self._idx_pos = idx_pos

            self._normed_part = NormedParticles(
                self._twiss,
                self._nemitt_x,
                self._nemitt_y,
                self._context,
                self._idx_pos,
                part=self._part,
            )
            self._ghost_normed_part = []
        else:
            self._normed_part = None
            self._ghost_normed_part = None

        self._ghost_part = []
        self._ghost_name = []
        self._original_displacement = []
        self._original_direction = []

    def _save_metadata(self, out: GenericWriter):
        """Save the metadata of the ghost particles.

        Parameters
        ----------
        out : GenericWriter
            Writer to save the metadata.
        """
        out.write_data("ghost_name", self._ghost_name)
        out.write_data("original_displacement", self._original_displacement)
        out.write_data("original_direction", self._original_direction)
        out.write_data("use_norm_coord", self._use_norm_coord)

    def add_displacement(
        self, module=1e-6, direction="x", custom_displacement=None, ghost_name=None
    ):
        """Add a ghost particle with a displacement in the given direction.

        Parameters
        ----------
        module : float, optional
            Module of the displacement, by default 1e-8
        direction : str, optional
            Direction of the displacement, by default 'x', must be one of: x, px, y,
            py, zeta, pzeta if use_norm_coord is False, or x_norm, px_norm, y_norm,
            py_norm, zeta_norm, pzeta_norm if use_norm_coord is True. Must be custom
            if custom_displacement is given.
        custom_displacement : np.ndarray, optional
            Custom displacement, by default None, must be given if direction is
            custom
        ghost_name : str, optional
            Name of the ghost particle, by default None, if None, the name is
            automatically generated
        """
        if custom_displacement is None:
            custom_displacement = np.zeros(6)
            if self._use_norm_coord:
                if direction == "x_norm":
                    custom_displacement[0] = module
                elif direction == "px_norm":
                    custom_displacement[1] = module
                elif direction == "y_norm":
                    custom_displacement[2] = module
                elif direction == "py_norm":
                    custom_displacement[3] = module
                elif direction == "zeta_norm":
                    custom_displacement[4] = module
                elif direction == "pzeta_norm":
                    custom_displacement[5] = module
                else:
                    raise ValueError(
                        "Invalid direction, with use_norm_coord=True, direction must be one of: x_norm, px_norm, y_norm, py_norm, zeta_norm, pzeta_norm"
                    )
            else:
                if direction == "x":
                    custom_displacement[0] = module
                elif direction == "px":
                    custom_displacement[1] = module
                elif direction == "y":
                    custom_displacement[2] = module
                elif direction == "py":
                    custom_displacement[3] = module
                elif direction == "zeta":
                    custom_displacement[4] = module
                elif direction == "ptau":
                    custom_displacement[5] = module
                else:
                    raise ValueError(
                        "Invalid direction, with use_norm_coord=False, direction must be one of: x, px, y, py, zeta, ptau"
                    )

            self._original_displacement.append(module)
            self._original_direction.append(
                custom_displacement / np.sum(custom_displacement**2) ** 0.5
            )
        else:
            if direction != "custom":
                raise ValueError(
                    "If custom_displacement is given, direction must be custom"
                )
            if np.asarray(custom_displacement).shape != (6,):
                raise ValueError(
                    "If custom_displacement is given, it must be a 6 element array"
                )
            self._original_displacement.append(np.sum(custom_displacement**2) ** 0.5)
            self._original_direction.append(
                custom_displacement / np.sum(custom_displacement**2) ** 0.5
            )

        # check if ghost_name is already used
        if ghost_name is None:
            ghost_name = f"ghost_{direction}"
        if ghost_name in self._ghost_name:
            raise ValueError("ghost_name is already used")

        self._ghost_name.append(ghost_name)

        # make copy of part
        disp_part = self._part.copy()

        if self._use_norm_coord == False:
            disp_part.x += custom_displacement[0]
            disp_part.px += custom_displacement[1]
            disp_part.y += custom_displacement[2]
            disp_part.py += custom_displacement[3]
            disp_part.zeta += custom_displacement[4]
            disp_part.ptau += custom_displacement[5]

            self._ghost_part.append(disp_part)
        else:
            # make copy of normed_part
            disp_normed_part = NormedParticles(
                self._twiss,
                self._nemitt_x,
                self._nemitt_y,
                self._context,
                self._idx_pos,
                part=self._part,
            )
            disp_normed_part.x_norm += custom_displacement[0]
            disp_normed_part.px_norm += custom_displacement[1]
            disp_normed_part.y_norm += custom_displacement[2]
            disp_normed_part.py_norm += custom_displacement[3]
            disp_normed_part.zeta_norm += custom_displacement[4]
            disp_normed_part.pzeta_norm += custom_displacement[5]

            # convert back to disp_part
            disp_part = disp_normed_part.norm_to_phys(disp_part)

            self._ghost_normed_part.append(disp_normed_part)
            self._ghost_part.append(disp_part)

    def get_displacement_data(self, get_context_arrays=False):
        """Get the displacement data for the ghost particles.

        Parameters
        ----------
        get_context_arrays : bool, optional
            If True, the displacement and direction are returned as context arrays, by default False

        Returns
        -------
        list
            List of the displacement for each ghost particle
        list
            List of the direction of the displacement for each ghost particle
        """
        direction_list = []
        displacement_list = []

        for i, ghost_part in enumerate(self._ghost_part):
            if self._use_norm_coord:
                argsort_ref = np.argsort(self._part.particle_id)
                argsort = np.argsort(ghost_part.particle_id)
                displacement, direction = get_normed_part_displacement_and_direction(
                    self._normed_part,
                    self._ghost_normed_part[i],
                    argsort_ref=argsort_ref,
                    argsort=argsort,
                    _context=self._context,
                )
            else:
                displacement, direction = get_part_displacement_and_direction(
                    self._part, ghost_part, self._context
                )

            direction_list.append(
                direction
                if get_context_arrays
                else self._context.nparray_from_context_array(direction)
            )
            displacement_list.append(
                displacement
                if get_context_arrays
                else self._context.nparray_from_context_array(displacement)
            )

        return displacement_list, direction_list

    def realign_particles(self, module=None, get_context_arrays=False):
        """Realign the ghost particles to the original particles.

        Parameters
        ----------
        module : float, optional
            The module of the displacement, by default None, if None, the original
            module is used
        get_context_arrays : bool, optional
            If True, the displacement and direction are returned as context arrays,
            by default False

        Returns
        -------
        list
            List of the displacement for each ghost particle before realignment
        list
            List of the direction of the displacement for each ghost particle
        """
        displacement_list, direction_list = self.get_displacement_data(
            get_context_arrays=True
        )
        ref_argsort = np.argsort(self._part.particle_id)

        if self._use_norm_coord is False:
            for i, (ghost_part, direction, displacement) in enumerate(
                zip(self._ghost_part, direction_list, displacement_list)
            ):
                if module is None:
                    module = self._original_displacement[i]
                tmp_argsort = np.argsort(ghost_part.particle_id)
                inv_argsort = np.argsort(tmp_argsort)

                ghost_part.x = (self._part.x[ref_argsort] + module * direction[0])[
                    inv_argsort
                ]
                ghost_part.px = (self._part.px[ref_argsort] + module * direction[1])[
                    inv_argsort
                ]
                ghost_part.y = (self._part.y[ref_argsort] + module * direction[2])[
                    inv_argsort
                ]
                ghost_part.py = (self._part.py[ref_argsort] + module * direction[3])[
                    inv_argsort
                ]
                ghost_part.zeta = (
                    self._part.zeta[ref_argsort] + module * direction[4]
                )[inv_argsort]
                ghost_part.ptau = (
                    self._part.ptau[ref_argsort] + module * direction[5]
                )[inv_argsort]

        else:
            for i, (
                ghost_part,
                ghost_normed_part,
                direction,
                displacement,
            ) in enumerate(
                zip(
                    self._ghost_part,
                    self._ghost_normed_part,
                    direction_list,
                    displacement_list,
                )
            ):
                if module is None:
                    module = self._original_displacement[i]
                argsort = np.argsort(ghost_part.particle_id)
                inv_argsort = np.argsort(argsort)

                ghost_normed_part.x_norm = (
                    self._normed_part.x_norm[ref_argsort] + module * direction[0]
                )[inv_argsort]
                ghost_normed_part.px_norm = (
                    self._normed_part.px_norm[ref_argsort] + module * direction[1]
                )[inv_argsort]
                ghost_normed_part.y_norm = (
                    self._normed_part.y_norm[ref_argsort] + module * direction[2]
                )[inv_argsort]
                ghost_normed_part.py_norm = (
                    self._normed_part.py_norm[ref_argsort] + module * direction[3]
                )[inv_argsort]
                ghost_normed_part.zeta_norm = (
                    self._normed_part.zeta_norm[ref_argsort] + module * direction[4]
                )[inv_argsort]
                ghost_normed_part.pzeta_norm = (
                    self._normed_part.pzeta_norm[ref_argsort] + module * direction[5]
                )[inv_argsort]

                ghost_part = ghost_normed_part.norm_to_phys(ghost_part)

        if get_context_arrays:
            return displacement_list, direction_list

        return [
            self._context.nparray_from_context_array(displacement)
            for displacement in displacement_list
        ], [
            self._context.nparray_from_context_array(direction)
            for direction in direction_list
        ]

    def track_displacement(
        self,
        line: xt.Line,
        sampling_turns,
        out: GenericWriter,
        realign_frequency=10,
        custom_realign=False,
        realing_module=None,
        tqdm_flag=True,
    ):
        """Track the displacement and direction of the ghost particles.

        Parameters
        ----------
        line : xt.Line
            The line to track
        sampling_turns : list
            List of the turns to sample the displacement
        out : GenericWriter
            The writer to write the data
        realign_frequency : int, optional
            The frequency of realignment, by default 10
        custom_realign : bool, optional
            If True, the realignment is done by the realing_module, by default
            False
        realing_module : float, optional
            The module to use for realignment, by default None
        """
        if custom_realign:
            if realing_module is None:
                raise ValueError("custom_realign is True but realing_module is None")
        else:
            realing_module = None

        self._save_metadata(out)

        sampling_turns = np.sort(np.unique(np.asarray(sampling_turns, dtype=int)))
        max_turn = np.max(sampling_turns)

        realigning_turns = np.arange(0, max_turn + 1, realign_frequency)[1:]

        s_events = [("sample", t, i) for i, t in enumerate(sampling_turns)]
        r_events = [("realign", t, i) for i, t in enumerate(realigning_turns)]
        events = sorted(
            s_events + r_events, key=lambda x: x[1] + 0.5 if x[0] == "realign" else x[1]
        )

        log_displacement_storage = self._context.nplike_array_type(
            (len(self._ghost_name), len(self._part.particle_id))
        )

        current_turn = 0
        pbar = tqdm(total=max_turn, disable=not tqdm_flag)
        for event, turn, event_idx in events:
            delta_turn = turn - current_turn
            if delta_turn > 0:
                line.track(self._part, num_turns=delta_turn)
                for part in self._ghost_part:
                    line.track(part, num_turns=delta_turn)
                current_turn = turn
                pbar.update(delta_turn)

            if event == "sample":
                displacement_list, direction_list = self.get_displacement_data(
                    get_context_arrays=True
                )
                for i, (
                    stored_log_displacement,
                    displacement,
                    direction,
                    name,
                ) in enumerate(
                    zip(
                        log_displacement_storage,
                        displacement_list,
                        direction_list,
                        self._ghost_name,
                    )
                ):
                    log_displacement_to_save = (
                        np.log10(displacement) + stored_log_displacement
                    )

                    out.write_data(
                        f"displacement/{name}/{current_turn}",
                        self._context.nparray_from_context_array(
                            log_displacement_to_save / current_turn
                        ),
                    )
                    local_direction = self._context.nparray_from_context_array(
                        direction
                    )
                    out.write_data(
                        f'direction/{name}/{"x_norm" if self._normed_part else "x"}/{current_turn}',
                        local_direction[0],
                    )
                    out.write_data(
                        f'direction/{name}/{"px_norm" if self._normed_part else "px"}/{current_turn}',
                        local_direction[1],
                    )
                    out.write_data(
                        f'direction/{name}/{"y_norm" if self._normed_part else "y"}/{current_turn}',
                        local_direction[2],
                    )
                    out.write_data(
                        f'direction/{name}/{"py_norm" if self._normed_part else "py"}/{current_turn}',
                        local_direction[3],
                    )
                    out.write_data(
                        f'direction/{name}/{"zeta_norm" if self._normed_part else "zeta"}/{current_turn}',
                        local_direction[4],
                    )
                    out.write_data(
                        f'direction/{name}/{"pzeta_norm" if self._normed_part else "ptau"}/{current_turn}',
                        local_direction[5],
                    )

            elif event == "realign":
                displacement_list, direction_list = self.realign_particles(
                    module=realing_module, get_context_arrays=True
                )

                for i, (stored_log_displacement, displacement, name) in enumerate(
                    zip(log_displacement_storage, displacement_list, self._ghost_name)
                ):
                    stored_log_displacement += np.log10(displacement)

        # save nturns of main particles
        out.write_data(
            "at_turn", self._context.nparray_from_context_array(self._part.at_turn)
        )

    def track_displacement_birkhoff(
        self,
        line: xt.Line,
        sampling_turns,
        out: GenericWriter,
        custom_realign=False,
        realing_module=None,
        tqdm_flag=True,
    ):
        """Track the displacement and direction of the ghost particles while using
        the birkhoff weights for the displacement.

        Parameters
        ----------
        line : xt.Line
            The line to track
        sampling_turns : list
            List of the turns to sample the displacement
        out : GenericWriter
            The writer to write the data
        custom_realign : bool, optional
            If True, the realignment is done by the realing_module, by default False
        realing_module : float, optional
            The module to use for realignment, by default None
        """
        if custom_realign:
            if realing_module is None:
                raise ValueError("custom_realign is True but realing_module is None")
        else:
            realing_module = None

        self._save_metadata(out)

        sampling_turns = np.sort(np.unique(np.asarray(sampling_turns, dtype=int)))
        max_turn = np.max(sampling_turns)

        birk_weights_list = [
            self._context.nparray_to_context_array(birkhoff_weights(t))
            for t in sampling_turns
        ]
        birk_log_displacement_storage = self._context.nplike_array_type(
            (len(sampling_turns), len(self._ghost_name), len(self._part.particle_id))
        )

        current_turn = 0
        pbar = tqdm(total=max_turn, disable=not tqdm_flag)
        for t in range(max_turn + 1):
            line.track(self._part, num_turns=1)
            for part in self._ghost_part:
                line.track(part, num_turns=1)
            current_turn += 1
            pbar.update(1)

            displacement_list, direction_list = self.realign_particles(
                module=realing_module, get_context_arrays=True
            )

            for s_idx, sample in enumerate(sampling_turns):
                if current_turn <= sample:
                    for i, (birk_stored_log_displacement, displacement) in enumerate(
                        zip(birk_log_displacement_storage[s_idx], displacement_list)
                    ):
                        birk_log_displacement_storage[s_idx][i] = (
                            np.log10(displacement) * birk_weights_list[s_idx][t]
                            + birk_stored_log_displacement
                        )

            if current_turn in sampling_turns:
                s_idx = np.where(sampling_turns == current_turn)[0][0]

                for i, (stored_log_displacement, direction, name) in enumerate(
                    zip(
                        birk_log_displacement_storage[s_idx],
                        direction_list,
                        self._ghost_name,
                    )
                ):
                    out.write_data(
                        f"displacement/{name}/{current_turn}",
                        self._context.nparray_from_context_array(
                            stored_log_displacement
                        ),
                    )
                    local_direction = self._context.nparray_from_context_array(
                        direction
                    )
                    out.write_data(
                        f'direction/{name}/{"x_norm" if self._normed_part else "x"}/{current_turn}',
                        local_direction[0],
                    )
                    out.write_data(
                        f'direction/{name}/{"px_norm" if self._normed_part else "px"}/{current_turn}',
                        local_direction[1],
                    )
                    out.write_data(
                        f'direction/{name}/{"y_norm" if self._normed_part else "y"}/{current_turn}',
                        local_direction[2],
                    )
                    out.write_data(
                        f'direction/{name}/{"py_norm" if self._normed_part else "py"}/{current_turn}',
                        local_direction[3],
                    )
                    out.write_data(
                        f'direction/{name}/{"zeta_norm" if self._normed_part else "zeta"}/{current_turn}',
                        local_direction[4],
                    )
                    out.write_data(
                        f'direction/{name}/{"pzeta_norm" if self._normed_part else "ptau"}/{current_turn}',
                        local_direction[5],
                    )

        # save nturns of main particles
        out.write_data(
            "at_turn", self._context.nparray_from_context_array(self._part.at_turn)
        )
