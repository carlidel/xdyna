import warnings

import numpy as np
import xtrack as xt
from tqdm.auto import tqdm

from .generic_writer import GenericWriter
from .ghost_particle_manager import GhostParticleManager


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


def track_displacement(
    gpm: GhostParticleManager,
    line: xt.Line,
    sampling_turns,
    out: GenericWriter,
    realign_frequency=10,
    realign_module=None,
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
    realign_module : float, optional
        The custom module to use for realignment, if None, the default module
        set in the GhostParticleManager will be used, by default None
    tqdm_flag : bool, optional
        If True, show the progress bar, by default True
    """
    gpm.save_metadata(out)

    sampling_turns = np.sort(np.unique(np.asarray(sampling_turns, dtype=int)))
    max_turn = np.max(sampling_turns)

    realigning_turns = np.arange(0, max_turn + 1, realign_frequency)[1:]

    s_events = [("sample", t, i) for i, t in enumerate(sampling_turns)]
    r_events = [("realign", t, i) for i, t in enumerate(realigning_turns)]
    events = sorted(
        s_events + r_events, key=lambda x: x[1] + 0.5 if x[0] == "realign" else x[1]
    )

    log_displacement_storage = gpm.context.nplike_array_type(
        (len(gpm.ghost_name), len(gpm.part.particle_id))
    )

    current_turn = 0
    pbar = tqdm(total=max_turn, disable=not tqdm_flag)
    for event, turn, event_idx in events:
        delta_turn = turn - current_turn
        if delta_turn > 0:
            line.track(gpm.part, num_turns=delta_turn)
            for part in gpm.ghost_part:
                line.track(part, num_turns=delta_turn)
            current_turn = turn
            pbar.update(delta_turn)

        if event == "sample":
            displacement_list, direction_list = gpm.get_displacement_data(
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
                    gpm.ghost_name,
                )
            ):
                log_displacement_to_save = (
                    np.log10(displacement) + stored_log_displacement
                )

                out.write_data(
                    f"displacement/{name}/{current_turn}",
                    gpm.context.nparray_from_context_array(
                        log_displacement_to_save / current_turn
                    ),
                )
                local_direction = gpm.context.nparray_from_context_array(direction)
                out.write_data(
                    f'direction/{name}/{"x_norm" if gpm.normed_part else "x"}/{current_turn}',
                    local_direction[0],
                )
                out.write_data(
                    f'direction/{name}/{"px_norm" if gpm.normed_part else "px"}/{current_turn}',
                    local_direction[1],
                )
                out.write_data(
                    f'direction/{name}/{"y_norm" if gpm.normed_part else "y"}/{current_turn}',
                    local_direction[2],
                )
                out.write_data(
                    f'direction/{name}/{"py_norm" if gpm.normed_part else "py"}/{current_turn}',
                    local_direction[3],
                )
                out.write_data(
                    f'direction/{name}/{"zeta_norm" if gpm.normed_part else "zeta"}/{current_turn}',
                    local_direction[4],
                )
                out.write_data(
                    f'direction/{name}/{"pzeta_norm" if gpm.normed_part else "ptau"}/{current_turn}',
                    local_direction[5],
                )

        elif event == "realign":
            displacement_list, direction_list = gpm.realign_particles(
                module=realign_module, get_context_arrays=True
            )

            for i, (stored_log_displacement, displacement, name) in enumerate(
                zip(log_displacement_storage, displacement_list, gpm.ghost_name)
            ):
                stored_log_displacement += np.log10(displacement)

    # save nturns of main particles
    out.write_data("at_turn", gpm.context.nparray_from_context_array(gpm.part.at_turn))


def track_displacement_birkhoff(
    gpm: GhostParticleManager,
    line: xt.Line,
    sampling_turns,
    out: GenericWriter,
    realign_frequency=1,
    realign_module=None,
    include_no_birkhoff=False,
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
    realign_frequency : int, optional
        The frequency of realignment, by default 1
    realign_module : float, optional
        The module to use for realignment, if None, the default module
        set in the GhostParticleManager will be used, by default None
    include_no_birkhoff : bool, optional
        If True, the displacement without birkhoff weights is also saved, by
        default False
    tqdm_flag : bool, optional
        If True, show the progress bar, by default True
    """
    gpm.save_metadata(out)

    if np.any(np.asarray(sampling_turns, dtype=int) % realign_frequency != 0):
        warnings.warn(
            "Some of the sampling turns are not multiple of the realign frequency.\n"
            + "The values will be rounded down to the closest multiple.",
            category=UserWarning,
        )

    sampling_turns = np.unique(
        (((np.asarray(sampling_turns, dtype=int))) // realign_frequency)
        * realign_frequency
    )
    max_turn = np.max(sampling_turns)

    n_realignments = sampling_turns // realign_frequency
    realigning_turns = np.arange(0, max_turn + 1, realign_frequency)[1:]

    s_events = [("sample", t, i) for i, t in enumerate(sampling_turns)]
    r_events = [("realign", t, i) for i, t in enumerate(realigning_turns)]
    events = sorted(
        s_events + r_events, key=lambda x: x[1] + 0.5 if x[0] == "realign" else x[1]
    )

    birk_weights_list = [
        gpm.context.nparray_to_context_array(birkhoff_weights(t))
        for t in n_realignments
    ]
    birk_log_displacement_storage = gpm.context.nplike_array_type(
        (len(sampling_turns), len(gpm.ghost_name), len(gpm.part.particle_id))
    )

    if include_no_birkhoff:
        log_displacement_storage = gpm.context.nplike_array_type(
            (len(gpm.ghost_name), len(gpm.part.particle_id))
        )

    current_turn = 0
    pbar = tqdm(total=max_turn, disable=not tqdm_flag)
    for event, turn, event_idx in events:
        delta_turn = turn - current_turn
        if delta_turn > 0:
            line.track(gpm.part, num_turns=delta_turn)
            for part in gpm.ghost_part:
                line.track(part, num_turns=delta_turn)
            current_turn = turn
            pbar.update(delta_turn)

        if event == "realign":
            displacement_list, direction_list = gpm.realign_particles(
                module=realign_module, get_context_arrays=True
            )

            for s_idx, sample in enumerate(sampling_turns):
                if current_turn <= sample:
                    for i, (birk_stored_log_displacement, displacement) in enumerate(
                        zip(birk_log_displacement_storage[s_idx], displacement_list)
                    ):
                        birk_log_displacement_storage[s_idx][i] = (
                            np.log10(displacement) * birk_weights_list[s_idx][event_idx]
                            + birk_stored_log_displacement
                        )
            if include_no_birkhoff:
                for i, (stored_log_displacement, displacement) in enumerate(
                    zip(log_displacement_storage, displacement_list)
                ):
                    log_displacement_storage[i] += np.log10(displacement)

        elif event == "sample":
            _, direction_list = gpm.get_displacement_data(get_context_arrays=True)

            s_idx = np.where(sampling_turns == current_turn)[0][0]

            for i, (stored_log_displacement, direction, name) in enumerate(
                zip(
                    birk_log_displacement_storage[s_idx],
                    direction_list,
                    gpm.ghost_name,
                )
            ):
                if include_no_birkhoff:
                    out.write_data(
                        f"displacement_nobirk/{name}/{current_turn}",
                        gpm.context.nparray_from_context_array(
                            log_displacement_storage[i] / current_turn
                        ),
                    )
                out.write_data(
                    f"displacement/{name}/{current_turn}",
                    gpm.context.nparray_from_context_array(stored_log_displacement),
                )
                local_direction = gpm.context.nparray_from_context_array(direction)
                out.write_data(
                    f'direction/{name}/{"x_norm" if gpm.normed_part else "x"}/{current_turn}',
                    local_direction[0],
                )
                out.write_data(
                    f'direction/{name}/{"px_norm" if gpm.normed_part else "px"}/{current_turn}',
                    local_direction[1],
                )
                out.write_data(
                    f'direction/{name}/{"y_norm" if gpm.normed_part else "y"}/{current_turn}',
                    local_direction[2],
                )
                out.write_data(
                    f'direction/{name}/{"py_norm" if gpm.normed_part else "py"}/{current_turn}',
                    local_direction[3],
                )
                out.write_data(
                    f'direction/{name}/{"zeta_norm" if gpm.normed_part else "zeta"}/{current_turn}',
                    local_direction[4],
                )
                out.write_data(
                    f'direction/{name}/{"pzeta_norm" if gpm.normed_part else "ptau"}/{current_turn}',
                    local_direction[5],
                )

    # save nturns of main particles
    out.write_data("at_turn", gpm.context.nparray_from_context_array(gpm.part.at_turn))
