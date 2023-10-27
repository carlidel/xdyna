import NAFFlib
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import xtrack.twiss as xtw
from tqdm.auto import tqdm

from .generic_writer import GenericWriter
from .normed_particles import NormedParticles
from .tools import birkhoff_weights


def frequency_map_analysis_naff(
    line: xt.Line,
    part: xp.Particles,
    max_turns: int,
    buffer_length: int,
    outfile: GenericWriter,
    twiss: xtw.TwissTable = None,
    id_pos=0,
    nemitt_x=None,
    nemitt_y=None,
    order=2,
    interpolation=0,
    _context=xo.ContextCpu(),
):
    """Evaluates the tunes and the Frequency Map Analysis (FMA) using the NAFF
    algorithm for evaluating the tune. For the parameters to be used for the
    NAFF algorithm, see the documentation of NAFFlib.
    (https://github.com/PyCOMPLETE/NAFFlib)

    Parameters
    ----------
    line : xt.Line
        Line to be used.
    part : xp.Particles
        Particles to be tracked.
    max_turns : int
        Number of turns to be tracked.
    buffer_length : int
        Buffer size to be used for the tune evaluation (in turns). Must be
        smaller or equal half the number of turns.
    _context : xo.Context
        xobjects context to be used.
    outfile : GenericWriter
        Writer object to store the tune data.
    twiss : xtw.TwissTable, optional
        Twiss table to be used for the normalization, by default None.
        If None, no normalization is performed.
    id_pos : int, optional
        Index to the element wanted for the normalization, by default 0.
    nemitt_x : float, optional
        Normalized emittance in the horizontal plane, by default None.
        If None, no normalization is performed.
    nemitt_y : float, optional
        Normalized emittance in the vertical plane, by default None.
        If None, no normalization is performed.
    order : int, optional
        Order of the NAFF algorithm, by default 2.
    interpolation : int, optional
        Interpolation order for the NAFF algorithm, by default 0.
    _context : xo.Context, optional
        xobjects context to be used, by default xo.ContextCpu().
    """
    if buffer_length * 2 > max_turns:
        raise ValueError(
            "The buffer length must be smaller than half the number of turns."
        )
    n_particles = len(part.x)
    storage_x = _context.nplike_array_type((buffer_length, n_particles))
    storage_px = _context.nplike_array_type((buffer_length, n_particles))
    storage_y = _context.nplike_array_type((buffer_length, n_particles))
    storage_py = _context.nplike_array_type((buffer_length, n_particles))

    use_normed_particles = False
    if twiss is not None and nemitt_x is not None and nemitt_y is not None:
        if twiss is None or nemitt_x is None or nemitt_y is None:
            raise ValueError(
                "If twiss is given, nemitt_x and nemitt_y must be given as well."
            )
        use_normed_particles = True

    if use_normed_particles:
        norm_part = NormedParticles(
            twiss, nemitt_x, nemitt_y, _context=_context, part=part, idx_pos=id_pos
        )

    for i in tqdm(range(buffer_length)):
        line.track(part, num_turns=1)

        if use_normed_particles:
            norm_part.phys_to_norm(part)
            storage_x[i] = norm_part.x_norm
            storage_px[i] = norm_part.px_norm
            storage_y[i] = norm_part.y_norm
            storage_py[i] = norm_part.py_norm
        else:
            storage_x[i] = part.x
            storage_px[i] = part.px
            storage_y[i] = part.y
            storage_py[i] = part.py

        alive = part.state > 0

        storage_x[i][~alive] = np.nan
        storage_px[i][~alive] = np.nan
        storage_y[i][~alive] = np.nan
        storage_py[i][~alive] = np.nan

    line.track(part, num_turns=max_turns - buffer_length * 2)

    local_storage_x = np.array(_context.nparray_from_context_array(storage_x))
    local_storage_px = np.array(_context.nparray_from_context_array(storage_px))
    local_storage_y = np.array(_context.nparray_from_context_array(storage_y))
    local_storage_py = np.array(_context.nparray_from_context_array(storage_py))

    for i in tqdm(range(buffer_length)):
        line.track(part, num_turns=1)

        if use_normed_particles:
            norm_part.phys_to_norm(part)
            storage_x[i] = norm_part.x_norm
            storage_px[i] = norm_part.px_norm
            storage_y[i] = norm_part.y_norm
            storage_py[i] = norm_part.py_norm
        else:
            storage_x[i] = part.x
            storage_px[i] = part.px
            storage_y[i] = part.y
            storage_py[i] = part.py

        alive = part.state > 0

        storage_x[i][~alive] = np.nan
        storage_px[i][~alive] = np.nan
        storage_y[i][~alive] = np.nan
        storage_py[i][~alive] = np.nan

    local_storage_x_bis = np.array(_context.nparray_from_context_array(storage_x))
    local_storage_px_bis = np.array(_context.nparray_from_context_array(storage_px))
    local_storage_y_bis = np.array(_context.nparray_from_context_array(storage_y))
    local_storage_py_bis = np.array(_context.nparray_from_context_array(storage_py))

    tune_x = np.array(
        [
            NAFFlib.get_tune(
                vx + 1j * vpx,
                order=order,
                interpolation=interpolation,
            )
            for vx, vpx in zip(local_storage_x, local_storage_px)
        ]
    )
    tune_y = np.array(
        [
            NAFFlib.get_tune(
                vy + 1j * vpy,
                order=order,
                interpolation=interpolation,
            )
            for vy, vpy in zip(local_storage_y, local_storage_py)
        ]
    )

    tune_x_bis = np.array(
        [
            NAFFlib.get_tune(
                vx + 1j * vpx,
                order=order,
                interpolation=interpolation,
            )
            for vx, vpx in zip(local_storage_x_bis, local_storage_px_bis)
        ]
    )
    tune_y_bis = np.array(
        [
            NAFFlib.get_tune(
                vy + 1j * vpy,
                order=order,
                interpolation=interpolation,
            )
            for vy, vpy in zip(local_storage_y_bis, local_storage_py_bis)
        ]
    )

    outfile.write_data(f"tune_x/{0}/{buffer_length}", tune_x)
    outfile.write_data(f"tune_y/{0}/{buffer_length}", tune_y)
    outfile.write_data(f"tune_x/{max_turns-buffer_length}/{max_turns}", tune_x_bis)
    outfile.write_data(f"tune_y/{max_turns-buffer_length}/{max_turns}", tune_y_bis)

    fma = np.sqrt((tune_x - tune_x_bis) ** 2 + (tune_y - tune_y_bis) ** 2)

    outfile.write_data(f"fma/{0}/{max_turns}", fma)


def evaluate_tune_birkhoff(
    line: xt.Line,
    part: xp.Particles,
    samples_from,
    samples_to,
    _context,
    outfile: GenericWriter,
    twiss: xtw.TwissTable = None,
    id_pos=0,
    nemitt_x=None,
    nemitt_y=None,
):
    """Track particles and save tune evaluated with birkhoff weights.

    Parameters
    ----------
    line : xt.Line
        Line to be used.
    part : xp.Particles
        Particles to be tracked.
    samples_from : list[int]
        List of starting values from which to compute the tune.
    samples_to : list[int]
        List of ending values to which to compute the tune.
    _context : xo.Context
        xobjects context to be used.
    outfile : GenericWriter
        Writer object to store the tune data.
    twiss : xtw.TwissTable, optional
        Twiss table to be used for the normalization, by default None.
        If None, no normalization is performed.
    id_pos : int, optional
        Index to the element wanted for the normalization, by default 0.
    nemitt_x : float, optional
        Normalized emittance in the horizontal plane, by default None.
        If None, no normalization is performed.
    nemitt_y : float, optional
        Normalized emittance in the vertical plane, by default None.
        If None, no normalization is performed.
    """
    assert len(samples_from) == len(samples_to)

    samples_length = [s_to - s_from for s_from, s_to in zip(samples_from, samples_to)]

    n_particles = len(part.x)
    birkhoff_list = [
        _context.nparray_to_context_array(birkhoff_weights(s)) for s in samples_length
    ]

    use_normed_particles = False
    if twiss is not None and nemitt_x is not None and nemitt_y is not None:
        if twiss is None or nemitt_x is None or nemitt_y is None:
            raise ValueError(
                "If twiss is given, nemitt_x and nemitt_y must be given as well."
            )
        use_normed_particles = True

    if use_normed_particles:
        norm_part = NormedParticles(
            twiss, nemitt_x, nemitt_y, _context=_context, part=part
        )

    angle_1_x = _context.nplike_array_type(n_particles)
    angle_1_y = _context.nplike_array_type(n_particles)
    angle_2_x = _context.nplike_array_type(n_particles)
    angle_2_y = _context.nplike_array_type(n_particles)

    angle_1_x = 0.0
    angle_1_y = 0.0
    angle_2_x = 0.0
    angle_2_y = 0.0

    sum_birkhoff_x = [
        _context.nplike_array_type(n_particles) for j in range(len(samples_length))
    ]
    sum_birkhoff_y = [
        _context.nplike_array_type(n_particles) for j in range(len(samples_length))
    ]

    for sx, sy in zip(samples_from, samples_to):
        sx = 0.0
        sy = 0.0

    if use_normed_particles:
        angle_1_x = np.angle(norm_part.x_norm + 1j * norm_part.px_norm)
        angle_1_y = np.angle(norm_part.y_norm + 1j * norm_part.py_norm)
    else:
        angle_1_x = np.angle(part.x + 1j * part.px)
        angle_1_y = np.angle(part.y + 1j * part.py)
    angle_1_x[angle_1_x < 0] += 2 * np.pi
    angle_1_y[angle_1_y < 0] += 2 * np.pi

    for time in tqdm(range(1, np.max(samples_to) + 1)):
        line.track(part, num_turns=1)

        if use_normed_particles:
            norm_part.phys_to_norm(part)
            angle_2_x = np.angle(norm_part.x_norm + 1j * norm_part.px_norm)
            angle_2_y = np.angle(norm_part.y_norm + 1j * norm_part.py_norm)
        else:
            angle_2_x = np.angle(part.x + 1j * part.px)
            angle_2_y = np.angle(part.y + 1j * part.py)

        angle_2_x[angle_2_x < 0] += 2 * np.pi
        angle_2_y[angle_2_y < 0] += 2 * np.pi

        alive = part.state > 0

        angle_2_x[~alive] = np.nan
        angle_2_y[~alive] = np.nan

        delta_angle_x = angle_2_x - angle_1_x
        delta_angle_y = angle_2_y - angle_1_y

        delta_angle_x[delta_angle_x < 0] += 2 * np.pi
        delta_angle_y[delta_angle_y < 0] += 2 * np.pi

        for j, (t_from, t_to) in enumerate(zip(samples_from, samples_to)):
            if time > t_from and time <= t_to:
                sum_birkhoff_x[j] += birkhoff_list[j][time - 1 - t_from] * delta_angle_x
                sum_birkhoff_y[j] += birkhoff_list[j][time - 1 - t_from] * delta_angle_y

        angle_1_x = angle_2_x
        angle_1_y = angle_2_y

    for j, (t_from, t_to) in enumerate(zip(samples_from, samples_to)):
        outfile.write_data(
            f"tune_birkhoff_x/{t_from}/{t_to}",
            1 - sum_birkhoff_x[j].get() / (2 * np.pi),
        )
        outfile.write_data(
            f"tune_birkhoff_y/{t_from}/{t_to}",
            1 - sum_birkhoff_y[j].get() / (2 * np.pi),
        )
