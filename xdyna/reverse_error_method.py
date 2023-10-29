import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import xtrack.twiss as xtw

from .generic_writer import GenericWriter
from .normed_particles import NormedParticles


def reverse_error_method(
    part: xp.Particles,
    turns_list,
    line: xt.Line,
    twiss: xtw.TwissTable,
    nemitt,
    out: GenericWriter,
    save_all=False,
    _context=xo.ContextCpu(),
):
    """Evaluate the reverse error method for the given values of turns.

    Parameters
    ----------
    part : xp.Particles
        Particle object to be used as reference.
    turns_list : List[int]
        List of turns to be used for the study.
    line : xt.Line
        Line to be used for the study.
    twiss : xtw.TwissTable
        Twiss table of the line to be used for the normalization.
    nemitt : Tuple[float, float]
        Normalized emittance in the horizontal and vertical planes.
    out : GenericWriter
        Writer object to store the results.
    save_all : bool, optional
        If True, the full particle distribution is saved at the time samples, by default False
    _context : xo.Context, optional
        xobjects context to be used, by default xo.ContextCPU()
    """
    turns_list = np.sort(np.unique(turns_list))
    f_part = part.copy()
    norm_f_part = NormedParticles(
        twiss=twiss,
        nemitt_x=nemitt[0],
        nemitt_y=nemitt[1],
        _context=_context,
        part=f_part,
    )

    out.write_data(
        "initial/x_norm",
        _context.nparray_from_context_array(norm_f_part.x_norm),
    )
    out.write_data(
        "initial/px_norm",
        _context.nparray_from_context_array(norm_f_part.px_norm),
    )
    out.write_data(
        "initial/y_norm",
        _context.nparray_from_context_array(norm_f_part.y_norm),
    )
    out.write_data(
        "initial/py_norm",
        _context.nparray_from_context_array(norm_f_part.py_norm),
    )
    out.write_data(
        "initial/zeta_norm",
        _context.nparray_from_context_array(norm_f_part.zeta_norm),
    )
    out.write_data(
        "initial/pzeta_norm",
        _context.nparray_from_context_array(norm_f_part.pzeta_norm),
    )

    out.write_data("initial/x", _context.nparray_from_context_array(f_part.x))
    out.write_data("initial/px", _context.nparray_from_context_array(f_part.px))
    out.write_data("initial/y", _context.nparray_from_context_array(f_part.y))
    out.write_data("initial/py", _context.nparray_from_context_array(f_part.py))
    out.write_data("initial/zeta", _context.nparray_from_context_array(f_part.zeta))
    out.write_data("initial/ptau", _context.nparray_from_context_array(f_part.ptau))

    current_t = 0
    for i, t in enumerate(turns_list):
        delta_t = t - current_t
        line.track(f_part, num_turns=delta_t)
        current_t = t
        r_part = f_part.copy()
        line.track(r_part, num_turns=t, backtrack=True)

        norm_r_part = NormedParticles(
            twiss=twiss,
            nemitt_x=nemitt[0],
            nemitt_y=nemitt[1],
            _context=_context,
            part=r_part,
        )

        if save_all:
            out.write_data(
                f"forward-backward/{i}/x_norm",
                _context.nparray_from_context_array(norm_r_part.x_norm),
            )
            out.write_data(
                f"forward-backward/{i}/px_norm",
                _context.nparray_from_context_array(norm_r_part.px_norm),
            )
            out.write_data(
                f"forward-backward/{i}/y_norm",
                _context.nparray_from_context_array(norm_r_part.y_norm),
            )
            out.write_data(
                f"forward-backward/{i}/py_norm",
                _context.nparray_from_context_array(norm_r_part.py_norm),
            )
            out.write_data(
                f"forward-backward/{i}/zeta_norm",
                _context.nparray_from_context_array(norm_r_part.zeta_norm),
            )
            out.write_data(
                f"forward-backward/{i}/pzeta_norm",
                _context.nparray_from_context_array(norm_r_part.pzeta_norm),
            )

            out.write_data(
                f"forward-backward/{i}/x",
                _context.nparray_from_context_array(r_part.x),
            )
            out.write_data(
                f"forward-backward/{i}/px",
                _context.nparray_from_context_array(r_part.px),
            )
            out.write_data(
                f"forward-backward/{i}/y",
                _context.nparray_from_context_array(r_part.y),
            )
            out.write_data(
                f"forward-backward/{i}/py",
                _context.nparray_from_context_array(r_part.py),
            )
            out.write_data(
                f"forward-backward/{i}/zeta",
                _context.nparray_from_context_array(r_part.zeta),
            )
            out.write_data(
                f"forward-backward/{i}/ptau",
                _context.nparray_from_context_array(r_part.ptau),
            )

        rem_norm = np.sqrt(
            (norm_f_part.x_norm - norm_r_part.x_norm) ** 2
            + (norm_f_part.px_norm - norm_r_part.px_norm) ** 2
            + (norm_f_part.y_norm - norm_r_part.y_norm) ** 2
            + (norm_f_part.py_norm - norm_r_part.py_norm) ** 2
            + (norm_f_part.zeta_norm - norm_r_part.zeta_norm) ** 2
            + (norm_f_part.pzeta_norm - norm_r_part.pzeta_norm) ** 2
        )

        rem = np.sqrt(
            (f_part.x - r_part.x) ** 2
            + (f_part.px - r_part.px) ** 2
            + (f_part.y - r_part.y) ** 2
            + (f_part.py - r_part.py) ** 2
            + (f_part.zeta - r_part.zeta) ** 2
            + (f_part.ptau - r_part.ptau) ** 2
        )

        out.write_data(f"rem_norm/{i}", _context.nparray_from_context_array(rem_norm))
        out.write_data(f"rem/{i}", _context.nparray_from_context_array(rem))

    # save final at_turn of the forward particles
    out.write_data("at_turn", _context.nparray_from_context_array(f_part.at_turn))
