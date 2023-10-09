import xobjects as xo
import xpart as xp
import xtrack as xt

from .generic_writer import GenericWriter


def stability_tracking(
    part: xp.Particles,
    max_turns: int,
    line: xt.Line,
    out: GenericWriter,
    _context=xo.ContextCpu(),
):
    """Evaluate the stability time up to the given number of turns.

    Parameters
    ----------
    part : xp.Particles
        Particle object to be used as reference.
    max_turns : int
        Maximum number of turns to be used for the study.
    line : xt.Line
        Line to be used for the study.
    out : GenericWriter
        Writer object to store the results.
    _context : xo.Context, optional
        xobjects context to be used, by default xo.ContextCPU()
    """
    line.track(part, num_turns=max_turns)
    # save final at_turn of the forward particles
    out.write_data("at_turn", _context.nparray_from_context_array(part.at_turn))
