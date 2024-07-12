"""Defines helper functions for training and plotting."""

from functools import partial
from typing import Callable

from ..distributions import BernsteinFlow


def gen_flow(**kwds: int) -> Callable[..., BernsteinFlow]:
    """Generate a Bernstein flow factory.

    Parameters
    ----------
    kwds
        Keyword arguments to pass to :meth:`BernsteinFlow.new`.

    Returns
    -------
    FlowFactory
        A factory function that creates Bernstein flows.

    """
    return partial(BernsteinFlow.new, **kwds)
