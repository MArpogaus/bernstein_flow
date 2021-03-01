from functools import partial

from bernstein_flow.distributions import BernsteinFlow


def bernstein_flow(**kwds):
    return partial(BernsteinFlow, **kwds)
