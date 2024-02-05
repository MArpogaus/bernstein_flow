from functools import partial

from ..distributions import BernsteinFlow


def gen_flow(**kwds):
    return partial(BernsteinFlow.from_pvector, **kwds)
