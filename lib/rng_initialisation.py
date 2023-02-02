import numpy
import numpy.random as npr

def get_seed(*data):
    """Get a unique seed for given data using numpy SeedSequence.

    Returns:
        int: The seed.
    """
    ss = npr.SeedSequence([abs(x.__hash__()) for x in data])
    return int(npr.default_rng(ss.spawn(1)[0]).integers(0, 2**32, 1)[0])