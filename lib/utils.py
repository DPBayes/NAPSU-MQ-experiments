from itertools import chain, combinations

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    s = list(iterable)
    return [set(t) for t in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]

