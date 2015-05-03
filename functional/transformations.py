import sys
from functools import reduce, partial
from itertools import dropwhile, takewhile, islice, ifilterfalse
import collections
from enum import Enum


if sys.version < '3':
    from itertools import imap as map
    from itertools import ifilter as filter
    from itertools import izip as zip
    range = xrange
    dict_item_iter = lambda d: d.viewitems()
else:
    dict_item_iter = lambda d: d.items()

Transformation = collections.namedtuple('Transformation', ['name', 'function', 'execution_strategies'])

EXECUTION_STRATEGIES = Enum('EXECUTION_STRATEGIES', 'PRE_COMPUTE')

CACHE_T = Transformation('cache', None, None)


def list_t():
    return Transformation('list', list, None)


def map_t(func):
    return Transformation('map({0})'.format(func.__name__), partial(map, func), None)


def filter_t(func):
    return Transformation('filter({0})'.format(func.__name__), partial(filter, func), None)


def filter_not_t(func):
    return Transformation('filter_not({0})'.format(func.__name__), partial(ifilterfalse, func), None)


def reversed_t():
    return Transformation('reversed', reversed, None)


def slice_t(start, until):
    return Transformation(
        'slice({0}, {1})'.format(start, until),
        lambda sequence: islice(sequence, start, until),
        None
    )


def distinct_t():
    def distinct(sequence):
        return iter(set(sequence))
    return Transformation('distinct', distinct, None)


def sorted_t(key=None, reverse=False):
    return Transformation(
        'sorted',
        lambda sequence: sorted(sequence, key=key, reverse=reverse),
        None
    )


def drop_right_t(n):
    if n <= 0:
        end_index = None
    else:
        end_index = -n
    return Transformation(
        'drop_right({0})'.format(n),
        lambda sequence: sequence[:end_index],
        None
    )


def drop_t(n):
    return Transformation(
        'drop({0})'.format(n),
        lambda sequence: islice(sequence, n, None),
        None
    )


def drop_while_t(func):
    return Transformation(
        'drop_while({0})'.format(func.__name__),
        partial(dropwhile, func),
        None
    )


def take_t(n):
    return Transformation(
        'take({0})'.format(n),
        lambda sequence: islice(sequence, 0, n),
        None
    )


def take_while_t(func):
    return Transformation(
        'take_while({0})'.format(func.__name__),
        partial(takewhile, func),
        None
    )


def flat_map_impl(func, sequence):
        for element in sequence:
            for value in func(element):
                yield value


def flat_map_t(func):
    return Transformation(
        'flat_map({0})'.format(func.__name__),
        partial(flat_map_impl, func),
        None
    )


def flatten_t():
    return Transformation(
        'flatten',
        partial(flat_map_impl, lambda x: x),
        None
    )


def zip_t(zip_sequence):
    return Transformation(
        'zip(<sequence>)',
        lambda sequence: zip(sequence, zip_sequence),
        None
    )


def zip_with_index_t():
    return Transformation(
        'zip_with_index',
        lambda sequence: enumerate(sequence),
        None
    )


def enumerate_t(start):
    return Transformation(
        'enumerate',
        lambda sequence: enumerate(sequence, start=start),
        None
    )


def init_t():
    return Transformation(
        'init',
        lambda sequence: sequence[:-1],
        {EXECUTION_STRATEGIES.PRE_COMPUTE}
    )


def tail_t():
    return Transformation(
        'tail',
        lambda sequence: islice(sequence, 1, None),
        None
    )


def inits_t(wrap):
    return Transformation(
        'inits',
        lambda sequence: [wrap(sequence[:i]) for i in reversed(range(len(sequence) + 1))],
        {EXECUTION_STRATEGIES.PRE_COMPUTE}
    )


def tails_t(wrap):
    return Transformation(
        'tails',
        lambda sequence: [wrap(sequence[i:]) for i in range(len(sequence) + 1)],
        {EXECUTION_STRATEGIES.PRE_COMPUTE}
    )
