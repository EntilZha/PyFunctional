# pylint: disable=redefined-builtin,missing-docstring

from future.builtins import map, filter, zip, range
from functools import reduce, partial
from itertools import dropwhile, takewhile, islice

import collections
from enum import Enum

from functional.util import dict_item_iter, filterfalse


Transformation = collections.namedtuple(
    'Transformation', ['name', 'function', 'execution_strategies']
)

EXECUTION_STRATEGIES = Enum('EXECUTION_STRATEGIES', 'PRE_COMPUTE')

CACHE_T = Transformation('cache', None, None)


def list_t():
    return Transformation('list', list, None)


def map_t(func):
    return Transformation('map({0})'.format(func.__name__), partial(map, func), None)


def filter_t(func):
    return Transformation('filter({0})'.format(func.__name__), partial(filter, func), None)


def filter_not_t(func):
    return Transformation('filter_not({0})'.format(func.__name__), partial(filterfalse, func), None)


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
        enumerate,
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


def union_t(other):
    return Transformation(
        'union',
        lambda sequence: set(sequence).union(other),
        None
    )


def intersection_t(other):
    return Transformation(
        'intersection',
        lambda sequence: set(sequence).intersection(other),
        None
    )


def difference_t(other):
    return Transformation(
        'difference',
        lambda sequence: set(sequence).difference(other),
        None
    )


def symmetric_difference_t(other):
    return Transformation(
        'symmetric_difference',
        lambda sequence: set(sequence).symmetric_difference(other),
        None
    )


def group_by_key_impl(sequence):
    result = {}
    for element in sequence:
        if result.get(element[0]):
            result.get(element[0]).append(element[1])
        else:
            result[element[0]] = [element[1]]
    return dict_item_iter(result)


def group_by_key_t():
    return Transformation(
        'group_by_key',
        group_by_key_impl,
        None
    )


def reduce_by_key_t(func):
    return Transformation(
        'reduce_by_key({0})'.format(func.__name__),
        lambda sequence: map(
            lambda kv: (kv[0], reduce(func, kv[1])), group_by_key_impl(sequence)
        ),
        None
    )


def group_by_impl(func, sequence):
    result = {}
    for element in sequence:
        if result.get(func(element)):
            result.get(func(element)).append(element)
        else:
            result[func(element)] = [element]
    return dict_item_iter(result)


def group_by_t(func):
    return Transformation(
        'group_by({0})'.format(func.__name__),
        partial(group_by_impl, func),
        None
    )


def grouped_impl(wrap, size, sequence):
    for i in range(0, len(sequence), size):
        yield wrap(sequence[i:i + size])


def grouped_t(wrap, size):
    return Transformation(
        'grouped({0})'.format(size),
        partial(grouped_impl, wrap, size),
        {EXECUTION_STRATEGIES.PRE_COMPUTE}
    )


def partition_t(wrap, func):
    return Transformation(
        'partition({0})'.format(func.__name__),
        lambda sequence: wrap(
            (wrap(filter(func, sequence)), wrap(filter(lambda val: not func(val), sequence)))
        ),
        None
    )


def inner_join_impl(other, sequence):
    seq_dict = {}
    for element in sequence:
        seq_dict[element[0]] = element[1]
    seq_kv = seq_dict
    other_kv = dict(other)
    keys = seq_kv.keys() if len(seq_kv) < len(other_kv) else other_kv.keys()
    result = {}
    for k in keys:
        if k in seq_kv and k in other_kv:
            result[k] = (seq_kv[k], other_kv[k])
    return dict_item_iter(result)


def join_impl(other, join_type, sequence):
    if join_type == "inner":
        return inner_join_impl(other, sequence)
    seq_dict = {}
    for element in sequence:
        seq_dict[element[0]] = element[1]
    seq_kv = seq_dict
    other_kv = dict(other)
    if join_type == "left":
        keys = seq_kv.keys()
    elif join_type == "right":
        keys = other_kv.keys()
    elif join_type == "outer":
        keys = set(list(seq_kv.keys()) + list(other_kv.keys()))
    else:
        raise TypeError("Wrong type of join specified")
    result = {}
    for k in keys:
        result[k] = (seq_kv.get(k), other_kv.get(k))
    return dict_item_iter(result)


def join_t(other, join_type):
    return Transformation(
        '{0}_join'.format(join_type),
        partial(join_impl, other, join_type),
        None
    )
