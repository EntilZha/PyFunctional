from __future__ import annotations

import collections
import collections.abc
import types
from collections.abc import Callable, Iterable, Set
from functools import partial
from itertools import (
    accumulate,
    chain,
    dropwhile,
    filterfalse,
    islice,
    product,
    starmap,
    takewhile,
)
from typing import TYPE_CHECKING, NamedTuple, Optional, TypeVar

from functional.execution import ExecutionStrategies
from functional.util import identity

if TYPE_CHECKING:
    from functional.pipeline import Sequence


class Transformation(NamedTuple):
    name: str
    function: Callable[[Iterable], Iterable]
    execution_strategies: Set[int] = frozenset()


T = TypeVar("T")

#: Cache transformation
CACHE_T = Transformation("cache", identity)
# this identity will not be used but it's to comply with typing


def name(function: Callable) -> str:
    """
    Retrieve a pretty name for the function
    :param function: function to get name from
    :return: pretty name
    """
    if isinstance(function, types.FunctionType):
        return function.__name__
    else:
        return str(function)


def listify(sequence: Iterable[T]) -> collections.abc.Sequence[T]:
    """
    Convert an iterable to a list
    :param sequence: sequence to convert
    :return: list
    """
    if isinstance(sequence, collections.abc.Sequence):
        return sequence
    return list(sequence)


def map_t(func: Callable) -> Transformation:
    """
    Transformation for Sequence.map
    :param func: map function
    :return: transformation
    """
    return Transformation(
        f"map({name(func)})", partial(map, func), {ExecutionStrategies.PARALLEL}
    )


def starmap_t(func: Callable) -> Transformation:
    """
    Transformation for Sequence.starmap and Sequence.smap
    :param func: starmap function
    :return: transformation
    """
    return Transformation(
        f"starmap({name(func)})", partial(starmap, func), {ExecutionStrategies.PARALLEL}
    )


def filter_t(func: Callable) -> Transformation:
    """
    Transformation for Sequence.filter
    :param func: filter function
    :return: transformation
    """
    return Transformation(
        f"filter({name(func)})", partial(filter, func), {ExecutionStrategies.PARALLEL}
    )


def filter_not_t(func: Callable) -> Transformation:
    """
    Transformation for Sequence.filter_not
    :param func: filter_not function
    :return: transformation
    """
    return Transformation(
        f"filter_not({name(func)})",
        partial(filterfalse, func),
        {ExecutionStrategies.PARALLEL},
    )


def _reverse_iter(iterable: Iterable[T]) -> Iterable[T]:
    """
    Reverse an iterable
    :param iterable: iterable to reverse
    :return: reversed iterable
    """
    try:  # avoid a copy if we can
        return reversed(iterable)  # type: ignore
    except TypeError:
        return reversed(list(iterable))


def reversed_t() -> Transformation:
    """
    Transformation for Sequence.reverse
    :return: transformation
    """
    return Transformation("reversed", _reverse_iter)


def slice_t(start: int, until: int) -> Transformation:
    """
    Transformation for Sequence.slice
    :param start: start index
    :param until: until index (does not include element at until)
    :return: transformation
    """
    return Transformation(
        f"slice({start}, {until})",
        lambda sequence: islice(sequence, start, until),
    )


def distinct_by_t(func: Callable) -> Transformation:
    """
    Transformation for Sequence.distinct_by
    :param func: distinct_by function
    :return: transformation
    """

    def distinct_by(sequence):
        seen = set()
        for element in sequence:
            key = func(element)
            if key not in seen:
                seen.add(key)
                yield element

    return Transformation(f"distinct_by({name(func)})", distinct_by)


def sorted_t(key: Optional[Callable] = None, reverse: bool = False):
    """
    Transformation for Sequence.sorted
    :param key: key to sort by
    :param reverse: reverse or not
    :return: transformation
    """
    return Transformation("sorted", partial(sorted, key=key, reverse=reverse))


def drop_right_t(n: int) -> Transformation:
    """
    Transformation for Sequence.drop_right
    :param n: number to drop from right
    :return: transformation
    """
    if n <= 0:
        end_index = None
    else:
        end_index = -n
    return Transformation(
        f"drop_right({n})", lambda sequence: listify(sequence)[:end_index]
    )


def drop_t(n: int) -> Transformation:
    """
    Transformation for Sequence.drop
    :param n: number to drop from left
    :return: transformation
    """
    return Transformation(f"drop({n})", lambda sequence: islice(sequence, n, None))


def drop_while_t(func: Callable) -> Transformation:
    """
    Transformation for Sequence.drop_while
    :param func: drops while func is true
    :return: transformation
    """
    return Transformation(f"drop_while({name(func)})", partial(dropwhile, func))


def take_t(n: int) -> Transformation:
    """
    Transformation for Sequence.take
    :param n: number to take
    :return: transformation
    """
    return Transformation(f"take({n})", lambda sequence: islice(sequence, 0, n))


def take_while_t(func: Callable) -> Transformation:
    """
    Transformation for Sequence.take_while
    :param func: takes while func is True
    :return: transformation
    """
    return Transformation(f"take_while({name(func)})", partial(takewhile, func))


def flat_map_impl(func: Callable, sequence):
    """
    Implementation for flat_map_t
    :param func: function to map
    :param sequence: sequence to flat_map over
    :return: flat_map generator
    """
    for element in sequence:
        yield from func(element)


def flat_map_t(func) -> Transformation:
    """
    Transformation for Sequence.flat_map
    :param func: function to flat_map
    :return: transformation
    """
    return Transformation(
        f"flat_map({name(func)})",
        partial(flat_map_impl, func),
        {ExecutionStrategies.PARALLEL},
    )


def zip_t(zip_sequence) -> Transformation:
    """
    Transformation for Sequence.zip
    :param zip_sequence: sequence to zip with
    :return: transformation
    """
    return Transformation(
        "zip(<sequence>)", lambda sequence: zip(sequence, zip_sequence)
    )


def enumerate_t(start) -> Transformation:
    """
    Transformation for Sequence.enumerate
    :param start: start index for enumerate
    :return: transformation
    """
    return Transformation(
        "enumerate", lambda sequence: enumerate(sequence, start=start)
    )


def cartesian_t(iterables, repeat: int) -> Transformation:
    """
    Transformation for Sequence.cartesian
    :param iterables: elements for cartesian product
    :param repeat: how many times to repeat iterables
    :return: transformation
    """
    return Transformation(
        "cartesian", lambda sequence: product(sequence, *iterables, repeat=repeat)
    )


def tail_t() -> Transformation:
    """
    Transformation for Sequence.tail
    :return: transformation
    """
    return Transformation("tail", lambda sequence: islice(sequence, 1, None))


def _inits(sequence: Iterable[T], wrap) -> list[Sequence[T]]:
    """
    Implementation for inits_t
    :param sequence: sequence to inits
    :return: inits of sequence
    """
    lseq = listify(sequence)
    return [wrap(lseq[:i]) for i in reversed(range(len(lseq) + 1))]


def inits_t(wrap):
    """
    Transformation for Sequence.inits
    :param wrap: wrap children values with this
    :return: transformation
    """
    return Transformation("inits", partial(_inits, wrap=wrap))


def _tails(sequence: Iterable[T], wrap) -> list[Sequence[T]]:
    """
    Implementation for tails_t
    :param sequence: sequence to tails
    :return: tails of sequence
    """
    lseq = listify(sequence)
    return [wrap(lseq[i:]) for i in range(len(lseq) + 1)]


def tails_t(wrap):
    """
    Transformation for Sequence.tails
    :param wrap: wrap children values with this
    :return: transformation
    """
    return Transformation("tails", partial(_tails, wrap=wrap))


def union_t(other):
    """
    Transformation for Sequence.union
    :param other: sequence to union with
    :return: transformation
    """
    return Transformation("union", lambda sequence: set(sequence).union(other))


def intersection_t(other):
    """
    Transformation for Sequence.intersection
    :param other: sequence to intersect with
    :return: transformation
    """
    return Transformation(
        "intersection", lambda sequence: set(sequence).intersection(other)
    )


def difference_t(other):
    """
    Transformation for Sequence.difference
    :param other: sequence to different with
    :return: transformation
    """
    return Transformation(
        "difference", lambda sequence: set(sequence).difference(other)
    )


def symmetric_difference_t(other):
    """
    Transformation for Sequence.symmetric_difference
    :param other: sequence to symmetric_difference with
    :return: transformation
    """
    return Transformation(
        "symmetric_difference",
        lambda sequence: set(sequence).symmetric_difference(other),
    )


def group_by_key_impl(sequence):
    """
    Implementation for group_by_key_t
    :param sequence: sequence to group
    :return: grouped sequence
    """
    result = {}
    for key, value in sequence:
        result.setdefault(key, []).append(value)
    return result.items()


def group_by_key_t():
    """
    Transformation for Sequence.group_by_key
    :return: transformation
    """
    return Transformation("group_by_key", group_by_key_impl)


def reduce_by_key_impl(func, sequence):
    """
    Implementation for reduce_by_key_t
    :param func: reduce function
    :param sequence: sequence to reduce
    :return: reduced sequence
    """
    result = {}
    for key, value in sequence:
        if key in result:
            result[key] = func(result[key], value)
        else:
            result[key] = value
    return result.items()


def reduce_by_key_t(func):
    """
    Transformation for Sequence.reduce_by_key
    :param func: reduce function
    :return: transformation
    """
    return Transformation(
        f"reduce_by_key({name(func)})", partial(reduce_by_key_impl, func)
    )


def accumulate_impl(func, sequence):
    # pylint: disable=no-name-in-module
    """
    Implementation for accumulate
    :param sequence: sequence to accumulate
    :param func: accumulate function
    """
    return accumulate(sequence, func)


def accumulate_t(func):
    """
    Transformation for Sequence.accumulate
    """
    return Transformation(f"accumulate({name(func)})", partial(accumulate_impl, func))


def count_by_key_impl(sequence):
    """
    Implementation for count_by_key_t
    :param sequence: sequence of (key, value) pairs
    :return: counts by key
    """
    return collections.Counter(key for key, _ in sequence).items()


def count_by_key_t():
    """
    Transformation for Sequence.count_by_key
    :return: transformation
    """
    return Transformation("count_by_key", count_by_key_impl)


def count_by_value_impl(sequence):
    """
    Implementation for count_by_value_t
    :param sequence: sequence of values
    :return: counts by value
    """
    return collections.Counter(sequence).items()


def count_by_value_t():
    """
    Transformation for Sequence.count_by_value
    :return: transformation
    """
    return Transformation("count_by_value", count_by_value_impl)


def group_by_impl(func, sequence):
    """
    Implementation for group_by_t
    :param func: grouping function
    :param sequence: sequence to group
    :return: grouped sequence
    """
    result = {}
    for element in sequence:
        result.setdefault(func(element), []).append(element)
    return result.items()


def group_by_t(func):
    """
    Transformation for Sequence.group_by
    :param func: grouping function
    :return: transformation
    """
    return Transformation(f"group_by({name(func)})", partial(group_by_impl, func))


def grouped_impl(size: int, sequence: Iterable[T]) -> Iterable[list[T]]:
    """
    Implementation for grouped_t
    :param size: size of groups
    :param sequence: sequence to group
    :return: grouped sequence
    """
    iterator = iter(sequence)
    try:
        while True:
            batch = islice(iterator, size)
            yield list(chain((next(batch),), batch))
    except StopIteration:
        return


def grouped_t(size: int) -> Transformation:
    """
    Transformation for Sequence.grouped
    :param size: size of groups
    :return: transformation
    """
    return Transformation(f"grouped({size})", partial(grouped_impl, size))


def sliding_impl(
    wrap, size: int, step: int, sequence: Iterable[T]
) -> Iterable[list[T]]:
    """
    Implementation for sliding_t
    :param wrap: wrap children values with this
    :param size: size of window
    :param step: step size
    :param sequence: sequence to create sliding windows from
    :return: sequence of sliding windows
    """
    lseq = listify(sequence)
    i = 0
    n = len(lseq)
    while i + size <= n or (step != 1 and i < n):
        yield wrap(lseq[i : i + size])
        i += step


def sliding_t(wrap, size, step):
    """
    Transformation for Sequence.sliding
    :param wrap: wrap children values with this
    :param size: size of window
    :param step: step size
    :return: transformation
    """
    return Transformation(
        f"sliding({size}, {step})", partial(sliding_impl, wrap, size, step)
    )


def partition_impl(wrap, predicate, sequence):
    truthy_partition = []
    falsy_partition = []
    for e in sequence:
        if predicate(e):
            truthy_partition.append(e)
        else:
            falsy_partition.append(e)

    return wrap((wrap(truthy_partition), wrap(falsy_partition)))


def partition_t(wrap, func):
    """
    Transformation for Sequence.partition
    :param wrap: wrap children values with this
    :param func: partition function
    :return: transformation
    """
    return Transformation(
        f"partition({name(func)})", partial(partition_impl, wrap, func)
    )


def inner_join_impl(other, sequence):
    """
    Implementation for part of join_impl
    :param other: other sequence to join with
    :param sequence: first sequence to join with
    :return: joined sequence
    """
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
    return result.items()


def join_impl(other, join_type, sequence):
    """
    Implementation for join_t
    :param other: other sequence to join with
    :param join_type: join type (inner, outer, left, right)
    :param sequence: first sequence to join with
    :return: joined sequence
    """
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
        raise ValueError("Wrong type of join specified")
    result = {}
    for k in keys:
        result[k] = (seq_kv.get(k), other_kv.get(k))
    return result.items()


def join_t(other, join_type):
    """
    Transformation for Sequence.join, Sequence.inner_join, Sequence.outer_join, Sequence.right_join,
    and Sequence.left_join
    :param other: other sequence to join with
    :param join_type: join type from left, right, inner, and outer
    :return: transformation
    """
    return Transformation(f"{join_type}_join", partial(join_impl, other, join_type))


def peek_impl(func, sequence):
    """
    Implementation for peek_t
    :param func: apply func
    :param sequence: sequence to peek
    :return: original sequence
    """
    for element in sequence:
        func(element)
        yield element


def peek_t(func: Callable):
    """
    Transformation for Sequence.peek
    :param func: peek function
    :return: transformation
    """
    return Transformation(f"peek({name(func)})", partial(peek_impl, func))
