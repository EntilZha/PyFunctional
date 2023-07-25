from functools import partial
from itertools import (
    dropwhile,
    takewhile,
    islice,
    count,
    product,
    chain,
    starmap,
    filterfalse,
)
import collections
import types
from collections.abc import Callable

from functional.execution import ExecutionStrategies


#: Defines a Transformation from a name, function, and execution_strategies
Transformation = collections.namedtuple(
    "Transformation", ["name", "function", "execution_strategies"]
)

#: Cache transformation
CACHE_T = Transformation("cache", None, None)


def name(function: Callable):
    """
    Retrieve a pretty name for the function
    :param function: function to get name from
    :return: pretty name
    """
    if isinstance(function, types.FunctionType):
        return function.__name__
    else:
        return str(function)


def map_t(func: Callable):
    """
    Transformation for Sequence.map
    :param func: map function
    :return: transformation
    """
    return Transformation(
        f"map({name(func)})",
        partial(map, func),
        {ExecutionStrategies.PARALLEL},
    )


def select_t(func: Callable):
    """
    Transformation for Sequence.select
    :param func: select function
    :return: transformation
    """
    return Transformation(
        "select({name(func)})",
        partial(map, func),
        {ExecutionStrategies.PARALLEL},
    )


def starmap_t(func: Callable):
    """
    Transformation for Sequence.starmap and Sequence.smap
    :param func: starmap function
    :return: transformation
    """
    return Transformation(
        "starmap({name(func)})",
        partial(starmap, func),
        {ExecutionStrategies.PARALLEL},
    )


def filter_t(func: Callable):
    """
    Transformation for Sequence.filter
    :param func: filter function
    :return: transformation
    """
    return Transformation(
        f"filter({name(func)})",
        partial(filter, func),
        {ExecutionStrategies.PARALLEL},
    )


def where_t(func: Callable):
    """
    Transformation for Sequence.where
    :param func: where function
    :return: transformation
    """
    return Transformation(
        f"where({name(func)})",
        partial(filter, func),
        {ExecutionStrategies.PARALLEL},
    )


def filter_not_t(func: Callable):
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


def reversed_t():
    """
    Transformation for Sequence.reverse
    :return: transformation
    """
    return Transformation("reversed", reversed, [ExecutionStrategies.PRE_COMPUTE])


def slice_t(start: int, until: int):
    """
    Transformation for Sequence.slice
    :param start: start index
    :param until: until index (does not include element at until)
    :return: transformation
    """
    return Transformation(
        f"slice({start}, {until})",
        lambda sequence: islice(sequence, start, until),
        None,
    )


def distinct_t():
    """
    Transformation for Sequence.distinct
    :return: transformation
    """

    def distinct(sequence):
        seen = set()
        for element in sequence:
            if element in seen:
                continue
            seen.add(element)
            yield element

    return Transformation("distinct", distinct, None)


def distinct_by_t(func: Callable):
    """
    Transformation for Sequence.distinct_by
    :param func: distinct_by function
    :return: transformation
    """

    def distinct_by(sequence):
        distinct_lookup = {}
        for element in sequence:
            key = func(element)
            if key not in distinct_lookup:
                distinct_lookup[key] = element
        return distinct_lookup.values()

    return Transformation(f"distinct_by({name(func)})", distinct_by, None)


def sorted_t(key=None, reverse: bool = False):
    """
    Transformation for Sequence.sorted
    :param key: key to sort by
    :param reverse: reverse or not
    :return: transformation
    """
    return Transformation(
        "sorted", lambda sequence: sorted(sequence, key=key, reverse=reverse), None
    )


def order_by_t(func: Callable):
    """
    Transformation for Sequence.order_by
    :param func: order_by function
    :return: transformation
    """
    return Transformation(
        f"order_by({name(func)})",
        lambda sequence: sorted(sequence, key=func),
        None,
    )


def drop_right_t(n: int):
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
        f"drop_right({n})",
        lambda sequence: sequence[:end_index],
        [ExecutionStrategies.PRE_COMPUTE],
    )


def drop_t(n: int):
    """
    Transformation for Sequence.drop
    :param n: number to drop from left
    :return: transformation
    """
    return Transformation(
        f"drop({n})", lambda sequence: islice(sequence, n, None), None
    )


def drop_while_t(func: Callable):
    """
    Transformation for Sequence.drop_while
    :param func: drops while func is true
    :return: transformation
    """
    return Transformation(f"drop_while({name(func)})", partial(dropwhile, func), None)


def take_t(n: int):
    """
    Transformation for Sequence.take
    :param n: number to take
    :return: transformation
    """
    return Transformation(f"take({n})", lambda sequence: islice(sequence, 0, n), None)


def take_while_t(func: Callable):
    """
    Transformation for Sequence.take_while
    :param func: takes while func is True
    :return: transformation
    """
    return Transformation(f"take_while({name(func)})", partial(takewhile, func), None)


def flat_map_impl(func: Callable, sequence):
    """
    Implementation for flat_map_t
    :param func: function to map
    :param sequence: sequence to flat_map over
    :return: flat_map generator
    """
    for element in sequence:
        for value in func(element):
            yield value


def flat_map_t(func):
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


def flatten_t():
    """
    Transformation for Sequence.flatten
    :return: transformation
    """
    return Transformation(
        "flatten", partial(flat_map_impl, lambda x: x), {ExecutionStrategies.PARALLEL}
    )


def zip_t(zip_sequence):
    """
    Transformation for Sequence.zip
    :param zip_sequence: sequence to zip with
    :return: transformation
    """
    return Transformation(
        "zip(<sequence>)", lambda sequence: zip(sequence, zip_sequence), None
    )


def zip_with_index_t(start):
    """
    Transformation for Sequence.zip_with_index
    :return: transformation
    """
    return Transformation(
        "zip_with_index", lambda sequence: zip(sequence, count(start=start)), None
    )


def enumerate_t(start):
    """
    Transformation for Sequence.enumerate
    :param start: start index for enumerate
    :return: transformation
    """
    return Transformation(
        "enumerate", lambda sequence: enumerate(sequence, start=start), None
    )


def cartesian_t(iterables, repeat):
    """
    Transformation for Sequence.cartesian
    :param iterables: elements for cartesian product
    :param repeat: how many times to repeat iterables
    :return: transformation
    """
    return Transformation(
        "cartesian", lambda sequence: product(sequence, *iterables, repeat=repeat), None
    )


def init_t():
    """
    Transformation for Sequence.init
    :return: transformation
    """
    return Transformation(
        "init", lambda sequence: sequence[:-1], {ExecutionStrategies.PRE_COMPUTE}
    )


def tail_t():
    """
    Transformation for Sequence.tail
    :return: transformation
    """
    return Transformation("tail", lambda sequence: islice(sequence, 1, None), None)


def inits_t(wrap):
    """
    Transformation for Sequence.inits
    :param wrap: wrap children values with this
    :return: transformation
    """
    return Transformation(
        "inits",
        lambda sequence: [
            wrap(sequence[:i]) for i in reversed(range(len(sequence) + 1))
        ],
        {ExecutionStrategies.PRE_COMPUTE},
    )


def tails_t(wrap):
    """
    Transformation for Sequence.tails
    :param wrap: wrap children values with this
    :return: transformation
    """
    return Transformation(
        "tails",
        lambda sequence: [wrap(sequence[i:]) for i in range(len(sequence) + 1)],
        {ExecutionStrategies.PRE_COMPUTE},
    )


def union_t(other):
    """
    Transformation for Sequence.union
    :param other: sequence to union with
    :return: transformation
    """
    return Transformation("union", lambda sequence: set(sequence).union(other), None)


def intersection_t(other):
    """
    Transformation for Sequence.intersection
    :param other: sequence to intersect with
    :return: transformation
    """
    return Transformation(
        "intersection", lambda sequence: set(sequence).intersection(other), None
    )


def difference_t(other):
    """
    Transformation for Sequence.difference
    :param other: sequence to different with
    :return: transformation
    """
    return Transformation(
        "difference", lambda sequence: set(sequence).difference(other), None
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
        None,
    )


def group_by_key_impl(sequence):
    """
    Implementation for group_by_key_t
    :param sequence: sequence to group
    :return: grouped sequence
    """
    result = {}
    for element in sequence:
        if result.get(element[0]):
            result.get(element[0]).append(element[1])
        else:
            result[element[0]] = [element[1]]
    return result.items()


def group_by_key_t():
    """
    Transformation for Sequence.group_by_key
    :return: transformation
    """
    return Transformation("group_by_key", group_by_key_impl, None)


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
        f"reduce_by_key({name(func)})", partial(reduce_by_key_impl, func), None
    )


def accumulate_impl(func, sequence):
    # pylint: disable=no-name-in-module
    """
    Implementation for accumulate
    :param sequence: sequence to accumulate
    :param func: accumulate function
    """
    from itertools import accumulate

    return accumulate(sequence, func)


def accumulate_t(func):
    """
    Transformation for Sequence.accumulate
    """
    return Transformation(
        f"accumulate({name(func)})", partial(accumulate_impl, func), None
    )


def count_by_key_impl(sequence):
    """
    Implementation for count_by_key_t
    :param sequence: sequence of (key, value) pairs
    :return: counts by key
    """
    counter = collections.Counter()
    for key, _ in sequence:
        counter[key] += 1
    return counter.items()


def count_by_key_t():
    """
    Transformation for Sequence.count_by_key
    :return: transformation
    """
    return Transformation("count_by_key", count_by_key_impl, None)


def count_by_value_impl(sequence):
    """
    Implementation for count_by_value_t
    :param sequence: sequence of values
    :return: counts by value
    """
    counter = collections.Counter()
    for e in sequence:
        counter[e] += 1
    return counter.items()


def count_by_value_t():
    """
    Transformation for Sequence.count_by_value
    :return: transformation
    """
    return Transformation("count_by_value", count_by_value_impl, None)


def group_by_impl(func, sequence):
    """
    Implementation for group_by_t
    :param func: grouping function
    :param sequence: sequence to group
    :return: grouped sequence
    """
    result = {}
    for element in sequence:
        if result.get(func(element)):
            result.get(func(element)).append(element)
        else:
            result[func(element)] = [element]
    return result.items()


def group_by_t(func):
    """
    Transformation for Sequence.group_by
    :param func: grouping function
    :return: transformation
    """
    return Transformation(f"group_by({name(func)})", partial(group_by_impl, func), None)


def grouped_impl(size, sequence):
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


def grouped_t(size):
    """
    Transformation for Sequence.grouped
    :param size: size of groups
    :return: transformation
    """
    return Transformation(f"grouped({size})", partial(grouped_impl, size), None)


def sliding_impl(wrap, size, step, sequence):
    """
    Implementation for sliding_t
    :param wrap: wrap children values with this
    :param size: size of window
    :param step: step size
    :param sequence: sequence to create sliding windows from
    :return: sequence of sliding windows
    """
    i = 0
    n = len(sequence)
    while i + size <= n or (step != 1 and i < n):
        yield wrap(sequence[i : i + size])
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
        f"sliding({size}, {step})",
        partial(sliding_impl, wrap, size, step),
        {ExecutionStrategies.PRE_COMPUTE},
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
        f"partition({name(func)})", partial(partition_impl, wrap, func), None
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
        raise TypeError("Wrong type of join specified")
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
    return Transformation(
        f"{join_type}_join", partial(join_impl, other, join_type), None
    )


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
    return Transformation(f"peek({name(func)})", partial(peek_impl, func), None)
