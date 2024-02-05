from __future__ import annotations

import math
from collections.abc import Iterable
from functools import reduce
from itertools import chain, count, islice, takewhile
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Optional, Protocol, Sized, TypeVar, Union, cast

import dill as serializer  # type: ignore
from typing_extensions import TypeAlias

T = TypeVar("T")
U = TypeVar("U")
_T_contra = TypeVar("_T_contra", contravariant=True)


# from typeshed
class SupportsDunderLT(Protocol[_T_contra]):
    def __lt__(self, __other: _T_contra) -> bool:
        ...


class SupportsDunderGT(Protocol[_T_contra]):
    def __gt__(self, __other: _T_contra) -> bool:
        ...


SupportsRichComparison: TypeAlias = Union[SupportsDunderLT[Any], SupportsDunderGT[Any]]
SupportsRichComparisonT = TypeVar(
    "SupportsRichComparisonT", bound=SupportsRichComparison
)  # noqa: Y001

PROTOCOL = serializer.HIGHEST_PROTOCOL
CPU_COUNT = cpu_count()


def is_primitive(val: object) -> bool:
    """
    Checks if the passed value is a primitive type.

    >>> is_primitive(1)
    True

    >>> is_primitive("abc")
    True

    >>> is_primitive(True)
    True

    >>> is_primitive({})
    False

    >>> is_primitive([])
    False

    >>> is_primitive(set([]))
    False

    :param val: value to check
    :return: True if value is a primitive, else False
    """
    return isinstance(val, (str, bool, float, complex, bytes, int))


def is_namedtuple(val: object) -> bool:
    """
    Use Duck Typing to check if val is a named tuple. Checks that val is of type tuple and contains
    the attribute _fields which is defined for named tuples.
    :param val: value to check type of
    :return: True if val is a namedtuple
    """
    val_type = type(val)
    bases = val_type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(val_type, "_fields")
    return all(isinstance(n, str) for n in fields)


def identity(arg: T) -> T:
    """
    Function which returns the argument. Used as a default lambda function.

    >>> obj = object()
    >>> obj is identity(obj)
    True

    :param arg: object to take identity of
    :return: return arg
    """
    return arg


def is_iterable_not_list(val: object) -> bool:
    """
    Check if val is not a list, but is a Iterable type. This is used to determine
    when list() should be called on val

    >>> l = [1, 2]
    >>> is_iterable_not_list(l)
    False
    >>> is_iterable_not_list(iter(l))
    True

    :param val: value to check
    :return: True if it is not a list, but is a Iterable
    """
    return not isinstance(val, list) and isinstance(val, Iterable)


def is_tabulatable(val: object) -> bool:
    return not is_primitive(val) and (
        is_iterable_not_list(val) or is_namedtuple(val) or isinstance(val, list)
    )


def split_every(parts: int, iterable: Iterable[T]) -> Iterable[list[T]]:
    """
    Split an iterable into parts of length parts

    >>> l = iter([1, 2, 3, 4])
    >>> list(split_every(2, l))
    [[1, 2], [3, 4]]

    :param iterable: iterable to split
    :param parts: number of chunks
    :return: return the iterable split in parts
    """
    return takewhile(bool, (list(islice(iterable, parts)) for _ in count()))


def unpack(packed: bytes) -> Optional[list]:
    """
    Unpack the function and args then apply the function to the arguments and return result
    :param packed: input packed tuple of (func, args)
    :return: result of applying packed function on packed args
    """
    func, args = serializer.loads(packed)
    result = func(*args)
    if isinstance(result, Iterable):
        return list(result)
    return None


def pack(func: Callable, args: Iterable) -> bytes:
    """
    Pack a function and the args it should be applied to
    :param func: Function to apply
    :param args: Args to evaluate with
    :return: Packed (func, args) tuple
    """
    return serializer.dumps((func, args), PROTOCOL)


def parallelize(
    func: Callable[[T], U],
    seq: Iterable[T],
    processes: Optional[int] = None,
    partition_size: Optional[int] = None,
):
    """
    Creates an iterable which is lazily computed in parallel from applying func on seq
    :param func: Function to apply
    :param seq: Data to apply to
    :param processes: Number of processes to use in parallel
    :param partition_size: Size of partitions for each parallel process
    :return: Iterable of applying func on seq
    """
    parallel_iter = lazy_parallelize(
        func, seq, processes=processes, partition_size=partition_size
    )
    return chain.from_iterable(parallel_iter)


def lazy_parallelize(
    func: Callable[[T], U],
    seq: Iterable[T],
    processes: Optional[int] = None,
    partition_size: Optional[int] = None,
) -> Iterable[list[U]]:
    """
    Lazily computes an map in parallel, and returns them in pool chunks
    :param func: Function to apply
    :param seq: Data to apply to
    :param processes: Number of processes to use in parallel
    :param partition_size: Size of partitions for each parallel process
    :return: Iterable of chunks where each chunk as func applied to it
    """
    if processes is None or processes < 1:
        processes = CPU_COUNT
    else:
        processes = min(processes, CPU_COUNT)
    partition_size = partition_size or compute_partition_size(
        cast(Sized, seq), processes
    )
    with Pool(processes=processes) as pool:
        partitions = split_every(partition_size, iter(seq))
        packed_partitions = (pack(func, (partition,)) for partition in partitions)
        yield from pool.imap(unpack, packed_partitions)


def compute_partition_size(result: Sized, processes: int) -> int:
    """
    Attempts to compute the partition size to evenly distribute work across processes. Defaults to
    1 if the length of result cannot be determined.

    :param result: Result to compute on
    :param processes: Number of processes to use
    :return: Best partition size
    """
    try:
        return max(math.ceil(len(result) / processes), 1)
    except TypeError:
        return 1


def compose(*functions: Callable) -> Callable:
    """
    Compose all the function arguments together
    :param functions: Functions to compose
    :return: Single composed function
    """
    # pylint: disable=undefined-variable
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def coalesce(*vals: Optional[bool]) -> bool:
    for val in vals:
        if val is not None:
            return val
    raise ValueError(f"All values are unset in: {vals}")
