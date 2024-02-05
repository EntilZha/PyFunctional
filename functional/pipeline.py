"""
The pipeline module contains the transformations and actions API of PyFunctional
"""

from __future__ import annotations

import builtins
import collections
import csv
import itertools
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from functools import partial, reduce, wraps
from numbers import Number
from operator import add, mul
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Literal,
    NoReturn,
    Optional,
    TypeVar,
    TypeVarTuple,
    Union,
    cast,
    overload,
)

from tabulate import tabulate
from typing_extensions import Self

from functional import transformations
from functional.execution import ExecutionEngine, ExecutionStrategies
from functional.io import WRITE_MODE, StrOrBytesPath, universal_write_open
from functional.lineage import Lineage
from functional.util import (
    SupportsRichComparison,
    coalesce,
    identity,
    is_iterable_not_list,
    is_namedtuple,
    is_primitive,
    is_tabulatable,
)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
Ts = TypeVarTuple("Ts")
Tnumber = TypeVar("Tnumber", bound=Number)

Unset = object()


class Sequence(Generic[T], Iterable[T]):
    """
    Sequence is a wrapper around any type of sequence which provides access to common
    functional transformations and reductions in a data pipeline style
    """

    engine: ExecutionEngine
    _max_repr_items: Optional[int]
    _base_sequence: Iterable[T]
    _lineage: Lineage
    no_wrap: Optional[bool]

    def __init__(
        self,
        sequence: Iterable[T],
        transform: Optional[transformations.Transformation] = None,
        engine: Optional[ExecutionEngine] = None,
        max_repr_items: Optional[int] = None,
        no_wrap: Optional[bool] = None,
    ):
        # pylint: disable=protected-access
        """
        Takes a Sequence, list, tuple. or iterable sequence and wraps it around a Sequence object.
        If the sequence is already an instance of Sequence, it will in total be wrapped exactly
        once. A TypeError is raised if sequence is none of these.

        :param sequence: sequence of items to wrap in a Sequence
        :param transform: transformation to apply
        :param engine: execution engine
        :param max_repr_items: maximum number of items to print with repr
        :param no_wrap: default value of no_wrap for functions like first() or last()
        :return: sequence wrapped in a Sequence
        """
        self.engine = engine or ExecutionEngine()
        if isinstance(sequence, Sequence):
            self._max_repr_items: Optional[int] = (
                max_repr_items or sequence._max_repr_items
            )
            self._base_sequence: Union[Iterable, list, tuple] = sequence._base_sequence
            self._lineage: Lineage = Lineage(
                prior_lineage=sequence._lineage, engine=engine
            )
        elif isinstance(sequence, (list, tuple)) or is_iterable_not_list(sequence):
            self._max_repr_items = max_repr_items
            self._base_sequence = sequence
            self._lineage = Lineage(engine=engine)
        else:
            raise TypeError("Given sequence must be an iterable value")
        if transform is not None:
            self._lineage.apply(transform)
        self.no_wrap = no_wrap

    def __iter__(self) -> Iterator[T]:
        """
        Return iterator of sequence.

        :return: iterator of sequence
        """
        return self._evaluate()

    def __eq__(self, other) -> bool:
        """
        Checks for equality with the sequence's equality operator.

        :param other: object to compare to
        :return: true if the underlying sequence is equal to other
        """
        return self.sequence == other

    def __ne__(self, other) -> bool:
        """
        Checks for inequality with the sequence's inequality operator.

        :param other: object to compare to
        :return: true if the underlying sequence is not equal to other
        """
        return self.sequence != other

    def __hash__(self) -> NoReturn:
        """
        Return the hash of the sequence.

        :return: hash of sequence
        """
        raise TypeError("unhashable type: Sequence")

    def __repr__(self) -> str:
        """
        Return repr using sequence's repr function.

        :return: sequence's repr
        """
        items = self.to_list()
        if self._max_repr_items is None or len(items) <= self._max_repr_items:
            return repr(items)
        else:
            return repr(items[: self._max_repr_items])[:-1] + ", ...]"

    def __str__(self) -> str:
        """
        Return string using sequence's string function.

        :return: sequence's string
        """
        return str(self.to_list())

    def __bool__(self) -> bool:
        """
        Returns True if size is not zero.

        :return: True if size is not zero
        """
        return self.len() != 0

    def __nonzero__(self) -> bool:
        """
        Returns True if size is not zero.

        :return: True if size is not zero
        """
        return self.len() != 0

    def __getitem__(self, item: int) -> T | Sequence:
        """
        Gets item at given index.

        :param item: key to use for getitem
        :return: item at index key
        """
        self.cache()
        return _wrap(self.sequence[item])

    def __reversed__(self) -> Sequence[T]:
        """
        Return reversed sequence using sequence's reverse function

        :return: reversed sequence
        """
        return self._transform(transformations.reversed_t())

    def __contains__(self, item) -> bool:
        """
        Checks if item is in sequence.

        :param item: item to check
        :return: True if item is in sequence
        """
        return self.sequence.__contains__(item)

    def __add__(self, other) -> Sequence[T]:
        """
        Concatenates sequence with other.

        :param other: sequence to concatenate
        :return: concatenated sequence with other
        """
        if isinstance(other, Sequence):
            return Sequence(self.sequence + other.sequence, no_wrap=self.no_wrap)
        else:
            return Sequence(self.sequence + other, no_wrap=self.no_wrap)

    def _evaluate(self) -> Iterator:
        """
        Creates and returns an iterator which applies all the transformations in the lineage

        :return: iterator over the transformed sequence
        """
        return self._lineage.evaluate(self._base_sequence)

    def _transform(self, *transforms: transformations.Transformation) -> Sequence:
        """
        Copies the given Sequence and appends new transformation
        :param transform: transform to apply or list of transforms to apply
        :return: transformed sequence
        """
        sequence = self
        for transform in transforms:
            sequence = Sequence(sequence, transform=transform, no_wrap=self.no_wrap)
        return sequence

    @property
    def sequence(self) -> list[T]:
        """
        Alias for to_list used internally for brevity

        :return: result of to_list() on sequence
        """
        return self.to_list()

    def cache(self, delete_lineage: bool = False) -> Self:
        """
        Caches the result of the Sequence so far. This means that any functions applied on the
        pipeline before cache() are evaluated, and the result is stored in the Sequence. This is
        primarily used internally and is no more helpful than to_list() externally. delete_lineage
        allows for cache() to be used in internal initialization calls without the caller having
        knowledge of the internals via the lineage

        :param delete_lineage: If set to True, it will cache then erase the lineage
        """
        if len(self._lineage) == 0 or self._lineage[-1] == transformations.CACHE_T:
            if not isinstance(self._base_sequence, list):
                self._base_sequence = list(self._base_sequence)
                self._lineage.apply(transformations.CACHE_T)
        else:
            self._base_sequence = list(self._evaluate())
            self._lineage.apply(transformations.CACHE_T)
        if delete_lineage:
            self._lineage = Lineage(engine=self.engine)
        return self

    def head(self, no_wrap: Optional[bool] = None) -> T | Sequence:
        """
        Returns the first element of the sequence. Raises IndexError when the sequence is empty.

        >>> seq([1, 2, 3]).head()
        1

        >>> seq([]).head()
        ...
        IndexError: list index out of range

        :param no_wrap: If set to True, the returned value will never be wrapped with Sequence
        :return: first element of sequence
        """
        if coalesce(no_wrap, self.no_wrap, False):
            return self.sequence[0]
        else:
            return _wrap(self.take(1)[0])

    def first(self, no_wrap: Optional[bool] = None) -> T | Sequence:
        """
        Returns the first element of the sequence. Raises IndexError when the sequence is empty.

        >>> seq([1, 2, 3]).first()
        1

        >>> seq([]).first()
        ...
        IndexError: list index out of range

        :param no_wrap: If set to True, the returned value will never be wrapped with Sequence
        :return: first element of sequence
        """
        return self.head(no_wrap=no_wrap)

    def head_option(self, no_wrap: Optional[bool] = None) -> T | Sequence | None:
        """
        Returns the first element of the sequence or None, if the sequence is empty.

        >>> seq([1, 2, 3]).head_option()
        1

        >>> seq([]).head_option()

        :param no_wrap: If set to True, the returned value will never be wrapped with Sequence
        :return: first element of sequence or None if sequence is empty
        """
        if not self.sequence:
            return None
        return self.head(no_wrap=no_wrap)

    def last(self, no_wrap: Optional[bool] = None) -> T | Sequence:
        """
        Returns the last element of the sequence.

        >>> seq([1, 2, 3]).last()
        3

        Raises IndexError when the sequence is empty.

        >>> seq([]).last()
        ...
        IndexError: list index out of range

        :param no_wrap: If set to True, the returned value will never be wrapped with Sequence
        :return: last element of sequence
        """
        if coalesce(no_wrap, self.no_wrap, False):
            return self.sequence[-1]
        else:
            return _wrap(self.sequence[-1])

    def last_option(self, no_wrap: Optional[bool] = None) -> T | Sequence | None:
        """
        Returns the last element of the sequence or None, if the sequence is empty.

        >>> seq([1, 2, 3]).last_option()
        3

        >>> seq([]).last_option()

        :param no_wrap: If set to True, the returned value will never be wrapped with Sequence
        :return: last element of sequence or None if sequence is empty
        """
        if not self.sequence:
            return None
        return self.last(no_wrap=no_wrap)

    def init(self) -> Sequence[T]:
        """
        Returns the sequence, without its last element.

        >>> seq([1, 2, 3]).init()
        [1, 2]

        :return: sequence without last element
        """
        return self.drop_right(1)

    def tail(self) -> Sequence[T]:
        """
        Returns the sequence, without its first element.

        >>> seq([1, 2, 3]).tail()
        [2, 3]

        :return: sequence without first element
        """
        return self._transform(transformations.tail_t())

    def inits(self) -> Sequence[Sequence[T]]:
        """
        Returns consecutive inits of the sequence.

        >>> seq([1, 2, 3]).inits()
        [[1, 2, 3], [1, 2], [1], []]

        :return: consecutive init()s on sequence
        """
        return self._transform(transformations.inits_t(_wrap))

    def tails(self) -> Sequence[Sequence[T]]:
        """
        Returns consecutive tails of the sequence.

        >>> seq([1, 2, 3]).tails()
        [[1, 2, 3], [2, 3], [3], []]

        :return: consecutive tail()s of the sequence
        """
        return self._transform(transformations.tails_t(_wrap))

    @overload
    def cartesian(self, __a: Iterable[U]) -> Sequence[tuple[T, U]]:
        ...

    @overload
    def cartesian(self, __a: Iterable[U], __b: Iterable[V]) -> Sequence[tuple[T, U, V]]:
        ...

    @overload
    def cartesian(
        self, __a: Iterable[U], __b: Iterable[V], __c: Iterable[W]
    ) -> Sequence[tuple[T, U, V, W]]:
        ...

    def cartesian(self, *iterables, repeat=1):
        """
        Returns the cartesian product of the passed iterables with the specified number of
        repetitions.

        Argument `repeat` is passed to itertools.product.

        >>> seq.range(2).cartesian(range(2))
        [(0, 0), (0, 1), (1, 0), (1, 1)]

        :param iterables: elements for cartesian product
        :param kwargs: the variable `repeat` is read from kwargs
        :return: cartesian product
        """
        return self._transform(transformations.cartesian_t(iterables, repeat))

    def drop(self, n: int) -> Sequence[T]:
        """
        Drop the first n elements of the sequence.

        >>> seq([1, 2, 3, 4, 5]).drop(2)
        [3, 4, 5]

        :param n: number of elements to drop
        :return: sequence without first n elements
        """
        return self._transform(transformations.drop_t(max(0, n)))

    def drop_right(self, n: int) -> Sequence[T]:
        """
        Drops the last n elements of the sequence.

        >>> seq([1, 2, 3, 4, 5]).drop_right(2)
        [1, 2, 3]

        :param n: number of elements to drop
        :return: sequence with last n elements dropped
        """
        return self._transform(transformations.drop_right_t(n))

    def drop_while(self, func: Callable[[T], object]) -> Sequence[T]:
        """
        Drops elements in the sequence while func evaluates to True, then returns the rest.

        >>> seq([1, 2, 3, 4, 5, 1, 2]).drop_while(lambda x: x < 3)
        [3, 4, 5, 1, 2]

        :param func: truth returning function
        :return: elements including and after func evaluates to False
        """
        return self._transform(transformations.drop_while_t(func))

    def take(self, n: int) -> Sequence[T]:
        """
        Take the first n elements of the sequence.

        >>> seq([1, 2, 3, 4]).take(2)
        [1, 2]

        :param n: number of elements to take
        :return: first n elements of sequence
        """
        return self._transform(transformations.take_t(max(0, n)))

    def take_while(self, func: Callable[[T], object]) -> Sequence[T]:
        """
        Take elements in the sequence until func evaluates to False, then return them.

        >>> seq([1, 2, 3, 4, 5, 1, 2]).take_while(lambda x: x < 3)
        [1, 2]

        :param func: truth returning function
        :return: elements taken until func evaluates to False
        """
        return self._transform(transformations.take_while_t(func))

    def union(self, other: Sequence[U]) -> Sequence[Union[T, U]]:
        """
        New sequence with unique elements from self and other.

        >>> seq([1, 1, 2, 3, 3]).union([1, 4, 5])
        [1, 2, 3, 4, 5]

        :param other: sequence to union with
        :return: union of sequence and other
        """
        return self._transform(transformations.union_t(other))

    def intersection(self, other: Sequence[T]) -> Sequence[T]:
        """
        New sequence with unique elements present in sequence and other.

        >>> seq([1, 1, 2, 3]).intersection([2, 3, 4])
        [2, 3]

        :param other: sequence to perform intersection with
        :return: intersection of sequence and other
        """
        return self._transform(transformations.intersection_t(other))

    def difference(self, other: Sequence[T]) -> Sequence[T]:
        """
        New sequence with unique elements present in sequence but not in other.

        >>> seq([1, 2, 3]).difference([2, 3, 4])
        [1]

        :param other: sequence to perform difference with
        :return: difference of sequence and other
        """
        return self._transform(transformations.difference_t(other))

    def symmetric_difference(self, other: Sequence[T]) -> Sequence[T]:
        """
        New sequence with elements in either sequence or other, but not both.

        >>> seq([1, 2, 3, 3]).symmetric_difference([2, 4, 5])
        [1, 3, 4, 5]

        :param other: sequence to perform symmetric difference with
        :return: symmetric difference of sequence and other
        """
        return self._transform(transformations.symmetric_difference_t(other))

    def map(self, func: Callable[[T], U]) -> Sequence[U]:
        """
        Maps f onto the elements of the sequence.

        >>> seq([1, 2, 3, 4]).map(lambda x: x * -1)
        [-1, -2, -3, -4]

        :param func: function to map with
        :return: sequence with func mapped onto it
        """
        if func is identity:
            return self  # type: ignore # U is T here but mypy doesn't understand
        return self._transform(transformations.map_t(func))

    def select(self, func: Callable[[T], U]) -> Sequence[U]:
        """Alias for map."""
        return self.map(func)

    def starmap(self: Sequence[tuple[*Ts]], func: Callable[[*Ts], U]) -> Sequence[U]:
        """
        starmaps f onto the sequence as itertools.starmap does.

        >>> seq([(2, 3), (-2, 1), (0, 10)]).starmap(lambda x, y: x + y)
        [5, -1, 10]

        :param func: function to starmap with
        :return: sequence with func starmapped onto it
        """
        return self._transform(transformations.starmap_t(func))

    def smap(self: Sequence[tuple[*Ts]], func: Callable[[*Ts], U]) -> Sequence[U]:
        """
        Alias to Sequence.starmap

        starmaps f onto the sequence as itertools.starmap does.

        >>> seq([(2, 3), (-2, 1), (0, 10)]).smap(lambda x, y: x + y)
        [5, -1, 10]

        :param func: function to starmap with
        :return: sequence with func starmapped onto it
        """
        return self._transform(transformations.starmap_t(func))

    def for_each(self, func: Callable[[T], Any]) -> None:
        """
        Executes func on each element of the sequence.

        >>> l = []
        >>> seq([1, 2, 3, 4]).for_each(l.append)
        >>> l
        [1, 2, 3, 4]

        :param func: function to execute
        """
        for e in self:
            func(e)

    def peek(self, func: Callable[[T], Any]) -> Sequence[T]:
        """
        Executes func on each element of the sequence and returns the element

        >>> seq([1, 2, 3, 4]).peek(print).map(lambda x: x ** 2).to_list()
        1
        2
        3
        4
        [1, 4, 9, 16]

        :param func: function to execute
        """
        return self._transform(transformations.peek_t(func))

    def filter(self, func: Callable[[T], object]) -> Sequence[T]:
        """
        Filters sequence to include only elements where func is True.

        >>> seq([-1, 1, -2, 2]).filter(lambda x: x > 0)
        [1, 2]

        :param func: function to filter on
        :return: filtered sequence
        """
        return self._transform(transformations.filter_t(func))

    def filter_not(self, func: Callable[[T], object]) -> Sequence[T]:
        """
        Filters sequence to include only elements where func is False.

        >>> seq([-1, 1, -2, 2]).filter_not(lambda x: x > 0)
        [-1, -2]

        :param func: function to filter_not on
        :return: filtered sequence
        """
        return self._transform(transformations.filter_not_t(func))

    def where(self, func: Callable[[T], object]) -> Sequence[T]:
        """Alias for filter."""
        return self.filter(func)

    def count(self, func: Callable[[T], object]) -> int:
        """
        Counts the number of elements in the sequence which satisfy the predicate func.

        >>> seq([-1, -2, 1, 2]).count(lambda x: x > 0)
        2

        :param func: predicate to count elements on
        :return: count of elements that satisfy predicate
        """
        return sum(bool(func(element)) for element in self)

    def len(self) -> int:
        """
        Return length of sequence using its length function.

        >>> seq([1, 2, 3]).len()
        3

        :return: length of sequence
        """
        self.cache()
        assert isinstance(self._base_sequence, list)
        return len(self._base_sequence)

    def size(self) -> int:
        """
        Return size of sequence using its length function.

        :return: size of sequence
        """
        return self.len()

    def empty(self) -> bool:
        """
        Returns True if the sequence has length zero.

        >>> seq([]).empty()
        True

        >>> seq([1]).empty()
        False

        :return: True if sequence length is zero
        """
        return self.len() == 0

    def non_empty(self) -> bool:
        """
        Returns True if the sequence does not have length zero.

        >>> seq([]).non_empty()
        False

        >>> seq([1]).non_empty()
        True

        :return: True if sequence length is not zero
        """
        return self.len() != 0

    def any(self) -> bool:
        """
        Returns True if any element in the sequence has truth value True

        >>> seq([True, False]).any()
        True

        >>> seq([False, False]).any()
        False

        :return: True if any element is True
        """
        return any(self)

    def all(self) -> bool:
        """
        Returns True if the truth value of all items in the sequence true.

        >>> seq([True, True]).all()
        True

        >>> seq([True, False]).all()
        False

        :return: True if all items truth value evaluates to True
        """
        return all(self)

    def exists(self, func: Callable[[T], object]) -> bool:
        """
        Returns True if an element in the sequence makes func evaluate to True.

        >>> seq([1, 2, 3, 4]).exists(lambda x: x == 2)
        True

        >>> seq([1, 2, 3, 4]).exists(lambda x: x < 0)
        False

        :param func: existence check function
        :return: True if any element satisfies func
        """
        return any(func(element) for element in self)

    def for_all(self, func: Callable[[T], object]) -> bool:
        """
        Returns True if all elements in sequence make func evaluate to True.

        >>> seq([1, 2, 3]).for_all(lambda x: x > 0)
        True

        >>> seq([1, 2, -1]).for_all(lambda x: x > 0)
        False

        :param func: function to check truth value of all elements with
        :return: True if all elements make func evaluate to True
        """
        return all(func(element) for element in self)

    def max(
        self: Sequence[SupportsRichComparison],
    ) -> SupportsRichComparison | Sequence:
        """
        Returns the largest element in the sequence.
        If the sequence has multiple maximal elements, only the first one is returned.

        The compared objects must have defined comparison methods.
        Raises TypeError when the objects are not comparable.

        The sequence can not be empty.
        Raises ValueError when the sequence is empty.

        >>> seq([2, 4, 5, 1, 3]).max()
        5

        >>> seq('aa', 'xyz', 'abcd', 'xyy').max()
        'xyz'

        >>> seq([1, "a"]).max()
        ...
        TypeError: ...

        >>> seq([]).max()
        ...
        ValueError: max() arg is an empty sequence

        :return: Maximal value of sequence
        """
        return _wrap(max(self))

    def min(
        self: Sequence[SupportsRichComparison],
    ) -> SupportsRichComparison | Sequence:
        """
        Returns the smallest element in the sequence.
        If the sequence has multiple minimal elements, only the first one is returned.

        The compared objects must have defined comparison methods.
        Raises TypeError when the objects are not comparable.

        The sequence can not be empty.
        Raises ValueError when the sequence is empty.

        >>> seq([2, 4, 5, 1, 3]).min()
        1

        >>> seq('aa', 'xyz', 'abcd', 'xyy').min()
        'aa'

        >>> seq([1, "a"]).min()
        ...
        TypeError: unorderable types: int() < str()

        >>> seq([]).min()
        ...
        ValueError: min() arg is an empty sequence

        :return: Minimal value of sequence
        """
        return _wrap(min(self))

    def max_by(self, func: Callable[[T], SupportsRichComparison]) -> T | Sequence:
        """
        Returns the largest element in the sequence.
        Provided function is used to generate key used to compare the elements.
        If the sequence has multiple maximal elements, only the first one is returned.

        The sequence can not be empty.
        Raises ValueError when the sequence is empty.

        >>> seq([2, 4, 5, 1, 3]).max_by(lambda num: num % 4)
        3

        >>> seq('aa', 'xyz', 'abcd', 'xyy').max_by(len)
        'abcd'

        >>> seq([]).max_by(lambda x: x)
        ...
        ValueError: max() arg is an empty sequence

        :param func: function to compute max by
        :return: Maximal element by func(element)
        """
        return _wrap(max(self, key=func))

    def min_by(self, func: Callable[[T], SupportsRichComparison]) -> T | Sequence:
        """
        Returns the smallest element in the sequence.
        Provided function is used to generate key used to compare the elements.
        If the sequence has multiple minimal elements, only the first one is returned.

        The sequence can not be empty.
        Raises ValueError when the sequence is empty.

        >>> seq([2, 4, 5, 1, 3]).min_by(lambda num: num % 5)
        5

        >>> seq('aa', 'xyz', 'abcd', 'xyy').min_by(len)
        'aa'

        >>> seq([]).min_by(lambda x: x)
        ...
        ValueError: min() arg is an empty sequence

        :param func: function to compute min by
        :return: Maximal element by func(element)
        """
        return _wrap(min(self, key=func))

    def find(self, func: Callable[[T], object]) -> T | None:
        """
        Finds the first element of the sequence that satisfies func. If no such element exists,
        then return None.

        >>> seq(["abc", "ab", "bc"]).find(lambda x: len(x) == 2)
        'ab'

        :param func: function to find with
        :return: first element to satisfy func or None
        """
        return next((element for element in self if func(element)), None)

    def flatten(self: Sequence[Iterable[U]]) -> Sequence[U]:
        """
        Flattens a sequence of sequences to a single sequence of elements.

        >>> seq([[1, 2], [3, 4], [5, 6]]).flatten()
        [1, 2, 3, 4, 5, 6]

        :return: flattened sequence
        """
        return self._transform(transformations.flat_map_t(identity))

    def flat_map(self, func: Callable[[T], Iterable[U]]) -> Sequence[U]:
        """
        Applies func to each element of the sequence, which themselves should be sequences.
        Then appends each element of each sequence to a final result

        >>> seq([[1, 2], [3, 4], [5, 6]]).flat_map(lambda x: x)
        [1, 2, 3, 4, 5, 6]

        >>> seq(["a", "bc", "def"]).flat_map(list)
        ['a', 'b', 'c', 'd', 'e', 'f']

        >>> seq([[1], [2], [3]]).flat_map(lambda x: x * 2)
        [1, 1, 2, 2, 3, 3]

        :param func: function to apply to each sequence in the sequence
        :return: application of func to elements followed by flattening
        """
        return self._transform(transformations.flat_map_t(func))

    def group_by(self, func: Callable[[T], U]) -> Sequence[tuple[U, Sequence[T]]]:
        """
        Group elements into a list of (Key, Value) tuples where func creates the key and maps
        to values matching that key.

        >>> seq(["abc", "ab", "z", "f", "qw"]).group_by(len)
        [(3, ['abc']), (2, ['ab', 'qw']), (1, ['z', 'f'])]

        :param func: group by result of this function
        :return: grouped sequence
        """
        return self._transform(transformations.group_by_t(func))

    def group_by_key(self: Sequence[tuple[U, V]]) -> Sequence[tuple[U, Sequence[V]]]:
        """
        Group sequence of (Key, Value) elements by Key.

        >>> seq([('a', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 3), ('c', 0)]).group_by_key()
        [('a', [1]), ('b', [2, 3, 4]), ('c', [3, 0])]

        :return: sequence grouped by key
        """
        return self._transform(transformations.group_by_key_t())

    def reduce_by_key(
        self: Sequence[tuple[U, V]], func: Callable[[V, V], V]
    ) -> Sequence[tuple[U, V]]:
        """
        Reduces a sequence of (Key, Value) using func on each sequence of values.

        >>> seq([('a', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 3), ('c', 0)]) \
                .reduce_by_key(lambda x, y: x + y)
        [('a', 1), ('b', 9), ('c', 3)]

        :param func: reduce each list of values using two parameter, associative func
        :return: Sequence of tuples where the value is reduced with func
        """
        return self._transform(transformations.reduce_by_key_t(func))

    def count_by_key(self: Sequence[tuple[U, V]]) -> Sequence[tuple[U, int]]:
        """
        Reduces a sequence of (Key, Value) by counting each key

        >>> seq([('a', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 3), ('c', 0)]).count_by_key()
        [('a', 1), ('b', 3), ('c', 2)]

        :return: Sequence of tuples where value is the count of each key
        """
        return self._transform(transformations.count_by_key_t())

    def count_by_value(self) -> Sequence[tuple[T, int]]:
        """
        Reduces a sequence of items by counting each unique item

        >>> seq(['a', 'a', 'a', 'b', 'b', 'c', 'd']).count_by_value()
        [('a', 3), ('b', 2), ('c', 1), ('d', 1)]

        :return: Sequence of tuples where value is the count of each key
        """
        return self._transform(transformations.count_by_value_t())

    @overload
    def reduce(self, func: Callable[[T, T], T]) -> T:
        ...

    @overload
    def reduce(self, func: Callable[[U, T], U], initial: U) -> U:
        ...

    def reduce(self, func, initial=Unset):
        """
        Reduce sequence of elements using func. API mirrors functools.reduce

        >>> seq([1, 2, 3]).reduce(lambda x, y: x + y)
        6

        :param func: two parameter, associative reduce function
        :param initial: single optional argument acting as initial value
        :return: reduced value using func
        """
        if initial is Unset:
            return _wrap(reduce(func, self))
        else:
            return _wrap(reduce(func, self, initial))

    def accumulate(self, func: Callable[[T, T], T] = add) -> Sequence[T]:
        """
        Accumulate sequence of elements using func. API mirrors itertools.accumulate

        >>> seq([1, 2, 3]).accumulate(lambda x, y: x + y)
        [1, 3, 6]

        >>> seq(['a', 'b', 'c']).accumulate()
        ['a', 'ab', 'abc']

        :param func: two parameter, associative accumulate function
        :return: accumulated values using func in sequence
        """
        return self._transform(transformations.accumulate_t(func))

    def make_string(self, separator: str) -> str:
        """
        Concatenate the elements of the sequence into a string separated by separator.

        >>> seq([1, 2, 3]).make_string("@")
        '1@2@3'

        :param separator: string separating elements in string
        :return: concatenated string separated by separator
        """
        return separator.join(str(e) for e in self)

    def product(self, projection: Callable[[T], U] = identity) -> U:  # type: ignore
        """
        Takes product of elements in sequence.

        >>> seq([1, 2, 3, 4]).product()
        24

        >>> seq([]).product()
        1

        >>> seq([(1, 2), (1, 3), (1, 4)]).product(lambda x: x[0])
        1

        :param projection: function to project on the sequence before taking the product
        :return: product of elements in sequence
        """
        if self.empty():
            return projection(1)  # type: ignore
        return self.map(projection).reduce(mul)

    def sum(self, projection: Callable[[T], Tnumber] = identity) -> Tnumber:  # type: ignore
        """
        Takes sum of elements in sequence.

        >>> seq([1, 2, 3, 4]).sum()
        10

        >>> seq([(1, 2), (1, 3), (1, 4)]).sum(lambda x: x[0])
        3

        :param projection: function to project on the sequence before taking the sum
        :return: sum of elements in sequence
        """
        return sum(self.map(projection))

    def average(self, projection: Callable[[T], Tnumber] = identity) -> Tnumber:  # type: ignore
        """
        Takes the average of elements in the sequence

        >>> seq([1, 2]).average()
        1.5

        >>> seq([('a', 1), ('b', 3)]).average(lambda x: x[1])
        2.0

        :param projection: function to project on the sequence before taking the average
        :return: average of elements in the sequence
        """
        length = self.len()  # call .len() before because it calls .cache()
        return sum(self.map(projection)) / length

    @overload
    def aggregate(
        self, func: Callable[[T, T], T], result_lambda: Callable[[T], U]
    ) -> U:
        ...

    @overload
    def aggregate(
        self, seed: U, func: Callable[[U, T], U], result_lambda: Callable[[U], V]
    ) -> V:
        ...

    def aggregate(self, func_or_seed, func=None, result_lambda=identity):
        """
        Aggregates the sequence by specified arguments. Its behavior varies depending on if one,
        two, or three arguments are passed. Assuming the type of the sequence is A:

        One Argument: argument specifies a function of the type f(current: B, next: A => result: B.
        current represents results computed so far, and next is the next element to aggregate into
        current in order to return result.

        Two Argument: the first argument is the seed value for the aggregation. The second argument
        is the same as for the one argument case.

        Three Argument: the first two arguments are the same as for one and two argument calls. The
        additional third parameter is a function applied to the result of the aggregation before
        returning the value.

        :param args: options for how to execute the aggregation
        :return: aggregated value
        """
        func, seed = (func, func_or_seed) if func else (func_or_seed, None)
        if seed is None:
            return result_lambda(self.drop(1).fold_left(self.first(), func))
        else:
            return result_lambda(self.fold_left(seed, func))

    def fold_left(self, zero_value: U, func: Callable[[U, T], U]) -> U | Sequence:
        """
        Assuming that the sequence elements are of type A, folds from left to right starting with
        the seed value given by zero_value (of type A) using a function of type
        func(current: B, next: A) => B. current represents the folded value so far and next is the
        next element from the sequence to fold into current.

        >>> seq('a', 'b', 'c').fold_left(['start'], lambda current, next: current + [next])
        ['start', 'a', 'b', 'c']

        :param zero_value: zero value to reduce into
        :param func: Two parameter function as described by function docs
        :return: value from folding values with func into zero_value from left to right.
        """
        result = zero_value
        for element in self:
            result = func(result, element)
        return _wrap(result)

    def fold_right(self, zero_value: U, func: Callable[[T, U], U]) -> U | Sequence:
        """
        Assuming that the sequence elements are of type A, folds from right to left starting with
        the seed value given by zero_value (of type A) using a function of type
        func(next: A, current: B) => B. current represents the folded value so far and next is the
        next element from the sequence to fold into current.

        >>> seq('a', 'b', 'c').fold_right(['start'], lambda next, current: current + [next])
        ['start', 'c', 'b', 'a']

        :param zero_value: zero value to reduce into
        :param func: Two parameter function as described by function docs
        :return: value from folding values with func into zero_value from right to left
        """
        result = zero_value
        for element in self.reverse():
            result = func(element, result)
        return _wrap(result)

    def zip(self, sequence: Iterable[U]) -> Sequence[tuple[T, U]]:
        """
        Zips the stored sequence with the given sequence.

        >>> seq([1, 2, 3]).zip([4, 5, 6])
        [(1, 4), (2, 5), (3, 6)]

        :param sequence: second sequence to zip
        :return: stored sequence zipped with given sequence
        """
        return self._transform(transformations.zip_t(sequence))

    def zip_with_index(self, start: int = 0) -> Sequence[tuple[T, int]]:
        """
        Zips the sequence to its index, with the index being the second element of each tuple.

        >>> seq(['a', 'b', 'c']).zip_with_index()
        [('a', 0), ('b', 1), ('c', 2)]

        :return: sequence zipped to its index
        """
        return self.zip(itertools.count(start))

    def enumerate(self, start: int = 0) -> Sequence[tuple[int, T]]:
        """
        Uses python enumerate to to zip the sequence with indexes starting at start.

        >>> seq(['a', 'b', 'c']).enumerate(start=1)
        [(1, 'a'), (2, 'b'), (3, 'c')]

        :param start: Beginning of zip
        :return: enumerated sequence starting at start
        """
        return self._transform(transformations.enumerate_t(start))

    def inner_join(
        self: Sequence[tuple[U, V]], other: Sequence[tuple[U, W]]
    ) -> Sequence[tuple[U, tuple[V, W]]]:
        """
        Sequence and other must be composed of (Key, Value) pairs.
        If self.sequence contains (K, V) pairs and other contains (K, W) pairs, the return result
        is a sequence of (K, (V, W)) pairs. Will return only elements
        where the key exists in both sequences.

        >>> seq([('a', 1), ('b', 2), ('c', 3)]).inner_join([('a', 2), ('c', 5)])
        [('a', (1, 2)), ('c', (3, 5))]

        :param other: sequence to join with
        :return: joined sequence of (K, (V, W)) pairs
        """
        return self.join(other, "inner")  # type: ignore

    def join(
        self: Sequence[tuple[U, V]],
        other: Sequence[tuple[U, W]],
        join_type: Literal["inner", "left", "right", "outer"] = "inner",
    ) -> Sequence[tuple[U, Union[tuple[V, Optional[W]], tuple[Optional[V], W]]]]:
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V)
        pairs and other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs.
        If join_type is "left", V values will always be present, W values may be present or None.
        If join_type is "right", W values will always be present, W values may be present or None.
        If join_type is "outer", V or W may be present or None,
        but never at the same time.

        >>> seq([('a', 1), ('b', 2), ('c', 3)]).join([('a', 2), ('c', 5)], "inner")
        [('a', (1, 2)), ('c', (3, 5))]

        >>> seq([('a', 1), ('b', 2), ('c', 3)]).join([('a', 2), ('c', 5)])
        [('a', (1, 2)), ('c', (3, 5))]

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)], "left")
        [('a', (1, 3)), ('b', (2, None))]

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)], "right")
        [('a', (1, 3)), ('c', (None, 4))]

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)], "outer")
        [('a', (1, 3)), ('b', (2, None)), ('c', (None, 4))]

        :param other: sequence to join with
        :param join_type: specifies join_type, may be "left", "right", or "outer"
        :return: side joined sequence of (K, (V, W)) pairs
        """
        return self._transform(transformations.join_t(other, join_type))

    def left_join(
        self: Sequence[tuple[U, V]], other: Sequence[tuple[U, W]]
    ) -> Sequence[tuple[U, tuple[V, Optional[W]]]]:
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V)
        pairs and other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs.
        V values will always be present, W values may be present or None.

        >>> seq([('a', 1), ('b', 2)]).left_join([('a', 3), ('c', 4)])
        [('a', (1, 3)), ('b', (2, None))]

        :param other: sequence to join with
        :return: left joined sequence of (K, (V, W)) pairs
        """
        return self.join(other, "left")  # type: ignore

    def right_join(
        self: Sequence[tuple[U, V]], other: Sequence[tuple[U, W]]
    ) -> Sequence[tuple[U, tuple[Optional[V], W]]]:
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V)
        pairs and other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs.
        W values will always bepresent, V values may be present or None.

        >>> seq([('a', 1), ('b', 2)]).right_join([('a', 3), ('c', 4)])
        [('a', (1, 3)), ('c', (None, 4))]

        :param other: sequence to join with
        :return: right joined sequence of (K, (V, W)) pairs
        """
        return self.join(other, "right")  # type: ignore

    def outer_join(
        self: Sequence[tuple[U, V]], other: Sequence[tuple[U, W]]
    ) -> Sequence[tuple[U, Union[tuple[V, Optional[W]], tuple[Optional[V], W]]]]:
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V)
        pairs and other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs.
        One of V or W will always be not None, but the other may be None

        >>> seq([('a', 1), ('b', 2)]).outer_join([('a', 3), ('c', 4)])
        [('a', (1, 3)), ('b', (2, None)), ('c', (None, 4))]

        :param other: sequence to join with
        :return: outer joined sequence of (K, (V, W)) pairs
        """
        return self.join(other, "outer")

    def partition(self, func: Callable[[T], object]) -> Sequence[Sequence[T]]:
        """
        Partition the sequence based on satisfying the predicate func.

        >>> seq([-1, 1, -2, 2]).partition(lambda x: x < 0)
        [[-1, -2], [1, 2]]

        :param func: predicate to partition on
        :return: tuple of partitioned sequences
        """
        return self._transform(transformations.partition_t(_wrap, func))

    def grouped(self, size: int) -> Sequence[Sequence[T]]:
        """
        Partitions the elements into groups of length size.

        >>> seq([1, 2, 3, 4, 5, 6, 7, 8]).grouped(2)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> seq([1, 2, 3, 4, 5, 6, 7, 8]).grouped(3)
        [[1, 2, 3], [4, 5, 6], [7, 8]]

        The last partition has at least one element but may have less than size elements.

        :param size: size of the partitions
        :return: sequence partitioned into groups of length size
        """
        return self._transform(transformations.grouped_t(size))

    def sliding(self, size: int, step: int = 1) -> Sequence[Sequence[T]]:
        """
        Groups elements in fixed size blocks by passing a sliding window over them.

        The last window has at least one element but may have less than size elements

        :param size: size of sliding window
        :param step: step size between windows
        :return: sequence of sliding windows
        """
        return self._transform(transformations.sliding_t(_wrap, size, step))

    def sorted(
        self,
        key: Callable[[T], SupportsRichComparison] | None = None,
        reverse: bool = False,
    ) -> Sequence[T]:
        """
        Uses python sort and its passed arguments to sort the input.

        >>> seq([2, 1, 4, 3]).sorted()
        [1, 2, 3, 4]

        >>> seq([(2, 'a'), (1, 'b'), (4, 'c'), (3, 'd')]).order_by(lambda x: x[0])
        [(1, 'b'), (2, 'a'), (3, 'd'), (4, 'c')]

        :param key: sort using key function
        :param reverse: return list reversed or not
        :return: sorted sequence
        """
        return self._transform(transformations.sorted_t(key=key, reverse=reverse))

    def order_by(self, func: Callable[[T], SupportsRichComparison]) -> Sequence[T]:
        """Alias for sorted."""
        return self._transform(transformations.sorted_t(key=func))

    def reverse(self) -> reversed:
        """
        Returns the reversed sequence.

        >>> seq([1, 2, 3]).reverse()
        [3, 2, 1]

        :return: reversed sequence
        """
        return reversed(self)  # type: ignore
        # __reversed__ is supposed to return an iterator but ours does not :/ it's a Sequence (can't call next())

    def distinct(self) -> Sequence[T]:
        """
        Returns sequence of distinct elements. Elements must be hashable.

        >>> seq([1, 1, 2, 3, 3, 3, 4]).distinct()
        [1, 2, 3, 4]

        :return: sequence of distinct elements
        """
        return self.distinct_by(identity)

    def distinct_by(self, func: Callable[[T], Hashable]) -> Sequence[T]:
        """
        Returns sequence of elements who are distinct by the passed function. The return
        value of func must be hashable. When two elements are distinct by func, the first is taken.

        :param func: function to use for determining distinctness
        :return: elements distinct by func
        """
        return self._transform(transformations.distinct_by_t(func))

    def slice(self, start: int, until: int) -> Sequence[T]:
        """
        Takes a slice of the sequence starting at start and until but not including until.

        >>> seq([1, 2, 3, 4]).slice(1, 2)
        [2]
        >>> seq([1, 2, 3, 4]).slice(1, 3)
        [2, 3]

        :param start: starting index
        :param until: ending index
        :return: slice including start until but not including until
        """
        return self._transform(transformations.slice_t(start, until))

    def to_list(self, n: Optional[int] = None) -> list[T]:
        """
        Converts sequence to list of elements.

        >>> type(seq([]).to_list())
        <class 'list'>

        >>> type(seq([]))
        <class 'functional.pipeline.Sequence'>

        >>> seq([1, 2, 3]).to_list()
        [1, 2, 3]

        :param n: Take n elements of sequence if not None
        :return: list of elements in sequence
        """
        if n is None:
            self.cache()
            assert isinstance(self._base_sequence, list)
            return self._base_sequence
        else:
            return self.cache().take(n).list()

    def list(self, n: Optional[int] = None) -> list[T]:
        """
        Converts sequence to list of elements.

        >>> type(seq([]).list())
        <class 'list'>

        >>> type(seq([]))
        <class 'functional.pipeline.Sequence'>

        >>> seq([1, 2, 3]).list()
        [1, 2, 3]

        :param n: Take n elements of sequence if not None
        :return: list of elements in sequence
        """
        return self.to_list(n=n)

    def to_set(self) -> set[T]:
        """
        Converts sequence to a set of elements.

        >>> type(seq([]).to_set())
        <class 'set'>

        >>> type(seq([]))
        <class 'functional.pipeline.Sequence'>

        >>> seq([1, 1, 2, 2]).to_set()
        {1, 2}

        :return:set of elements in sequence
        """
        return set(self.sequence)

    def set(self) -> set[T]:
        """
        Converts sequence to a set of elements.

        >>> type(seq([]).to_set())
        <class 'set'>

        >>> type(seq([]))
        <class 'functional.pipeline.Sequence'>

        >>> seq([1, 1, 2, 2]).set()
        {1, 2}

        :return:set of elements in sequence
        """
        return self.to_set()

    @overload
    def to_dict(self: Sequence[tuple[U, V]], default: None) -> dict[U, V]:
        ...

    @overload
    def to_dict(
        self: Sequence[tuple[U, V]], default: Callable[[], V]
    ) -> collections.defaultdict[U, V]:
        ...

    def to_dict(
        self: Sequence[tuple[U, V]], default: Callable[[], V] | V | None = None
    ) -> dict[U, V] | collections.defaultdict[U, V]:
        """
        Converts sequence of (Key, Value) pairs to a dictionary.

        >>> type(seq([('a', 1)]).to_dict())
        <class 'dict'>

        >>> seq([('a', 1), ('b', 2)]).to_dict()
        {'a': 1, 'b': 2}

        :param default: Can be a callable zero argument function. When not None, the returned
            dictionary is a collections.defaultdict with default as value for missing keys. If the
            value is not callable, then a zero argument lambda function is created returning the
            value and used for collections.defaultdict
        :return: dictionary from sequence of (Key, Value) elements
        """
        dictionary = dict(self.sequence)
        if default is None:
            return dictionary
        else:
            return collections.defaultdict(
                default if callable(default) else lambda: cast(V, default), dictionary
            )

    @overload
    def dict(self: Sequence[tuple[U, V]]) -> dict[U, V]:
        ...

    @overload
    def dict(
        self: Sequence[tuple[U, V]], default: Callable[[], V]
    ) -> collections.defaultdict[U, V]:
        ...

    def dict(
        self: Sequence[tuple[U, V]], default: Optional[Callable[[], V]] = None
    ) -> dict[U, V] | collections.defaultdict[U, V]:
        """
        Converts sequence of (Key, Value) pairs to a dictionary.

        >>> type(seq([('a', 1)]).dict())
        <class 'dict'>

        >>> seq([('a', 1), ('b', 2)]).dict()
        {'a': 1, 'b': 2}

        :param default: Can be a callable zero argument function. When not None, the returned
            dictionary is a collections.defaultdict with default as value for missing keys. If the
            value is not callable, then a zero argument lambda function is created returning the
            value and used for collections.defaultdict
        :return: dictionary from sequence of (Key, Value) elements
        """
        return self.to_dict(default=default)

    # pylint: disable=too-many-locals
    def to_file(
        self,
        path: StrOrBytesPath,
        delimiter: Optional[str] = None,
        mode: str = "wt",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        compresslevel: int = 9,
        format: Optional[int] = None,
        check: int = -1,
        preset: Optional[int] = None,
        filters: Optional[Iterable[builtins.dict]] = None,
        compression: Optional[str] = None,
    ):
        """
        Saves the sequence to a file by executing str(self) which becomes str(self.to_list()). If
        delimiter is defined will instead execute self.make_string(delimiter)

        :param path: path to write file
        :param delimiter: if defined, will call make_string(delimiter) and save that to file.
        :param mode: file open mode
        :param buffering: passed to builtins.open
        :param encoding: passed to builtins.open
        :param errors: passed to builtins.open
        :param newline: passed to builtins.open
        :param compression: compression format
        :param compresslevel: passed to gzip.open
        :param format: passed to lzma.open
        :param check: passed to lzma.open
        :param preset: passed to lzma.open
        :param filters: passed to lzma.open
        """
        with universal_write_open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            compression=compression,
            compresslevel=compresslevel,
            format=format,
            check=check,
            preset=preset,
            filters=filters,
        ) as output:
            if delimiter:
                output.write(self.make_string(delimiter))
            else:
                output.write(str(self))

    def to_jsonl(
        self,
        path: StrOrBytesPath,
        mode: str = "wb",
        compression: Optional[str] = None,
    ):
        """
        Saves the sequence to a jsonl file. Each element is mapped using json.dumps then written
        with a newline separating each element.

        :param path: path to write file
        :param mode: mode to write in, defaults to 'w' to overwrite contents
        :param compression: compression format
        """
        with universal_write_open(path, mode=mode, compression=compression) as output:
            output.write(
                (self.map(json.dumps).make_string("\n") + "\n").encode("utf-8")
            )

    def to_json(self, path, root_array=True, mode=WRITE_MODE, compression=None):
        """
        Saves the sequence to a json file. If root_array is True, then the sequence will be written
        to json with an array at the root. If it is False, then the sequence will be converted from
        a sequence of (Key, Value) pairs to a dictionary so that the json root is a dictionary.

        :param path: path to write file
        :param root_array: write json root as an array or dictionary
        :param mode: file open mode
        """
        with universal_write_open(path, mode=mode, compression=compression) as output:
            if root_array:
                json.dump(self.to_list(), output)
            else:
                json.dump(self.to_dict(), output)

    def to_csv(
        self,
        path,
        mode=WRITE_MODE,
        dialect="excel",
        compression=None,
        newline="",
        **fmtparams,
    ):
        """
        Saves the sequence to a csv file. Each element should be an iterable which will be expanded
        to the elements of each row.

        :param path: path to write file
        :param mode: file open mode
        :param dialect: passed to csv.writer
        :param fmtparams: passed to csv.writer
        """

        if "b" in mode:
            newline = None

        with universal_write_open(
            path, mode=mode, compression=compression, newline=newline
        ) as output:
            csv_writer = csv.writer(output, dialect=dialect, **fmtparams)
            for row in self:
                csv_writer.writerow([str(element) for element in row])

    def _to_sqlite3_by_query(self, conn, sql):
        """
        Saves the sequence to sqlite3 database by supplied query.
        Each element should be an iterable which will be expanded
        to the elements of each row. Target table must be created in advance.

        :param conn: path or sqlite connection, cursor
        :param sql: SQL query string
        """
        conn.executemany(sql, self)

    def _to_sqlite3_by_table(self, conn, table_name):
        """
        Saves the sequence to the specified table of sqlite3 database.
        Each element can be a dictionary, namedtuple, tuple or list.
        Target table must be created in advance.

        :param conn: path or sqlite connection, cursor
        :param table_name: table name string
        """

        def _insert_item(item):
            if isinstance(item, dict):
                cols = ", ".join(item.keys())
                placeholders = ", ".join("?" * len(item))
                sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
                conn.execute(sql, tuple(item.values()))
            elif is_namedtuple(item):
                cols = ", ".join(item._fields)
                placeholders = ", ".join("?" * len(item))
                sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
                conn.execute(sql, item)
            elif isinstance(item, (list, tuple)):
                placeholders = ", ".join("?" * len(item))
                sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                conn.execute(sql, item)
            else:
                raise TypeError(
                    "item must be one of dict, namedtuple, tuple or list, got:",
                    type(item),
                    item,
                )

        self.for_each(_insert_item)

    def to_sqlite3(
        self,
        conn: StrOrBytesPath | sqlite3.Connection | sqlite3.Cursor,
        target: str,
        *args,
        **kwargs,
    ):
        """
        Saves the sequence to sqlite3 database.
        Target table must be created in advance.
        The table schema is inferred from the elements in the sequence
        if only target table name is supplied.

        >>> seq([(1, 'Tom'), (2, 'Jack')])\
                .to_sqlite3('examples/users.db', 'INSERT INTO user (id, name) VALUES (?, ?)')

        >>> seq([{'id': 1, 'name': 'Tom'}, {'id': 2, 'name': 'Jack'}]).to_sqlite3(conn, 'user')

        :param conn: path or sqlite connection, cursor
        :param target: SQL query string or table name
        :param args: passed to sqlite3.connect
        :param kwargs: passed to sqlite3.connect
        """
        # pylint: disable=no-member
        insert_regex = re.compile(r"(insert|update)\s+into", flags=re.IGNORECASE)
        if insert_regex.match(target):
            insert_f = self._to_sqlite3_by_query
        else:
            insert_f = self._to_sqlite3_by_table

        if isinstance(conn, (sqlite3.Connection, sqlite3.Cursor)):
            insert_f(conn, target)
            conn.commit()
        elif isinstance(conn, str):
            with sqlite3.connect(conn, *args, **kwargs) as input_conn:
                insert_f(input_conn, target)
                input_conn.commit()
        else:
            raise TypeError(
                "conn must be a file path or sqlite3 Connection/Cursor, got:",
                type(conn),
                conn,
            )

    def to_pandas(self, columns=None):
        # pylint: disable=import-error
        """
        Converts sequence to a pandas DataFrame using pandas.DataFrame.from_records

        :param columns: columns for pandas to use
        :return: DataFrame of sequence
        """
        import pandas

        return pandas.DataFrame.from_records(self.to_list(), columns=columns)

    def show(
        self,
        n=10,
        headers=(),
        tablefmt="simple",
        floatfmt="g",
        numalign="decimal",
        stralign="left",
        missingval="",
    ):
        """
        Pretty print first n rows of sequence as a table. See
        https://bitbucket.org/astanin/python-tabulate for details on tabulate parameters

        :param n: Number of rows to show
        :param headers: Passed to tabulate
        :param tablefmt: Passed to tabulate
        :param floatfmt: Passed to tabulate
        :param numalign: Passed to tabulate
        :param stralign: Passed to tabulate
        :param missingval: Passed to tabulate
        """
        formatted_seq = self.tabulate(
            n=n,
            headers=headers,
            tablefmt=tablefmt,
            floatfmt=floatfmt,
            numalign=numalign,
            stralign=stralign,
            missingval=missingval,
        )
        print(formatted_seq)

    def _repr_html_(self):
        """
        Allows  IPython render HTML tables
        :return: First 10 rows of data as an HTML table
        """
        return self.tabulate(10, tablefmt="html")

    def tabulate(
        self,
        n=None,
        headers=(),
        tablefmt="simple",
        floatfmt="g",
        numalign="decimal",
        stralign="left",
        missingval="",
    ):
        """
        Return pretty string table of first n rows of sequence or everything if n is None. See
        https://bitbucket.org/astanin/python-tabulate for details on tabulate parameters

        :param n: Number of rows to show, if set to None return all rows
        :param headers: Passed to tabulate
        :param tablefmt: Passed to tabulate
        :param floatfmt: Passed to tabulate
        :param numalign: Passed to tabulate
        :param stralign: Passed to tabulate
        :param missingval: Passed to tabulate
        """
        self.cache()
        length = self.len()
        if length == 0 or not is_tabulatable(self[0]):
            return None

        if n is None or n >= length:
            rows = self.list()
            message = ""
        else:
            rows = self.take(n).list()
            if tablefmt == "simple":
                message = f"\nShowing {n} of {length} rows"
            elif tablefmt == "html":
                message = f"<p>Showing {n} of {length} rows"
            else:
                message = ""
        if len(headers) == 0 and is_namedtuple(rows[0]):
            headers = rows[0]._fields
        return (
            tabulate(
                rows,
                headers=headers,
                tablefmt=tablefmt,
                floatfmt=floatfmt,
                numalign=numalign,
                stralign=stralign,
                missingval=missingval,
            )
            + message
        )


def _wrap(value):
    """
    Wraps the passed value in a Sequence if it is not a primitive.

    >>> _wrap(1)
    1
    >>> _wrap("abc")
    'abc'

    >>> type(_wrap([1, 2]))
    <class 'functional.pipeline.Sequence'>

    :param value: value to wrap
    :return: wrapped or not wrapped value
    """
    if is_primitive(value):
        return value
    if isinstance(value, (dict, set)) or is_namedtuple(value):
        return value
    elif isinstance(value, collections.abc.Iterable):
        try:
            if type(value).__name__ == "DataFrame":
                import pandas

                if isinstance(value, pandas.DataFrame):
                    return Sequence(value.values)
        except ImportError:  # pragma: no cover
            pass

        return Sequence(value)
    else:
        return value


def extend(
    func: Optional[Callable[[Any], Any]] = None,
    aslist: bool = False,
    final: bool = False,
    name: str = "",
    parallel: bool = False,
):
    """
    Function decorator for adding new methods to the Sequence class.

    >>> @extend()
    ... def zip2(it):
    ...     return [(i,i) for i in it]

    >>> seq.range(3).zip2()
    [(0, 0), (1, 1), (2, 2)]


    >>> @extend(aslist=True)
    ... def zip2(it):
    ...     return zip(it,it)

    >>> seq.range(3).zip2()
    [(0, 0), (1, 1), (2, 2)]


    >>> @extend(final=True)
    ... def make_set(it):
    ...     return set(it)
    >>> r = seq([0,1,1]).make_set()
    >>> r
    {0, 1}

    :param func: function to decorate
    :param aslist: if True convert input sequence to list (default False)
    :param final: If True decorated function does not return a sequence. Useful
        for creating functions such as to_list.
    :param name: name of the function (default function definition name)
    :param parallel: if true the function is executed in parallel execution strategy (default False)
    """
    if func is None:
        return partial(extend, aslist=aslist, final=final, name=name, parallel=parallel)
    assert func is not None  # this is for mypy

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # pylint: disable=protected-access

        # do not create a new Sequence - just apply a function
        if final:
            return func(self.sequence, *args, **kwargs)

        if aslist:
            func_ = lambda seq: func(list(seq), *args, **kwargs)
        else:
            func_ = lambda seq: func(seq, *args, **kwargs)

        transform = transformations.Transformation(
            f"extended[{name or func.__name__}]",
            func_,
            {ExecutionStrategies.PARALLEL} if parallel else None,
        )
        return self._transform(transform)

    # dynamically add a new method
    setattr(Sequence, func.__name__, wrapper)
    return wrapper
