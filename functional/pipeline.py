"""
The pipeline module contains the primary data structure of ScalaFunctional, Sequence
"""

from __future__ import division, absolute_import

from operator import mul
import collections
from functools import reduce

import json
import csv
import sqlite3
import re

import six
import future.builtins as builtins

from functional.lineage import Lineage
from functional.util import is_iterable, is_primitive, is_namedtuple, identity, CSV_WRITE_MODE
from functional import transformations


class Sequence(object):
    """
    Sequence is a wrapper around any type of sequence which provides access to common
    functional transformations and reductions in a data pipelining style
    """
    def __init__(self, sequence, transform=None):
        # pylint: disable=protected-access
        """
        Takes a sequence and wraps it around a Sequence object.

        If the sequence
        is already an instance of Sequence, __init__ will insure that it is
        at most wrapped exactly once.

        If the sequence is a list or tuple, it is set as the sequence.

        If it is an iterable, then it is expanded into a list then set to the sequence

        If the object does not fit any of these classes, a TypeError is thrown

        :param sequence: sequence of items to wrap in a Sequence
        :return: sequence wrapped in a Sequence
        """
        if isinstance(sequence, Sequence):
            self._base_sequence = sequence._base_sequence
            self._lineage = Lineage(prior_lineage=sequence._lineage)
        elif isinstance(sequence, list) or isinstance(sequence, tuple) or is_iterable(sequence):
            self._base_sequence = sequence
            self._lineage = Lineage()
        else:
            raise TypeError("Given sequence must be an iterable value")
        if transform is not None:
            self._lineage.apply(transform)

    def __iter__(self):
        """
        Return iterator of sequence.

        :return: iterator of sequence
        """
        return self._evaluate()

    def __eq__(self, other):
        """
        Checks for equality with the sequence's equality operator.

        :param other: object to compare to
        :return: true if the underlying sequence is equal to other
        """
        return self.sequence == other

    def __ne__(self, other):
        """
        Checks for inequality with the sequence's inequality operator.

        :param other: object to compare to
        :return: true if the underlying sequence is not equal to other
        """
        return self.sequence != other

    def __hash__(self):
        """
        Return the hash of the sequence.

        :return: hash of sequence
        """
        raise TypeError("unhashable type: Sequence")

    def __repr__(self):
        """
        Return repr using sequence's repr function.

        :return: sequence's repr
        """
        return repr(self.to_list())

    def __str__(self):
        """
        Return string using sequence's string function.

        :return: sequence's string
        """
        return str(self.to_list())

    def __bool__(self):
        """
        Returns True if size is not zero.

        :return: True if size is not zero
        """
        return self.size() != 0

    def __nonzero__(self):
        """
        Returns True if size is not zero.

        :return: True if size is not zero
        """
        return self.size() != 0

    def __getitem__(self, item):
        """
        Gets item at given index.

        :param item: key to use for getitem
        :return: item at index key
        """
        self.cache()
        return _wrap(self.sequence[item])

    def __reversed__(self):
        """
        Return reversed sequence using sequence's reverse function

        :return: reversed sequence
        """
        return self._transform(transformations.reversed_t())

    def __contains__(self, item):
        """
        Checks if item is in sequence.

        :param item: item to check
        :return: True if item is in sequence
        """
        return self.sequence.__contains__(item)

    def __add__(self, other):
        """
        Concatenates sequence with other.

        :param other: sequence to concatenate
        :return: concatenated sequence with other
        """
        if isinstance(other, Sequence):
            return Sequence(self.sequence + other.sequence)
        else:
            return Sequence(self.sequence + other)

    def _evaluate(self):
        """
        Creates and returns an iterator which applies all the transformations in the lineage

        :return: iterator over the transformed sequence
        """
        return self._lineage.evaluate(self._base_sequence)

    def _transform(self, *transforms):
        """
        Copies the given Sequence and appends new transformation
        :param transform: transform to apply or list of transforms to apply
        :return: transformed sequence
        """
        sequence = None
        for transform in transforms:
            if sequence:
                sequence = Sequence(sequence, transform=transform)
            else:
                sequence = Sequence(self, transform=transform)
        return sequence

    @property
    def sequence(self):
        """
        Alias for to_list used internally for brevity

        :return: result of to_list() on sequence
        """
        return self.to_list()

    def cache(self, delete_lineage=False):
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
            self._lineage = Lineage()
        return self

    def head(self):
        """
        Returns the first element of the sequence.

        >>> seq([1, 2, 3]).head()
        1

        Raises IndexError when the sequence is empty.

        >>> seq([]).head()
        Traceback (most recent call last):
         ...
        IndexError: list index out of range

        :return: first element of sequence
        """
        return _wrap(self.sequence[0])

    def first(self):
        """
        Returns the first element of the sequence.

        >>> seq([1, 2, 3]).first()
        1

        Raises IndexError when the sequence is empty.

        >>> seq([]).first()
        Traceback (most recent call last):
         ...
        IndexError: list index out of range

        :return: first element of sequence
        """
        return self.head()

    def head_option(self):
        """
        Returns the first element of the sequence or None, if the sequence is empty.

        >>> seq([1, 2, 3]).head_option()
        1

        >>> seq([]).head_option()
        None

        :return: first element of sequence or None if sequence is empty
        """
        if not self.sequence:
            return None
        return self.head()

    def last(self):
        """
        Returns the last element of the sequence.

        >>> seq([1, 2, 3]).last()
        3

        Raises IndexError when the sequence is empty.

        >>> seq([]).last()
        Traceback (most recent call last):
         ...
        IndexError: list index out of range

        :return: last element of sequence
        """
        return _wrap(self.sequence[-1])

    def last_option(self):
        """
        Returns the last element of the sequence or None, if the sequence is empty.

        >>> seq([1, 2, 3]).last_option()
        3

        >>> seq([]).last_option()
        None

        :return: last element of sequence or None if sequence is empty
        """
        if not self.sequence:
            return None
        return self.last()

    def init(self):
        """
        Returns the sequence, without its last element.

        >>> seq([1, 2, 3]).init()
        [1, 2]

        :return: sequence without last element
        """
        return self._transform(transformations.init_t())

    def tail(self):
        """
        Returns the sequence, without its first element.

        >>> seq([1, 2, 3]).init()
        [2, 3]

        :return: sequence without first element
        """
        return self._transform(transformations.tail_t())

    def inits(self):
        """
        Returns consecutive inits of the sequence.

        >>> seq([1, 2, 3]).inits()
        [[1, 2, 3], [1, 2], [1], []]

        :return: consecutive init()s on sequence
        """
        return self._transform(transformations.inits_t(_wrap))

    def tails(self):
        """
        Returns consecutive tails of the sequence.

        >>> seq([1, 2, 3]).tails()
        [[1, 2, 3], [2, 3], [3], []]

        :return: consecutive tail()s of the sequence
        """
        return self._transform(transformations.tails_t(_wrap))

    def drop(self, n):
        """
        Drop the first n elements of the sequence.

        >>> seq([1, 2, 3, 4, 5]).drop(2)
        [3, 4, 5]

        :param n: number of elements to drop
        :return: sequence without first n elements
        """
        if n <= 0:
            return self._transform(transformations.drop_t(0))
        else:
            return self._transform(transformations.drop_t(n))

    def drop_right(self, n):
        """
        Drops the last n elements of the sequence.

        >>> seq([1, 2, 3, 4, 5]).drop_right(2)
        [1, 2, 3]

        :param n: number of elements to drop
        :return: sequence with last n elements dropped
        """
        return self._transform(transformations.CACHE_T, transformations.drop_right_t(n))

    def drop_while(self, func):
        """
        Drops elements in the sequence while func evaluates to True, then returns the rest.

        >>> seq([1, 2, 3, 4, 5, 1, 2]).drop_while(lambda x: x < 3)
        [3, 4, 5, 1, 2]

        :param func: truth returning function
        :return: elements including and after func evaluates to False
        """
        return self._transform(transformations.drop_while_t(func))

    def take(self, n):
        """
        Take the first n elements of the sequence.

        >>> seq([1, 2, 3, 4]).take(2)
        [1, 2]

        :param n: number of elements to take
        :return: first n elements of sequence
        """
        if n <= 0:
            return self._transform(transformations.take_t(0))
        else:
            return self._transform(transformations.take_t(n))

    def take_while(self, func):
        """
        Take elements in the sequence until func evaluates to False, then return them.

        >>> seq([1, 2, 3, 4, 5, 1, 2]).take_while(lambda x: x < 3)
        [1, 2]

        :param func: truth returning function
        :return: elements taken until func evaluates to False
        """
        return self._transform(transformations.take_while_t(func))

    def union(self, other):
        """
        New sequence with unique elements from self and other.

        >>> seq([1, 1, 2, 3, 3]).union([1, 4, 5])
        [1, 2, 3, 4, 5]

        :param other: sequence to union with
        :return: union of sequence and other
        """
        return self._transform(transformations.union_t(other))

    def intersection(self, other):
        """
        New sequence with unique elements present in sequence and other.

        >>> seq([1, 1, 2, 3]).intersection([2, 3, 4])
        [2, 3]

        :param other: sequence to perform intersection with
        :return: intersection of sequence and other
        """
        return self._transform(transformations.intersection_t(other))

    def difference(self, other):
        """
        New sequence with unique elements present in sequence but not in other.

        >>> seq([1, 2, 3]).difference([2, 3, 4])
        [1]

        :param other: sequence to perform difference with
        :return: difference of sequence and other
        """
        return self._transform(transformations.difference_t(other))

    def symmetric_difference(self, other):
        """
        New sequence with elements in either sequence or other, but not both.

        >>> seq([1, 2, 3, 3]).symmetric_difference([2, 4, 5])
        [1, 3, 4, 5]

        :param other: sequence to perform symmetric difference with
        :return: symmetric difference of sequence and other
        """
        return self._transform(transformations.symmetric_difference_t(other))

    def map(self, func):
        """
        Maps f onto the elements of the sequence.

        >>> seq([1, 2, 3, 4]).map(lambda x: x * -1)
        [-1, -2, -3, -4]

        :param func: function to map with
        :return: sequence with func mapped onto it
        """
        return self._transform(transformations.map_t(func))

    def select(self, func):
        """
        Selects f from the elements of the sequence.

        >>> seq([1, 2, 3, 4]).select(lambda x: x * -1)
        [-1, -2, -3, -4]

        :param func: function to select with
        :return: sequence with func mapped onto it
        """
        return self._transform(transformations.select_t(func))

    def for_each(self, func):
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

    def filter(self, func):
        """
        Filters sequence to include only elements where func is True.

        >>> seq([-1, 1, -2, 2]).filter(lambda x: x > 0)
        [1, 2]

        :param func: function to filter on
        :return: filtered sequence
        """
        return self._transform(transformations.filter_t(func))

    def filter_not(self, func):
        """
        Filters sequence to include only elements where func is False.

        >>> seq([-1, 1, -2, 2]).filter_not(lambda x: x > 0)
        [-1, -2]

        :param func: function to filter_not on
        :return: filtered sequence
        """
        return self._transform(transformations.filter_not_t(func))

    def where(self, func):
        """
        Selects elements where func evaluates to True.

        >>> seq([-1, 1, -2, 2]).where(lambda x: x > 0)
        [1, 2]

        :param func: function to filter on
        :return: filtered sequence
        """
        return self._transform(transformations.where_t(func))

    def count(self, func):
        """
        Counts the number of elements in the sequence which satisfy the predicate func.

        >>> seq([-1, -2, 1, 2]).count(lambda x: x > 0)
        2

        :param func: predicate to count elements on
        :return: count of elements that satisfy predicate
        """
        n = 0
        for element in self:
            if func(element):
                n += 1
        return n

    def len(self):
        """
        Return length of sequence using its length function.

        >>> seq([1, 2, 3]).len()
        3

        :return: length of sequence
        """
        self.cache()
        return len(self._base_sequence)

    def size(self):
        """
        Return size of sequence using its length function.

        :return: size of sequence
        """
        return self.len()

    def empty(self):
        """
        Returns True if the sequence has length zero.

        >>> seq([]).empty()
        True

        >>> seq([1]).empty()
        False

        :return: True if sequence length is zero
        """
        return self.size() == 0

    def non_empty(self):
        """
        Returns True if the sequence does not have length zero.

        >>> seq([]).non_empty()
        False

        >>> seq([1]).non_empty()
        True

        :return: True if sequence length is not zero
        """
        return self.size() != 0

    def any(self):
        """
        Returns True if any element in the sequence has truth value True

        >>> seq([True, False]).any()
        True

        >>> seq([False, False]).any()
        False

        :return: True if any element is True
        """
        return any(self)

    def all(self):
        """
        Returns True if the truth value of all items in the sequence true.

        >>> seq([True, True]).all()
        True

        >>> seq([True, False]).all()
        False

        :return: True if all items truth value evaluates to True
        """
        return all(self)

    def exists(self, func):
        """
        Returns True if an element in the sequence makes func evaluate to True.

        >>> seq([1, 2, 3, 4]).exists(lambda x: x == 2)
        True

        >>> seq([1, 2, 3, 4]).exists(lambda x: x < 0)
        False

        :param func: existence check function
        :return: True if any element satisfies func
        """
        for element in self:
            if func(element):
                return True
        return False

    def for_all(self, func):
        """
        Returns True if all elements in sequence make func evaluate to True.

        >>> seq([1, 2, 3]).for_all(lambda x: x > 0)
        True

        >>> seq([1, 2, -1]).for_all(lambda x: x > 0)
        False

        :param func: function to check truth value of all elements with
        :return: True if all elements make func evaluate to True
        """
        for element in self:
            if not func(element):
                return False
        return True

    def max(self):
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
        Traceback (most recent call last):
         ...
        TypeError: unorderable types: int() < str()

        >>> seq([]).max()
        Traceback (most recent call last):
         ...
        ValueError: max() arg is an empty sequence

        :return: Maximal value of sequence
        """
        return _wrap(max(self))

    def min(self):
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
        Traceback (most recent call last):
         ...
        TypeError: unorderable types: int() < str()

        >>> seq([]).min()
        Traceback (most recent call last):
         ...
        ValueError: min() arg is an empty sequence

        :return: Minimal value of sequence
        """
        return _wrap(min(self))

    def max_by(self, func):
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
        Traceback (most recent call last):
         ...
        ValueError: max() arg is an empty sequence

        :param func: function to compute max by
        :return: Maximal element by func(element)
        """
        return _wrap(max(self, key=func))

    def min_by(self, func):
        """
        Returns the smallest element in the sequence.
        Provided function is used to generate key used to compare the elements.
        If the sequence has multiple minimal elements, only the first one is returned.

        The sequence can not be empty.
        Raises ValueError when the sequence is empty.

        >>> seq([2, 4, 5, 1, 3]).min_by(lambda num: num % 6)
        5

        >>> seq('aa', 'xyz', 'abcd', 'xyy').min_by(len)
        'aa'

        >>> seq([]).min_by(lambda x: x)
        Traceback (most recent call last):
         ...
        ValueError: min() arg is an empty sequence

        :param func: function to compute min by
        :return: Maximal element by func(element)
        """
        return _wrap(min(self, key=func))

    def find(self, func):
        """
        Finds the first element of the sequence that satisfies func. If no such element exists,
        then return None.

        >>> seq(["abc", "ab", "bc"]).find(lambda x: len(x) == 2)
        'ab'

        :param func: function to find with
        :return: first element to satisfy func or None
        """
        for element in self:
            if func(element):
                return element
        return None

    def flatten(self):
        """
        Flattens a sequence of sequences to a single sequence of elements.

        >>> seq([[1, 2], [3, 4], [5, 6]])
        [1, 2, 3, 4, 5, 6]

        :return: flattened sequence
        """
        return self._transform(transformations.flatten_t())

    def flat_map(self, func):
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

    def group_by(self, func):
        """
        Group elements into a list of (Key, Value) tuples where func creates the key and maps
        to values matching that key.

        >>> seq(["abc", "ab", "z", "f", "qw"]).group_by(len)
        [(1, ['z', 'f']), (2, ['ab', 'qw']), (3, ['abc'])]

        :param func: group by result of this function
        :return: grouped sequence
        """
        return self._transform(transformations.group_by_t(func))

    def group_by_key(self):
        """
        Group sequence of (Key, Value) elements by Key.

        >>> seq([('a', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 3), ('c', 0)]).group_by_key()
        [('a', [1]), ('c', [3, 0]), ('b', [2, 3, 4])]

        :return: sequence grouped by key
        """
        return self._transform(transformations.group_by_key_t())

    def reduce_by_key(self, func):
        """
        Reduces a sequence of (Key, Value) using func on each sequence of values.

        >>> seq([('a', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 3), ('c', 0)]) \
                .reduce_by_key(lambda x, y: x + y)
        [('a', 1), ('c', 3), ('b', 9)]

        :param func: reduce each list of values using two parameter, associative func
        :return: Sequence of tuples where the value is reduced with func
        """
        return self._transform(transformations.reduce_by_key_t(func))

    def reduce(self, func):
        """
        Reduce sequence of elements using func.

        >>> seq([1, 2, 3]).reduce(lambda x, y: x + y)
        6

        :param func: two parameter, associative reduce function
        :return: reduced value using func
        """
        return _wrap(reduce(func, self))

    def make_string(self, separator):
        """
        Concatenate the elements of the sequence into a string separated by separator.

        >>> seq([1, 2, 3]).make_string("@")
        '1@2@3'

        :param separator: string separating elements in string
        :return: concatenated string separated by separator
        """
        return separator.join(str(e) for e in self)

    def product(self, projection=None):
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
            if projection:
                return projection(1)
            else:
                return 1
        if self.size() == 1:
            if projection:
                return projection(self.first())
            else:
                return self.first()

        if projection:
            return self.map(projection).reduce(mul)
        else:
            return self.reduce(mul)

    def sum(self, projection=None):
        """
        Takes sum of elements in sequence.

        >>> seq([1, 2, 3, 4]).sum()
        10

        >>> seq([(1, 2), (1, 3), (1, 4)]).sum(lambda x: x[0])
        3

        :param projection: function to project on the sequence before taking the sum
        :return: sum of elements in sequence
        """
        if projection:
            return sum(self.map(projection))
        else:
            return sum(self)

    def average(self, projection=None):
        """
        Takes the average of elements in the sequence

        >>> seq([1, 2]).average()
        1.5

        >>> seq([('a', 1), ('b', 2)]).average(lambda x: x[1])

        :param projection: function to project on the sequence before taking the average
        :return: average of elements in the sequence
        """
        length = self.size()
        if projection:
            return sum(self.map(projection)) / length
        else:
            return sum(self) / length

    def aggregate(self, *args):
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
        seed = None
        result_lambda = identity
        if len(args) == 1:
            func = args[0]
        elif len(args) == 2:
            seed = args[0]
            func = args[1]
        elif len(args) == 3:
            seed = args[0]
            func = args[1]
            result_lambda = args[2]
        else:
            raise ValueError('aggregate takes 1-3 arguments, {0} were given'.format(len(args)))
        if len(args) == 1:
            return result_lambda(self.drop(1).fold_left(self.first(), func))
        else:
            return result_lambda(self.fold_left(seed, func))

    def fold_left(self, zero_value, func):
        """
        Assuming that the sequence elements are of type A, folds from left to right starting with
        the seed value given by zero_value (of type A) using a function of type
        func(current: B, next: A) => B. current represents the folded value so far and next is the
        next element from the sequence to fold into current.

        >>> seq('a', 'b', 'c').fold_left(['start'], lambda current, next: current + [next]))
        ['start', 'a', 'b', 'c']

        :param zero_value: zero value to reduce into
        :param func: Two parameter function as described by function docs
        :return: value from folding values with func into zero_value from left to right.
        """
        result = zero_value
        for element in self:
            result = func(result, element)
        return _wrap(result)

    def fold_right(self, zero_value, func):
        """
        Assuming that the sequence elements are of type A, folds from right to left starting with
        the seed value given by zero_value (of type A) using a function of type
        func(next: A, current: B) => B. current represents the folded value so far and next is the
        next element from the sequence to fold into current.

        >>> seq('a', 'b', 'c').fold_left(['start'], lambda next, current: current + [next])
        ['start', 'c', 'b', a']

        :param zero_value: zero value to reduce into
        :param func: Two parameter function as described by function docs
        :return: value from folding values with func into zero_value from right to left
        """
        result = zero_value
        for element in self.reverse():
            result = func(element, result)
        return _wrap(result)

    def zip(self, sequence):
        """
        Zips the stored sequence with the given sequence.

        >>> seq([1, 2, 3]).zip([4, 5, 6])
        [(1, 4), (2, 5), (3, 6)]

        :param sequence: second sequence to zip
        :return: stored sequence zipped with given sequence
        """
        return self._transform(transformations.zip_t(sequence))

    def zip_with_index(self, start=0):
        """
        Zips the sequence to its index, with the index being the second element of each tuple.

        >>> seq(['a', 'b', 'c']).zip_with_index()
        [('a', 0), ('b', 1), ('c', 2)]

        :return: sequence zipped to its index
        """
        return self._transform(transformations.zip_with_index_t(start))

    def enumerate(self, start=0):
        """
        Uses python enumerate to to zip the sequence with indexes starting at start.

        >>> seq(['a', 'b', 'c']).enumerate(start=1)
        [(1, 'a'), (2, 'b'), (3, 'c')]

        :param start: Beginning of zip
        :return: enumerated sequence starting at start
        """
        return self._transform(transformations.enumerate_t(start))

    def inner_join(self, other):
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
        return self.join(other, 'inner')

    def join(self, other, join_type="inner"):
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
        [('a', (1, 3)), ('b', (2, None)]

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)], "right")
        [('a', (1, 3)), ('c', (None, 4)]

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)], "outer")
        [('a', (1, 3)), ('b', (2, None)), ('c', (None, 4))]

        :param other: sequence to join with
        :param join_type: specifies join_type, may be "left", "right", or "outer"
        :return: side joined sequence of (K, (V, W)) pairs
        """
        return self._transform(transformations.join_t(other, join_type))

    def left_join(self, other):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V)
        pairs and other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs.
        V values will always be present, W values may be present or None.

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)])
        [('a', (1, 3)), ('b', (2, None)]

        :param other: sequence to join with
        :return: left joined sequence of (K, (V, W)) pairs
        """
        return self.join(other, "left")

    def right_join(self, other):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V)
        pairs and other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs.
        W values will always bepresent, V values may be present or None.

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)])
        [('a', (1, 3)), ('b', (2, None)]

        :param other: sequence to join with
        :return: right joined sequence of (K, (V, W)) pairs
        """
        return self.join(other, "right")

    def outer_join(self, other):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V)
        pairs and other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs.
        One of V or W will always be not None, but the other may be None

        >>> seq([('a', 1), ('b', 2)]).outer_join([('a', 3), ('c', 4)], "outer")
        [('a', (1, 3)), ('b', (2, None)), ('c', (None, 4))]

        :param other: sequence to join with
        :return: outer joined sequence of (K, (V, W)) pairs
        """
        return self.join(other, "outer")

    def partition(self, func):
        """
        Partition the sequence based on satisfying the predicate func.

        >>> seq([-1, 1, -2, 2]).partition(lambda x: x < 0)
        ([-1, -2], [1, 2])

        :param func: predicate to partition on
        :return: tuple of partitioned sequences
        """
        return self._transform(transformations.partition_t(_wrap, func))

    def grouped(self, size):
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
        return self._transform(transformations.grouped_t(_wrap, size))

    def sliding(self, size, step=1):
        """
        Groups elements in fixed size blocks by passing a sliding window over them.

        The last window has at least one element but may have less than size elements

        :param size: size of sliding window
        :param step: step size between windows
        :return: sequence of sliding windows
        """
        return self._transform(transformations.sliding_t(_wrap, size, step))

    def sorted(self, key=None, reverse=False):
        """
        Uses python sort and its passed arguments to sort the input.

        >>> seq([2, 1, 4, 3]).sorted()
        [1, 2, 3, 4]

        :param key: sort using key function
        :param reverse: return list reversed or not
        :return: sorted sequence
        """
        return self._transform(transformations.sorted_t(key=key, reverse=reverse))

    def order_by(self, func):
        """
        Orders the input according to func

        >>> seq([(2, 'a'), (1, 'b'), (4, 'c'), (3, 'd')]).order_by(lambda x: x[0])
        [1, 2, 3, 4]

        :param func: order by funciton
        :return: ordered sequence
        """
        return self._transform(transformations.order_by_t(func))

    def reverse(self):
        """
        Returns the reversed sequence.

        >>> seq([1, 2, 3]).reverse()
        [3, 2, 1]

        :return: reversed sequence
        """
        return reversed(self)

    def distinct(self):
        """
        Returns sequence of distinct elements. Elements must be hashable.

        >>> seq([1, 1, 2, 3, 3, 3, 4]).distinct()
        [1, 2, 3, 4]

        :return: sequence of distinct elements
        """
        return self._transform(transformations.distinct_t())

    def distinct_by(self, func):
        """
        Returns sequence of elements who are distinct by the passed function. The return
        value of func must be hashable. When two elements are distinct by func, the first is taken.

        :param func: function to use for determining distinctness
        :return: elements distinct by func
        """
        return self._transform(transformations.distinct_by_t(func))

    def slice(self, start, until):
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

    def to_list(self):
        """
        Converts sequence to list of elements.

        >>> type(seq([]).to_list())
        list

        >>> type(seq([]))
        functional.pipeline.Sequence

        >>> seq([1, 2, 3]).to_list()
        [1, 2, 3]

        :return: list of elements in sequence
        """
        self.cache()
        return self._base_sequence

    def list(self):
        """
        Converts sequence to list of elements.

        >>> type(seq([]).list())
        list

        >>> type(seq([]))
        functional.pipeline.Sequence

        >>> seq([1, 2, 3]).list()
        [1, 2, 3]

        :return: list of elements in sequence
        """
        return self.to_list()

    def to_set(self):
        """
        Converts sequence to a set of elements.

        >>> type(seq([])).to_set()
        set

        >>> type(seq([]))
        functional.pipeline.Sequence

        >>> seq([1, 1, 2, 2]).to_set()
        {1, 2}

        :return:set of elements in sequence
        """
        return set(self.sequence)

    def set(self):
        """
        Converts sequence to a set of elements.

        >>> type(seq([])).to_set()
        set

        >>> type(seq([]))
        functional.pipeline.Sequence

        >>> seq([1, 1, 2, 2]).set()
        {1, 2}

        :return:set of elements in sequence
        """
        return self.to_set()

    def to_dict(self, default=None):
        """
        Converts sequence of (Key, Value) pairs to a dictionary.

        >>> type(seq([('a', 1)]).to_dict())
        dict

        >>> seq([('a', 1), ('b', 2)]).to_dict()
        {'a': 1, 'b': 2}

        :param default: Can be a callable zero argument function. When not None, the returned
            dictionary is a collections.defaultdict with default as value for missing keys. If the
            value is not callable, then a zero argument lambda function is created returning the
            value and used for collections.defaultdict
        :return: dictionary from sequence of (Key, Value) elements
        """
        dictionary = {}
        for e in self.sequence:
            dictionary[e[0]] = e[1]
        if default is None:
            return dictionary
        else:
            if hasattr(default, '__call__'):
                return collections.defaultdict(default, dictionary)
            else:
                return collections.defaultdict(lambda: default, dictionary)

    def dict(self, default=None):
        """
        Converts sequence of (Key, Value) pairs to a dictionary.

        >>> type(seq([('a', 1)]).dict())
        dict

        >>> seq([('a', 1), ('b', 2)]).dict()
        {'a': 1, 'b': 2}

        :param default: Can be a callable zero argument function. When not None, the returned
            dictionary is a collections.defaultdict with default as value for missing keys. If the
            value is not callable, then a zero argument lambda function is created returning the
            value and used for collections.defaultdict
        :return: dictionary from sequence of (Key, Value) elements
        """
        return self.to_dict(default=default)

    def to_file(self, path, delimiter=None,
                mode='w', buffering=-1, encoding=None, errors=None, newline=None):
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
        """
        with builtins.open(path, mode=mode, buffering=buffering, encoding=encoding, errors=errors,
                           newline=newline) as output:
            if delimiter:
                output.write(six.u(self.make_string(delimiter)))
            else:
                output.write(six.u(str(self)))

    def to_jsonl(self, path, mode='w'):
        """
        Saves the sequence to a jsonl file. Each element is mapped using json.dumps then written
        with a newline separating each element.

        :param path: path to write file
        :param mode: mode to write in, defaults to 'w' to overwrite contents
        """
        with builtins.open(path, mode=mode) as output:
            output.write(six.u(
                self.map(json.dumps).make_string('\n') + '\n'
            ))

    def to_json(self, path, root_array=True, mode=CSV_WRITE_MODE):
        """
        Saves the sequence to a json file. If root_array is True, then the sequence will be written
        to json with an array at the root. If it is False, then the sequence will be converted from
        a sequence of (Key, Value) pairs to a dictionary so that the json root is a dictionary.

        :param path: path to write file
        :param root_array: write json root as an array or dictionary
        :param mode: file open mode
        """
        with builtins.open(path, mode=mode) as output:
            if root_array:
                json.dump(self.to_list(), output)
            else:
                json.dump(self.to_dict(), output)

    def to_csv(self, path, mode=CSV_WRITE_MODE, dialect='excel', **fmtparams):
        """
        Saves the sequence to a csv file. Each element should be an iterable which will be expanded
        to the elements of each row.

        :param path: path to write file
        :param dialect: passed to csv.writer
        :param fmtparams: passed to csv.writer
        """
        with builtins.open(path, mode) as output:
            csv_writer = csv.writer(output, dialect=dialect, **fmtparams)
            for row in self:
                csv_writer.writerow([six.u(str(element)) for element in row])

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
                cols = ', '.join(item.keys())
                placeholders = ', '.join('?' * len(item))
                sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, cols, placeholders)
                conn.execute(sql, tuple(item.values()))
            elif is_namedtuple(item):
                cols = ', '.join(item._fields)
                placeholders = ', '.join('?' * len(item))
                sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, cols, placeholders)
                conn.execute(sql, item)
            elif isinstance(item, (list, tuple)):
                placeholders = ', '.join('?' * len(item))
                sql = 'INSERT INTO {} VALUES ({})'.format(table_name, placeholders)
                conn.execute(sql, item)
            else:
                raise TypeError('item must be one of dict, namedtuple, tuple or list got {}'
                                .format(type(item)))

        self.for_each(_insert_item)

    def to_sqlite3(self, conn, target, *args, **kwargs):
        """
        Saves the sequence to sqlite3 database.
        Target table must be created in advance.
        The table schema is inferred from the elements in the sequence
        if only target table name is supplied.

        >>> seq([(1, 'Tom'), (2, 'Jack')])\
                .to_sqlite3('users.db', 'INSERT INTO user (id, name) VALUES (?, ?)')

        >>> seq([{'id': 1, 'name': 'Tom'}, {'id': 2, 'name': 'Jack'}]).to_sqlite3(conn, 'user')

        :param conn: path or sqlite connection, cursor
        :param target: SQL query string or table name
        :param args: passed to sqlite3.connect
        :param kwargs: passed to sqlite3.connect
        """
        insert_regex = re.compile(r'(insert|update)\s+into', flags=re.IGNORECASE)
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
            raise ValueError('conn must be a must be a file path or sqlite3 Connection/Cursor')

    def to_pandas(self, columns=None):
        # pylint: disable=import-error
        """
        Converts sequence to a pandas DataFrame using pandas.DataFrame.from_records

        :param columns: columns for pandas to use
        :return: DataFrame of sequence
        """
        import pandas
        return pandas.DataFrame.from_records(self.to_list(), columns=columns)


def _wrap(value):
    """
    Wraps the passed value in a Sequence if it is not a primitive. If it is a string
    argument it is expanded to a list of characters.

    >>> _wrap(1)
    1

    >>> _wrap("abc")
    ['a', 'b', 'c']

    >>> type(_wrap([1, 2]))
    functional.pipeline.Sequence

    :param value: value to wrap
    :return: wrapped or not wrapped value
    """
    if is_primitive(value):
        return value
    if isinstance(value, dict) or isinstance(value, set) or is_namedtuple(value):
        return value
    elif isinstance(value, collections.Iterable):
        return Sequence(value)
    else:
        return value
