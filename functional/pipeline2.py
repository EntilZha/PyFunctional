"""
The pipeline module contains the transformations and actions API of PyFunctional
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
from tabulate import tabulate

from functional.execution import ExecutionEngine
from functional.lineage import Lineage
from functional.util import is_iterable, is_primitive, is_namedtuple, is_tabulatable, identity
from functional.io import WRITE_MODE, universal_write_open
from functional import transformations


class Pipe(object):
    """
    Sequence is a wrapper around any type of sequence which provides access to common
    functional transformations and reductions in a data pipeline style
    """
    def __init__(self, sequence, transform=None, engine=None, max_repr_items=None):
        # pylint: disable=protected-access
        """
        Takes a Sequence, list, tuple. or iterable sequence and wraps it around a Sequence object.
        If the sequence is already an instance of Sequence, it will in total be wrapped exactly
        once. A TypeError is raised if sequence is none of these.

        :param sequence: sequence of items to wrap in a Sequence
        :param transform: transformation to apply
        :param engine: execution engine
        :param max_repr_items: maximum number of items to print with repr
        :return: sequence wrapped in a Sequence
        """
        self.engine = engine or ExecutionEngine()

        if isinstance(sequence, Sequence):
            self._max_repr_items = max_repr_items or sequence._max_repr_items
            self._base_sequence = sequence._base_sequence
            self._lineage = Lineage(prior_lineage=sequence._lineage,
                                    engine=engine)

        elif isinstance(sequence, (list, tuple)) or is_iterable(sequence):
            self._max_repr_items = max_repr_items
            self._base_sequence = sequence
            self._lineage = Lineage(engine=engine)

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

    def __hash__(self):
        """
        Return the hash of the sequence.

        :return: hash of sequence
        """
        raise TypeError("unhashable type: Sequence")

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

    def map(self, func):
        """
        Maps f onto the elements of the sequence.

        >>> seq([1, 2, 3, 4]).map(lambda x: x * -1)
        [-1, -2, -3, -4]

        :param func: function to map with
        :return: sequence with func mapped onto it
        """
        return self._transform(transformations.map_t(func))

    def smap(self, func):
        """
        Alias to Sequence.starmap

        starmaps f onto the sequence as itertools.starmap does.

        >>> seq([(2, 3), (-2, 1), (0, 10)]).smap(lambda x, y: x + y)
        [5, -1, 10]

        :param func: function to starmap with
        :return: sequence with func starmapped onto it
        """
        return self._transform(transformations.starmap_t(func))

    def filter(self, func):
        """
        Filters sequence to include only elements where func is True.

        >>> seq([-1, 1, -2, 2]).filter(lambda x: x > 0)
        [1, 2]

        :param func: function to filter on
        :return: filtered sequence
        """
        return self._transform(transformations.filter_t(func))

    def sfilter(self, func):
        """
        Filters sequence to include only elements where func is True.

        :param func: function to filter on
        :return: filtered sequence
        """
        return self._transform(transformations.filter_t(lambda args: func(*args)))

    def max(self, func=None):
        """
        Returns the largest element in the sequence.
        Provided function is used to generate key used to compare the elements.
        If the sequence has multiple maximal elements, only the first one is returned.

        The sequence can not be empty.
        Raises ValueError when the sequence is empty.

        >>> seq([2, 4, 5, 1, 3]).max(lambda num: num % 4)
        3

        >>> seq('aa', 'xyz', 'abcd', 'xyy').max(len)
        'abcd'

        >>> seq([]).max(lambda x: x)
        Traceback (most recent call last):
         ...
        ValueError: max() arg is an empty sequence

        :param func: function to compute max by
        :return: Maximal element by func(element)
        """
        return _wrap(max(self, key=func))

    def min(self, func=None):
        """
        Returns the smallest element in the sequence.
        Provided function is used to generate key used to compare the elements.
        If the sequence has multiple minimal elements, only the first one is returned.

        The sequence can not be empty.
        Raises ValueError when the sequence is empty.

        >>> seq([2, 4, 5, 1, 3]).min(lambda num: num % 6)
        5

        >>> seq('aa', 'xyz', 'abcd', 'xyy').min(len)
        'aa'

        >>> seq([]).min(lambda x: x)
        Traceback (most recent call last):
         ...
        ValueError: min() arg is an empty sequence

        :param func: function to compute min by
        :return: Maximal element by func(element)
        """
        return _wrap(min(self, key=func))

    def to_list(self, n=None):
        """
        Converts sequence to list of elements.

        >>> type(seq([]).to_list())
        list

        >>> type(seq([]))
        functional.pipeline.Sequence

        >>> seq([1, 2, 3]).to_list()
        [1, 2, 3]

        :param n: Take n elements of sequence if not None
        :return: list of elements in sequence
        """
        if n is None:
            self.cache()
            return self._base_sequence
        else:
            return self.cache().take(n).list()


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
    if isinstance(value, (dict, set)) or is_namedtuple(value):
        return value
    elif isinstance(value, collections.Iterable):
        return Sequence(value)
    else:
        return value
