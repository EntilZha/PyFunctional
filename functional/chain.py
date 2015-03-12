import collections
import sys
from functools import reduce


class FunctionalSequence(object):
    def __init__(self, sequence):
        if isinstance(sequence, FunctionalSequence):
            self.sequence = sequence._get_base_sequence()
        elif isinstance(sequence, list):
            self.sequence = sequence
        elif isinstance(sequence, collections.Iterable):
            self.sequence = list(sequence)
        else:
            raise TypeError("Given sequence must be a list")

    def _get_base_sequence(self):
        if isinstance(self.sequence, FunctionalSequence):
            return self.sequence._get_base_sequence()
        else:
            return self.sequence

    def __eq__(self, other):
        return self.sequence == other

    def __ne__(self, other):
        return self.sequence != other

    def __hash__(self):
        return hash(self.sequence)

    def __repr__(self):
        return repr(self.sequence)

    def __str__(self):
        return str(self.sequence)

    def __unicode__(self):
        return unicode(self.sequence)

    def __format__(self, formatstr):
        return format(self.sequence, formatstr)

    def __nonzero__(self):
        return self.size() != 0

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return FunctionalSequence(self.sequence[key])
        return _wrap(self.sequence[key])

    def __setitem__(self, key, value):
        self.sequence[key] = value

    def __delitem__(self, key):
        del self.sequence[key]

    def __iter__(self):
        return iter(self.sequence)

    def __reversed__(self):
        return FunctionalSequence(reversed(self.sequence))

    def __contains__(self, item):
        return self.sequence.__contains__(item)

    def __add__(self, other):
        if isinstance(other, FunctionalSequence):
            return FunctionalSequence(self.sequence + other.sequence)
        else:
            return FunctionalSequence(self.sequence + other)

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
        """
        return _wrap(self.sequence[0])

    def first(self):
        """
        Returns the first element of the sequence.

        >>> seq([1, 2, 3]).head()
        1

        Raises IndexError when the sequence is empty.

        >>> seq([]).head()
        Traceback (most recent call last):
         ...
        IndexError: list index out of range
        """
        return _wrap(self.sequence[0])

    def head_option(self):
        """
        Returns the first element of the sequence or None, if the sequence is empty.

        >>> seq([1, 2, 3]).head_option()
        1

        >>> seq([]).head_option()
        None
        """
        if not self.sequence:
            return None
        return self.head()

    def last(self):
        return _wrap(self.sequence[-1])

    def tail(self):
        return FunctionalSequence(self.sequence[1:])

    def drop(self, n):
        return FunctionalSequence(self.sequence[n:])

    def drop_while(self, f):
        for i, e in enumerate(self.sequence):
            if not f(e):
                break
        return self.drop(i)

    def take(self, n):
        return FunctionalSequence(self.sequence[:n])

    def take_while(self, f):
        for i, e in enumerate(self.sequence):
            if not f(e):
                break
        return self.take(i)

    def map(self, f):
        return FunctionalSequence(map(f, self.sequence))

    def filter(self, f):
        return FunctionalSequence(filter(f, self.sequence))

    def filter_not(self, f):
        g = lambda x: not f(x)
        return FunctionalSequence(filter(g, self.sequence))

    def reduce(self, f):
        return reduce(f, self.sequence)

    def count(self, f):
        """
        Counts the number of elements in the sequence which satisfy the predicate f
        :param f: predicate to count elements on
        :return: count satisfying predicate
        """
        n = 0
        for e in self.sequence:
            if f(e):
                n += 1
        return n

    def len(self):
        return len(self.sequence)

    def size(self):
        return len(self.sequence)

    def reverse(self):
        return reversed(self)

    def distinct(self):
        return FunctionalSequence(list(set(self.sequence)))

    def any(self):
        return any(self.sequence)

    def all(self):
        return all(self.sequence)

    def enumerate(self, start=0):
        return FunctionalSequence(enumerate(self.sequence, start=start))

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
        """
        return _wrap(max(self.sequence))

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
        """
        return _wrap(min(self.sequence))

    def max_by(self, f):
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
        """
        return _wrap(max(self.sequence, key=f))

    def min_by(self, f):
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
        """
        return _wrap(min(self.sequence, key=f))

    def find(self, f):
        for e in self.sequence:
            if f(e):
                return e
        else:
            return None

    def flatten(self):
        return self.flat_map(lambda x: x)

    def flat_map(self, f):
        l = []
        for e in self.sequence:
            l.extend(f(e))
        return FunctionalSequence(l)

    def group_by(self, f):
        result = {}
        for e in self.sequence:
            if result.get(f(e)):
                result.get(f(e)).append(e)
            else:
                result[f(e)] = [e]
        return FunctionalSequence(result.items())

    def group_by_key(self):
        result = {}
        for e in self.sequence:
            if result.get(e[0]):
                result.get(e[0]).append(e[1])
            else:
                result[e[0]] = [e[1]]
        return FunctionalSequence(result.items())

    def grouped(self, size):
        """
        Partitions the elements into groups of length size

        The last partition will be at least of size 1 and no more than length size
        :param size: size of the partitions
        :return: sequence partitioned into groups of length size

        >>>  seq([1, 2, 3, 4, 5, 6, 7, 8]).grouped(2)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> seq([1, 2, 3, 4, 5, 6, 7, 8]).grouped(3)
        [[1, 2, 3], [4, 5, 6], [7, 8]]
        """
        result = []
        for i in range(0, self.size(), size):
            result.append(FunctionalSequence(self.sequence[i:i+size]))
        return FunctionalSequence(result)

    def empty(self):
        return len(self.sequence) == 0

    def non_empty(self):
        return len(self.sequence) != 0

    def make_string(self, separator):
        return separator.join(str(e) for e in self.sequence)

    def partition(self, f):
        p1 = self.filter(f)
        p2 = self.filter_not(f)
        return FunctionalSequence((p1, p2))

    def product(self):
        return self.reduce(lambda x, y: x * y)

    def slice(self, start, until):
        return FunctionalSequence(self.sequence[start:until])

    def sum(self):
        return sum(self.sequence)

    def fold_left(self, zero_value, f):
        """
        Takes a zero_value and performs a reduction to a value using f. f should take two
        parameters, 1) the value to accumulate into the result and 2) the current result.
        Elements are folded left to right.

        :param zero_value: zero value to reduce into
        :return: value from folding values with f into zero_value
        """
        result = zero_value
        for e in self.sequence:
            result = f(e, result)
        return _wrap(result)

    def fold_right(self, zero_value, f):
        """
        Takes a zero_value and performs a reduction to a value using f. f should take two
        parameters, 1) the value to accumulate into the result and 2) the current result.
        Elements are folded right to left.

        :param zero_value: zero value to reduce into
        :return: value from folding values with f into zero_value
        """
        return self.reverse().fold_left(zero_value, f)

    def set(self):
        return self.to_set()

    def to_set(self):
        return set(self.sequence)

    def zip(self, sequence):
        return FunctionalSequence(zip(self.sequence, sequence))

    def zip_with_index(self):
        return FunctionalSequence(list(enumerate(self.sequence)))

    def sorted(self, comp=None, key=None, reverse=False):
        return FunctionalSequence(sorted(self.sequence, cmp=comp, key=key, reverse=reverse))

    def to_list(self):
        return self.sequence

    def list(self):
        return self.to_list()

    def for_each(self, f):
        for e in self.sequence:
            f(e)

    def exists(self, f):
        for e in self.sequence:
            if f(e):
                return True
        return False

    def for_all(self, f):
        for e in self.sequence:
            if not f(e):
                return False
        return True

    def to_dict(self):
        d = {}
        for e in self.sequence:
            d[e[0]] = e[1]
        return d

    def reduce_by_key(self, f):
        return self.group_by_key().map(lambda kv: (kv[0], reduce(f, kv[1])))


def seq(l):
    return FunctionalSequence(l)


if sys.version < '3':
    _integer_types = (int, long)
    _str_types = (str, unicode)
else:
    _integer_types = int
    _str_types = str


def _is_primitive(v):
    return isinstance(v, str) \
        or isinstance(v, bool) \
        or isinstance(v, _str_types) \
        or isinstance(v, _integer_types) \
        or isinstance(v, float) \
        or isinstance(v, complex) \
        or isinstance(v, bytes)


def _wrap(v):
    if _is_primitive(v):
        return v
    elif isinstance(v, collections.Iterable):
        return FunctionalSequence(v)
    else:
        return v
