import collections
import sys
from functools import reduce


class FunctionalSequence(object):
    def __init__(self, sequence):
        """
        Takes a sequence and wraps it around a FunctionalSequence object.

        If the sequence
        is already an instance of FunctionalSequence, __init__ will insure that it is
        at most wrapped exactly once.

        If the sequence is a list or tuple, it is set as the sequence.

        If it is an iterable, then it is expanded into a list then set to the sequence

        If the object does not fit any of these classes, a TypeError is thrown

        :param sequence: sequence of items to wrap in a FunctionalSequence
        :return: sequence wrapped in a FunctionalSequence
        """
        if isinstance(sequence, FunctionalSequence):
            self.sequence = sequence._get_base_sequence()
        elif isinstance(sequence, list) or isinstance(sequence, tuple):
            self.sequence = sequence
        elif isinstance(sequence, collections.Iterable):
            self.sequence = list(sequence)
        else:
            raise TypeError("Given sequence must be a list")

    def _get_base_sequence(self):
        """
        Retrieves the root sequence wrapped by one or more FunctionalSequence objects

        :return: root sequence
        """
        if isinstance(self.sequence, FunctionalSequence):
            return self.sequence._get_base_sequence()
        else:
            return self.sequence

    def __eq__(self, other):
        """
        Checks for equality with the sequence's equality operator

        :param other: object to compare to
        :return: true if the underlying sequence is equal to other
        """
        return self.sequence == other

    def __ne__(self, other):
        """
        Checks for inequality with the sequence's inequality operator

        :param other: object to compare to
        :return: true if the underlying sequence is not equal to other
        """
        return self.sequence != other

    def __hash__(self):
        """
        Return the hash of the sequence

        :return: hash of sequence
        """
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
        return _wrap(self.sequence[key])

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

        :return: first element of sequence
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

        :return: first element of sequence
        """
        return _wrap(self.sequence[0])

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
        return FunctionalSequence(self.sequence[:-1])

    def tail(self):
        """
        Returns the sequence, without its first element.

        >>> seq([1, 2, 3]).init()
        [2, 3]

        :return: sequence without first element
        """
        return FunctionalSequence(self.sequence[1:])

    def inits(self):
        """
        Returns consecutive inits of the sequence.

        >>> seq([1, 2, 3]).inits()
        [[1, 2, 3], [1, 2], [1], []]

        :return: consecutive init()s on sequence
        """
        result = [_wrap(self.sequence[:i]) for i in reversed(range(len(self.sequence) + 1))]
        return FunctionalSequence(result)

    def tails(self):
        """
        Returns consecutive tails of the sequence.

        >>> seq([1, 2, 3]).tails()
        [[1, 2, 3], [2, 3], [3], []]

        :return: consecutive tail()s of the sequence
        """
        result = [_wrap(self.sequence[i:]) for i in range(len(self.sequence) + 1)]
        return FunctionalSequence(result)

    def drop(self, n):
        """
        Drop the first n elements of the sequence.

        >>> seq([1, 2, 3, 4, 5]).drop(2)
        [3, 4, 5]

        :param n: number of elements to drop
        :return: sequence without first n elements
        """
        return FunctionalSequence(self.sequence[n:])

    def drop_while(self, f):
        """
        Drops elements in the sequence while f evaluates to True, then returns the rest.

        >>> seq([1, 2, 3, 4, 5, 1, 2]).drop_while(lambda x: x < 3)
        [3, 4, 5, 1, 2]

        :param f: truth returning function
        :return: elements including and after f evaluates to False
        """
        i = 0
        for i, e in enumerate(self.sequence):
            if not f(e):
                break
        return self.drop(i)

    def take(self, n):
        """
        Take the first n elements of the sequence.

        >>> seq([1, 2, 3, 4]).take(2)
        [1, 2]

        :param n: number of elements to take
        :return: first n elements of sequence
        """
        return FunctionalSequence(self.sequence[:n])

    def take_while(self, f):
        """
        Take elements in the sequence until f evaluates to False, then return them.

        >>> seq([1, 2, 3, 4, 5, 1, 2]).take_while(lambda x: x < 3)
        [1, 2]

        :param f: truth returning function
        :return: elements taken until f evaluates to False
        """
        i = 0
        for i, e in enumerate(self.sequence):
            if not f(e):
                break
        return self.take(i)

    def map(self, f):
        """
        Maps f onto the elements of the sequence.

        >>> seq([1, 2, 3, 4]).map(lambda x: x * -1)
        [-1, -2, -3, -4]

        :param f: function to map with
        :return: sequence with f mapped onto it
        """
        return FunctionalSequence(map(f, self.sequence))

    def for_each(self, f):
        """
        Executes f on each element of the sequence.

        >>> l = []
        >>> seq([1, 2, 3, 4]).for_each(l.append)
        >>> l
        [1, 2, 3, 4]

        :param f: function to execute
        :return: None
        """
        for e in self.sequence:
            f(e)

    def filter(self, f):
        """
        Filters sequence to include only elements where f is True.

        >>> seq([-1, 1, -2, 2]).filter(lambda x: x > 0)
        [1, 2]

        :param f: function to filter on
        :return: filtered sequence
        """
        return FunctionalSequence(filter(f, self.sequence))

    def filter_not(self, f):
        """
        Filters sequence to include only elements where f is False.

        >>> seq([-1, 1, -2, 2]).filter_not(lambda x: x > 0)
        [-1, -2]

        :param f: function to filter_not on
        :return: filtered sequence
        """
        g = lambda x: not f(x)
        return FunctionalSequence(filter(g, self.sequence))

    def count(self, f):
        """
        Counts the number of elements in the sequence which satisfy the predicate f.

        >>> seq([-1, -2, 1, 2]).count(lambda x: x > 0)
        2

        :param f: predicate to count elements on
        :return: count of elements that satisf predicate
        """
        n = 0
        for e in self.sequence:
            if f(e):
                n += 1
        return n

    def len(self):
        """
        Return length of sequence using its length function.

        >>> seq([1, 2, 3]).len()
        3

        :return: length of sequence
        """
        return len(self.sequence)

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
        return len(self.sequence) == 0

    def non_empty(self):
        """
        Returns True if the sequence does not have length zero.

        >>> seq([]).non_empty()
        False

        >>> seq([1]).non_empty()
        True

        :return: True if sequence length is not zero
        """
        return len(self.sequence) != 0

    def any(self):
        """
        Returns True if any element in the sequence has truth value True

        >>> seq([True, False]).any()
        True

        >>> seq([False, False]).any()
        False

        :return: True if any element is True
        """
        return any(self.sequence)

    def all(self):
        """
        Returns True if the truth value of all items in the sequence true.

        >>> seq([True, True]).all()
        True

        >>> seq([True, False]).all()
        False

        :return: True if all items truth value evaluates to True
        """
        return all(self.sequence)

    def exists(self, f):
        """
        Returns True if an element in the sequence makes f evaluate to True.

        >>> seq([1, 2, 3, 4]).exists(lambda x: x == 2)
        True

        >>> seq([1, 2, 3, 4]).exists(lambda x: x < 0)
        False

        :param f: existence check function
        :return: True if any element satisfies f
        """
        for e in self.sequence:
            if f(e):
                return True
        return False

    def for_all(self, f):
        """
        Returns True if all elements in sequence make f evaluate to True.

        >>> seq([1, 2, 3]).for_all(lambda x: x > 0)
        True

        >>> seq([1, 2, -1]).for_all(lambda x: x > 0)
        False

        :param f: function to check truth value of all elements with
        :return: True if all elements make f evaluate to True
        """
        for e in self.sequence:
            if not f(e):
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
        """
        Finds the first element of the sequence that satisfies f. If no such element exists,
        then return None.

        >>> seq(["abc", "ab", "bc"]).find(lambda x: len(x) == 2)
        'ab'

        :param f: function to find with
        :return: first element to satisfy f or None
        """
        for e in self.sequence:
            if f(e):
                return e
        else:
            return None

    def flatten(self):
        """
        Flattens a sequence of sequences to a single sequence of elements.

        >>> seq([[1, 2], [3, 4], [5, 6]])
        [1, 2, 3, 4, 5, 6]

        :return: flattened sequence
        """
        return self.flat_map(lambda x: x)

    def flat_map(self, f):
        """
        Applies f to each element of the sequence, which themselves should be sequences.
        Then appends each element of each sequence to a final result

        >>> seq([[1, 2], [3, 4], [5, 6]]).flat_map(lambda x: x)
        [1, 2, 3, 4, 5, 6]

        >>> seq(["a", "bc", "def"]).flat_map(list)
        ['a', 'b', 'c', 'd', 'e', 'f']

        >>> seq([[1], [2], [3]]).flat_map(lambda x: x * 2)
        [1, 1, 2, 2, 3, 3]

        :param f: function to apply to each sequence in the sequence
        :return: application of f to elements followed by flattening
        """
        l = []
        for e in self.sequence:
            l.extend(f(e))
        return FunctionalSequence(l)

    def group_by(self, f):
        """
        Group elements into a list of (Key, Value) tuples where f creates the key and maps
        to values matching that key.

        >>> seq(["abc", "ab", "z", "f", "qw"]).group_by(len)
        [(1, ['z', 'f']), (2, ['ab', 'qw']), (3, ['abc'])]

        :param f: group by result of this function
        :return: grouped sequence
        """
        result = {}
        for e in self.sequence:
            if result.get(f(e)):
                result.get(f(e)).append(e)
            else:
                result[f(e)] = [e]
        return FunctionalSequence(result.items())

    def group_by_key(self):
        """
        Group sequence of (Key, Value) elements by Key.

        >>> seq([('a', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 3), ('c', 0)]).group_by_key()
        [('a', [1]), ('c', [3, 0]), ('b', [2, 3, 4])]

        :return: sequence grouped by key
        """
        result = {}
        for e in self.sequence:
            if result.get(e[0]):
                result.get(e[0]).append(e[1])
            else:
                result[e[0]] = [e[1]]
        return FunctionalSequence(result.items())

    def reduce_by_key(self, f):
        """
        Reduces a sequence of (Key, Value) using f on each Key.

        >>> seq([('a', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 3), ('c', 0)]).reduce_by_key(lambda x, y: x + y)
        [('a', 1), ('c', 3), ('b', 9)]

        :param f: reduce each list of values using two parameter, associative f
        :return: Sequence of tuples where the value is reduced with f
        """
        return self.group_by_key().map(lambda kv: (kv[0], reduce(f, kv[1])))

    def reduce(self, f):
        """
        Reduce sequence of elements using f.

        >>> seq([1, 2, 3]).reduce(lambda x, y: x + y)
        6

        :param f: two parameter, associative reduce function
        :return: reduced value using f
        """
        return reduce(f, self.sequence)

    def make_string(self, separator):
        """
        Concatenate the elements of the sequence into a string separated by separator.

        >>> seq([1, 2, 3]).make_string("@")
        '1@2@3'

        :param separator: string separating elements in string
        :return: concatenated string separated by separator
        """
        return separator.join(str(e) for e in self.sequence)

    def product(self):
        """
        Takes product of elements in sequence.

        >>> seq([1, 2, 3, 4]).product()
        24

        :return: product of elements in sequence
        """
        return self.reduce(lambda x, y: x * y)

    def sum(self):
        """
        Takes sum of elements in sequence.

        >>> seq([1, 2, 3, 4]).sum()
        10

        :return: sum of elements in sequence
        """
        return sum(self.sequence)

    def fold_left(self, zero_value, f):
        """
        Takes a zero_value and performs a reduction to a value using f. f should take two
        parameters, 1) the value to accumulate into the result and 2) the current result.
        Elements are folded left to right.

        >>> seq(['a', 'bc', 'de', 'f', 'm', 'nop']).fold_left("Start:", lambda v, curr: curr + 2 * v)
        'Start:aabcbcdedeffmmnopnop'

        :param zero_value: zero value to reduce into
        :param f: Two parameter function. First parameter is value to be accumulated into the result. Second parameter is the current result
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

        >>> seq(['a', 'bc', 'de', 'f', 'm', 'nop']).fold_right("Start:", lambda v, curr: curr + 2 * v)
        'Start:nopnopmmffdedebcbcaa'

        :param zero_value: zero value to reduce into
        :param f: Two parameter function. First parameter is value to be accumulated into the result. Second parameter is the current result
        :return: value from folding values with f into zero_value
        """
        return self.reverse().fold_left(zero_value, f)

    def zip(self, sequence):
        return FunctionalSequence(zip(self.sequence, sequence))

    def zip_with_index(self):
        return FunctionalSequence(list(enumerate(self.sequence)))

    def enumerate(self, start=0):
        return FunctionalSequence(enumerate(self.sequence, start=start))

    def partition(self, f):
        p1 = self.filter(f)
        p2 = self.filter_not(f)
        return FunctionalSequence((p1, p2))

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

    def sorted(self, comp=None, key=None, reverse=False):
        return FunctionalSequence(sorted(self.sequence, cmp=comp, key=key, reverse=reverse))

    def reverse(self):
        return reversed(self)

    def distinct(self):
        return FunctionalSequence(list(set(self.sequence)))

    def slice(self, start, until):
        return FunctionalSequence(self.sequence[start:until])

    def to_list(self):
        return list(self.sequence)

    def list(self):
        return self.to_list()

    def to_set(self):
        return set(self.sequence)

    def set(self):
        return self.to_set()

    def to_dict(self):
        d = {}
        for e in self.sequence:
            d[e[0]] = e[1]
        return d

    def dict(self):
        return self.to_dict()


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
    if isinstance(v, dict) or isinstance(v, set):
        return v
    elif isinstance(v, collections.Iterable):
        return FunctionalSequence(v)
    else:
        return v
