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
        Retrieves the root sequence wrapped by one or more FunctionalSequence objects.

        :return: root sequence
        """
        if isinstance(self.sequence, FunctionalSequence):
            return self.sequence._get_base_sequence()
        else:
            return self.sequence

    def __getattr__(self, item):
        """
        Extends attribute access to any attributes that the underlying sequence may have. Since
        __getattr__ is used instead of __getattribute__, attributes in FunctionalSequence take precedent

        :param item: attribute to get
        :return: either result of getting the item attribute from the sequence or an error
        """
        return getattr(self.sequence, item)

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
        return hash(self.sequence)

    def __repr__(self):
        """
        Return repr using sequence's repr function.

        :return: sequence's repr
        """
        return repr(self.sequence)

    def __str__(self):
        """
        Return string using sequence's string function.

        :return: sequence's string
        """
        return str(self.sequence)

    def __unicode__(self):
        """
        Return unicode using sequence's unicode function.

        :return: sequence's unicode
        """
        return unicode(self.sequence)

    def __format__(self, formatstr):
        """
        Format the sequence using formatstr and format()

        :param formatstr: format passed to format()
        :return: formatted string
        """
        return format(self.sequence, formatstr)

    def __nonzero__(self):
        """
        Returns True if size is not zero.

        :return: True if size is not zero
        """
        return self.size() != 0

    def __len__(self):
        """
        Return length of sequence.

        :return: length of sequence
        """
        return len(self.sequence)

    def __getitem__(self, key):
        """
        Kets item at index key.

        :param key: key to use for getitem
        :return: item at index key
        """
        return _wrap(self.sequence[key])

    def __iter__(self):
        """
        Return iterator of sequence.

        :return: iterator of sequence
        """
        return iter(self.sequence)

    def __reversed__(self):
        """
        Return reversed sequence using sequence's reverse function

        :return: reversed sequence
        """
        return FunctionalSequence(reversed(self.sequence))

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

    def drop_right(self, n):
        """
        Drops the last n elements of the sequence.

        >>> seq([1, 2, 3, 4]).drop_right(2)
        [1, 2]

        :param n: number of elements to drop
        :return: sequence with last n elements dropped
        """
        return FunctionalSequence(self.sequence[:n])

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

    def union(self, other):
        """
        New sequence with unique elements from self.sequence and other.

        >>> seq([1, 1, 2, 3, 3]).union([1, 4, 5])
        [1, 2, 3, 4, 5]

        :param other: sequence to union with
        :return: union of sequence and other
        """
        result = set(self.sequence).union(set(other))
        return FunctionalSequence(list(result))

    def intersection(self, other):
        """
        New sequence with unique elements present in sequence and other.

        >>> seq([1, 1, 2, 3]).intersection([2, 3, 4])
        [2, 3]

        :param other: sequence to perform intersection with
        :return: intersection of sequence and other
        """
        result = set(self.sequence).intersection(set(other))
        return FunctionalSequence(list(result))

    def difference(self, other):
        """
        New sequence with unique elements present in sequence but not in other.

        >>> seq([1, 2, 3]).difference([2, 3, 4])
        [1]

        :param other: sequence to perform difference with
        :return: difference of sequence and other
        """
        result = set(self.sequence).difference(set(other))
        return FunctionalSequence(list(result))

    def symmetric_difference(self, other):
        """
        New sequence with elements in either sequence or other, but not both.

        >>> seq([1, 2, 3, 3]).symmetric_difference([2, 4, 5])
        [1, 3, 4, 5]

        :param other: sequence to perform symmetric difference with
        :return: symmetric difference of sequence and other
        """

        result = set(self.sequence).symmetric_difference(set(other))
        return FunctionalSequence(list(result))

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
        """
        Zips the stored sequence with the given sequence.

        >>> seq([1, 2, 3]).zip([4, 5, 6])
        [(1, 4), (2, 5), (3, 6)]

        :param sequence: second sequence to zip
        :return: stored sequence zipped with given sequence
        """
        return FunctionalSequence(zip(self.sequence, sequence))

    def zip_with_index(self):
        """
        Zips the sequence to its index, with the index being the first element of each tuple.

        >>> seq(['a', 'b', 'c']).zip_with_index()
        [(0, 'a'), (1, 'b'), (2, 'c')]

        :return: sequence zipped to its index
        """
        return FunctionalSequence(enumerate(self.sequence))

    def enumerate(self, start=0):
        """
        Uses python enumerate to to zip the sequence with indexes starting at start.

        >>> seq(['a', 'b', 'c']).enumerate(start=1)
        [(1, 'a'), (2, 'b'), (3, 'c')]

        :param start: Beginning of zip
        :return: enumerated sequence starting at start
        """
        return FunctionalSequence(enumerate(self.sequence, start=start))

    def join(self, other):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V) pairs and
        other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs. Will return only elements
        where the key exists in both sequences.

        >>> seq([('a', 1), ('b', 2), ('c', 3)]).join([('a', 2), ('c', 5)])
        [('a', (1, 2)), ('c', (3, 5))]

        :param other: sequence to join with
        :return: joined sequence of (K, (V, W)) pairs
        """
        seq_kv = self.to_dict()
        other_kv = dict(other)
        keys = seq_kv.keys() if len(seq_kv) < len(other_kv) else other_kv.keys()
        result = {}
        for k in keys:
            if k in seq_kv and k in other_kv:
                result[k] = (seq_kv[k], other_kv[k])
        return FunctionalSequence(result.items())

    def _general_join(self, other, join_type):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V) pairs and
        other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs. If join_type is "left",
        V values will always be present, W values may be present or None. If join_type is "right", W values will
        always be present, W values may be present or None. If join_type is "outer", V or W may be present or None,
        but never at the same time.

        >>> seq([('a', 1), ('b', 2)])._general_join([('a', 3), ('c', 4)], "left")
        [('a', (1, 3)), ('b', (2, None)]

        >>> seq([('a', 1), ('b', 2)])._general_join([('a', 3), ('c', 4)], "right")
        [('a', (1, 3)), ('c', (None, 4)]

        >>> seq([('a', 1), ('b', 2)])._general_join([('a', 3), ('c', 4)], "outer")
        [('a', (1, 3)), ('b', (2, None)), ('c', (None, 4))]

        :param other: sequence to join with
        :param join_type: specifies join_type, may be "left", "right", or "outer"
        :return: side joined sequence of (K, (V, W)) pairs
        """
        seq_kv = self.to_dict()
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
        return FunctionalSequence(result.items())

    def left_join(self, other):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V) pairs and
        other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs. V values will always be
        present, W values may be present or None.

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)])
        [('a', (1, 3)), ('b', (2, None)]

        :param other: sequence to join with
        :return: left joined sequence of (K, (V, W)) pairs
        """
        return self._general_join(other, "left")

    def right_join(self, other):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V) pairs and
        other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs. W values will always be
        present, V values may be present or None.

        >>> seq([('a', 1), ('b', 2)]).join([('a', 3), ('c', 4)])
        [('a', (1, 3)), ('b', (2, None)]

        :param other: sequence to join with
        :return: right joined sequence of (K, (V, W)) pairs
        """
        return self._general_join(other, "right")

    def outer_join(self, other):
        """
        Sequence and other must be composed of (Key, Value) pairs. If self.sequence contains (K, V) pairs and
        other contains (K, W) pairs, the return result is a sequence of (K, (V, W)) pairs. One of V or W will always
        be not None, but the other may be None

        >>> seq([('a', 1), ('b', 2)]).outer_join([('a', 3), ('c', 4)], "outer")
        [('a', (1, 3)), ('b', (2, None)), ('c', (None, 4))]

        :param other: sequence to join with
        :return: outer joined sequence of (K, (V, W)) pairs
        """
        return self._general_join(other, "outer")

    def partition(self, f):
        """
        Partition the sequence based on satisfying the predicate f.

        >>> seq([-1, 1, -2, 2]).partition(lambda x: x < 0)
        ([-1, -2], [1, 2])

        :param f: predicate to partition on
        :return: tuple of partitioned sequences
        """
        p1 = self.filter(f)
        p2 = self.filter_not(f)
        return FunctionalSequence((p1, p2))

    def grouped(self, size):
        """
        Partitions the elements into groups of length size.

        >>> seq([1, 2, 3, 4, 5, 6, 7, 8]).grouped(2)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> seq([1, 2, 3, 4, 5, 6, 7, 8]).grouped(3)
        [[1, 2, 3], [4, 5, 6], [7, 8]]

        The last partition will be at least of size 1 and no more than length size
        :param size: size of the partitions
        :return: sequence partitioned into groups of length size
        """
        result = []
        for i in range(0, self.size(), size):
            result.append(FunctionalSequence(self.sequence[i:i+size]))
        return FunctionalSequence(result)

    def sorted(self, key=None, reverse=False):
        """
        Uses python sort and its passed arguments to sort the input.

        >>> seq([2, 1, 4, 3]).sorted()
        [1, 2, 3, 4]

        :param comp: comparator
        :param key:
        :param reverse: return list reversed or not
        :return: sorted sequence
        """
        return FunctionalSequence(sorted(self.sequence, key=key, reverse=reverse))

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
        return FunctionalSequence(list(set(self.sequence)))

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
        return FunctionalSequence(self.sequence[start:until])

    def to_list(self):
        """
        Converts sequence to list of elements.

        >>> type(seq([]).to_list())
        list

        >>> type(seq([]))
        functional.chain.FunctionalSequence

        >>> seq([1, 2, 3]).to_list()
        [1, 2, 3]

        :return: list of elements in sequence
        """
        return list(self.sequence)

    def list(self):
        """
        Converts sequence to list of elements.

        >>> type(seq([]).list())
        list

        >>> type(seq([]))
        functional.chain.FunctionalSequence

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
        functional.chain.FunctionalSequence

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
        functional.chain.FunctionalSequence

        >>> seq([1, 1, 2, 2]).set()
        {1, 2}

        :return:set of elements in sequence
        """
        return self.to_set()

    def to_dict(self):
        """
        Converts sequence of (Key, Value) pairs to a dictionary.

        >>> type(seq([('a', 1)]).to_dict())
        dict

        >>> type(seq([]))
        functional.chain.FunctionalSequence

        >>> seq([('a', 1), ('b', 2)]).to_dict()
        {'a': 1, 'b': 2}

        :return: dictionary from sequence of (Key, Value) elements
        """
        d = {}
        for e in self.sequence:
            d[e[0]] = e[1]
        return d

    def dict(self):
        """
        Converts sequence of (Key, Value) pairs to a dictionary.

        >>> type(seq([('a', 1)]).to_dict())
        dict

        >>> type(seq([]))
        functional.chain.FunctionalSequence

        >>> seq([('a', 1), ('b', 2)]).to_dict()
        {'a': 1, 'b': 2}

        :return: dictionary from sequence of (Key, Value) elements
        """
        return self.to_dict()


def seq(l):
    """
    Alias function for creating a new FunctionalSequence. Calling seq() and FunctionalSequence() is functionally
    equivalent

    >>> type(seq([1, 2]))
    functional.chain.FunctionalSequence

    >>> type(FunctionalSequence([1, 2]))
    functional.chain.FunctionalSequence

    :param l: sequence to wrap in FunctionalSequence
    :return: wrapped sequence
    """
    return FunctionalSequence(l)


if sys.version < '3':
    _integer_types = (int, long)
    _str_types = (str, unicode)
else:
    _integer_types = int
    _str_types = str


def _is_primitive(v):
    """
    Checks if the passed value is a primitive type.

    >>> _is_primitive(1)
    True

    >>> _is_primitive("abc")
    True

    >>> _is_primitive(True)
    True

    >>> _is_primitive({})
    False

    >>> _is_primitive([])
    False

    >>> _is_primitive(set([]))

    :param v: value to check
    :return: True if value is a primitive, else False
    """
    return isinstance(v, str) \
        or isinstance(v, bool) \
        or isinstance(v, _str_types) \
        or isinstance(v, _integer_types) \
        or isinstance(v, float) \
        or isinstance(v, complex) \
        or isinstance(v, bytes)


def _wrap(v):
    """
    Wraps the passed value in a FunctionalSequence if it is not a primitive. If it is a string
    argument it is expanded to a list of characters.

    >>> _wrap(1)
    1

    >>> _wrap("abc")
    ['a', 'b', 'c']

    >>> type(_wrap([1, 2]))
    functional.chain.FunctionalSequence

    :param v: value to wrap
    :return: wrapped or not wrapped value
    """
    if _is_primitive(v):
        return v
    if isinstance(v, dict) or isinstance(v, set):
        return v
    elif isinstance(v, collections.Iterable):
        return FunctionalSequence(v)
    else:
        return v
