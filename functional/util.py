# pylint: disable=no-name-in-module,unused-import,too-many-instance-attributes,too-many-arguments, too-few-public-methods

import collections
import six
import builtins

if six.PY2:
    from itertools import ifilterfalse as filterfalse

    def dict_item_iter(dictionary):
        return dictionary.viewitems()
else:
    from itertools import filterfalse

    def dict_item_iter(dictionary):
        return dictionary.items()


def is_primitive(val):
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

    :param val: value to check
    :return: True if value is a primitive, else False
    """
    return isinstance(val, str) \
        or isinstance(val, bool) \
        or isinstance(val, six.string_types + (six.text_type,)) \
        or isinstance(val, six.integer_types) \
        or isinstance(val, float) \
        or isinstance(val, complex) \
        or isinstance(val, bytes)


def is_iterable(val):
    if isinstance(val, list):
        return False
    return isinstance(val, collections.Iterable)


class LazyFile(object):
    def __init__(self, path, delimiter=None, mode='r', buffering=-1, encoding=None,
                 errors=None, newline=None):
        self.path = path
        self.delimiter = delimiter
        self.mode = mode
        self.buffering = buffering
        self.encoding = encoding
        self.errors = errors
        self.newline = newline
        self.file = None

    def __iter__(self):
        self.file = builtins.open(self.path, mode=self.mode, buffering=self.buffering,
                                  encoding=self.encoding, errors=self.errors, newline=self.newline)
        return self

    def next(self):
        line = self.file.readline()
        if line:
            return line
        else:
            self.file.close()
            raise StopIteration

    def __next__(self):
        return self.next()
