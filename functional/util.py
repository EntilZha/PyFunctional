# pylint: disable=no-name-in-module,unused-import

import sys
import collections

if sys.version < '3':
    from itertools import ifilterfalse as filterfalse

    def dict_item_iter(dictionary):
        return dictionary.viewitems()
    INTEGER_TYPES = (int, long)
    STR_TYPES = (str, unicode)
else:
    from itertools import filterfalse

    def dict_item_iter(dictionary):
        return dictionary.items()
    INTEGER_TYPES = (int,)
    STR_TYPES = (str,)


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
        or isinstance(val, STR_TYPES) \
        or isinstance(val, INTEGER_TYPES) \
        or isinstance(val, float) \
        or isinstance(val, complex) \
        or isinstance(val, bytes)


def is_iterable(val):
    if isinstance(val, list):
        return False
    return isinstance(val, collections.Iterable)
