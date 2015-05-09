import sys
import collections

if sys.version < '3':
    from itertools import ifilterfalse as filterfalse
    dict_item_iter = lambda d: d.viewitems()
    integer_types = (int, long)
    str_types = (str, unicode)
else:
    from itertools import filterfalse
    dict_item_iter = lambda d: d.items()
    integer_types = int
    str_types = str


def is_primitive(v):
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

    :param v: value to check
    :return: True if value is a primitive, else False
    """
    return isinstance(v, str) \
        or isinstance(v, bool) \
        or isinstance(v, str_types) \
        or isinstance(v, integer_types) \
        or isinstance(v, float) \
        or isinstance(v, complex) \
        or isinstance(v, bytes)


def is_iterable(v):
    if isinstance(v, list):
        return False
    return isinstance(v, collections.Iterable)