# pylint: disable=no-name-in-module,unused-import

import collections
import six

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
