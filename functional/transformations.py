import sys
from functools import reduce, partial
from itertools import dropwhile, takewhile, islice, ifilterfalse
import collections


if sys.version < '3':
    from itertools import imap as map
    from itertools import ifilter as filter
    from itertools import izip as zip
    range = xrange
    dict_item_iter = lambda d: d.viewitems()
else:
    dict_item_iter = lambda d: d.items()

Transformation = collections.namedtuple('Transformation', ['name', 'function'])

CACHE_T = Transformation('cache', None)


def list_t():
    return Transformation('list', list)


def map_t(func):
    return Transformation('map({0})'.format(func.__name__), partial(map, func))


def filter_t(func):
    return Transformation('filter({0})'.format(func.__name__), partial(filter, func))


def filter_not_t(func):
    return Transformation('filter_not({0})'.format(func.__name__), partial(ifilterfalse, func))


def reversed_t():
    return Transformation('reversed', reversed)
