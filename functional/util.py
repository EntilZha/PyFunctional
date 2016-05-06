from __future__ import absolute_import, division
from itertools import chain, count, islice, takewhile
from functools import reduce
from multiprocessing import Pool, cpu_count
import collections

try:
    import dill as serializer
except ImportError:
    import pickle as serializer
import future.builtins as builtins
import six


if six.PY2:
    CSV_WRITE_MODE = 'wb'
    PROTOCOL = 2
else:
    CSV_WRITE_MODE = 'w'
    PROTOCOL = serializer.HIGHEST_PROTOCOL
CPU_COUNT = cpu_count()


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


def is_namedtuple(val):
    """
    Use Duck Typing to check if val is a named tuple. Checks that val is of type tuple and contains
    the attribute _fields which is defined for named tuples.
    :param val: value to check type of
    :return: True if val is a namedtuple
    """
    val_type = type(val)
    bases = val_type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(val_type, '_fields', None)
    return all(isinstance(n, str) for n in fields)


def identity(arg):
    """
    Function which returns the argument. Used as a default lambda function.

    >>> obj = object()
    >>> obj is identity(obj)
    True

    :param arg: object to take identity of
    :return: return arg
    """
    return arg


def is_iterable(val):
    """
    Check if val is not a list, but is a collections.Iterable type. This is used to determine
    when list() should be called on val

    >>> l = [1, 2]
    >>> is_iterable(l)
    False
    >>> is_iterable(iter(l))
    True

    :param val: value to check
    :return: True if it is not a list, but is a collections.Iterable
    """
    if isinstance(val, list):
        return False
    return isinstance(val, collections.Iterable)


def split_every(parts, iterable):
    """
    Split an iterable into parts of length parts

    >>> l = iter([1, 2, 3, 4])
    >>> split_every(2, l)
    [[1, 2], [3, 4]]

    :param iterable: iterable to split
    :param parts: number of chunks
    :return: return the iterable split in parts
    """
    return takewhile(bool, (list(islice(iterable, parts)) for _ in count()))


def unpack(packed):
    """
    Unpack the function and args then apply the function to the arguments and return result
    :param packed: input packed tuple of (func, args)
    :return: result of applying packed function on packed args
    """
    func, args = serializer.loads(packed)
    result = func(*args)
    if isinstance(result, collections.Iterable):
        return list(result)


def pack(func, args):
    """
    Pack a function and the args it should be applied to
    :param func: Function to apply
    :param args: Args to evaluate with
    :return: Packed (func, args) tuple
    """
    return serializer.dumps((func, args), PROTOCOL)


def parallelize(func, result, processes=None):
    """
    Creates an iterable which is lazily computed in parallel from applying func on result
    :param func: Function to apply
    :param result: Data to apply to
    :param processes: Number of processes to use in parallel
    :return: Iterable of applying func on result
    """
    parallel_iter = lazy_parallelize(func, result, processes=processes)
    return chain.from_iterable(parallel_iter)


def lazy_parallelize(func, result, processes=None):
    """
    Lazily computes an iterable in parallel, and returns them in pool chunks
    :param func: Function to apply
    :param result: Data to apply to
    :param processes: Number of processes to use in parallel
    :return: Iterable of chunks where each chunk as func applied to it
    """
    if processes is None or processes < 1:
        processes = CPU_COUNT
    else:
        processes = min(processes, CPU_COUNT)
    try:
        chunk_size = (len(result) // processes) or processes
    except TypeError:
        chunk_size = processes
    pool = Pool(processes=processes)
    chunks = split_every(chunk_size, iter(result))
    packed_chunks = (pack(func, (chunk, )) for chunk in chunks)
    for pool_result in pool.imap(unpack, packed_chunks):
        yield pool_result
    pool.terminate()


def compose(*functions):
    """
    Compose all the function arguments together
    :param functions: Functions to compose
    :return: Single composed function
    """
    # pylint: disable=undefined-variable
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


class ReusableFile(object):
    """
    Class which emulates the builtin file except that calling iter() on it will return separate
    iterators on different file handlers (which are automatically closed when iteration stops). This
    is useful for allowing a file object to be iterated over multiple times while keep evaluation
    lazy.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, path, delimiter=None, mode='r', buffering=-1, encoding=None,
                 errors=None, newline=None):
        """
        Constructor arguments are passed directly to builtins.open
        :param path: passed to open
        :param delimiter: passed to open
        :param mode: passed to open
        :param buffering: passed to open
        :param encoding: passed to open
        :param errors: passed to open
        :param newline: passed to open
        :return: ReusableFile from the arguments
        """
        self.path = path
        self.delimiter = delimiter
        self.mode = mode
        self.buffering = buffering
        self.encoding = encoding
        self.errors = errors
        self.newline = newline

    def __iter__(self):
        """
        Returns a new iterator over the file using the arguments from the constructor. Each call
        to __iter__ returns a new iterator independent of all others
        :return: iterator over file
        """
        with builtins.open(self.path,
                           mode=self.mode,
                           buffering=self.buffering,
                           encoding=self.encoding,
                           errors=self.errors,
                           newline=self.newline) as file_content:
            for line in file_content:
                yield line
