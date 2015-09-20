# pylint: disable=redefined-builtin,too-many-arguments

import builtins
import re
from .pipeline import Sequence
from .util import is_primitive, LazyFile


def seq(*args):
    """
    Primary entrypoint for the functional package. Returns a functional.pipeline.Sequence wrapping
    the original sequence.

    Additionally it parses various types of input to a Sequence as best it can.

    >>> type(seq([1, 2]))
    functional.pipeline.Sequence

    >>> type(Sequence([1, 2]))
    functional.pipeline.Sequence

    >>> seq([1, 2, 3])
    [1, 2, 3]

    >>> seq(1, 2, 3)
    [1, 2, 3]

    >>> seq(1)
    [1]

    >>> seq(range(4))
    [0, 1, 2, 3]

    :param args: Three types of arguments are valid.
        1) Iterable which is then directly wrapped as a Sequence
        2) A list of arguments is converted to a Sequence
        3) A single non-iterable is converted to a single element Sequence
    :return: wrapped sequence

    """
    if len(args) == 0:
        raise TypeError("seq() takes at least 1 argument ({0} given)".format(len(args)))
    elif len(args) > 1:
        return Sequence(list(args))
    elif is_primitive(args[0]):
        return Sequence([args[0]])
    else:
        return Sequence(args[0])


def open(path, delimiter=None, mode='r', buffering=-1, encoding=None,
         errors=None, newline=None):
    """
    Additional entry point to Sequence which parses input files as defined by options. Path
    specifies what file to parse. If delimiter is not None, then the file is read in bulk then
    split on it. If it is None (the default), then the file is parsed as sequence of lines. The
    rest of the options are passed directly to builtins.open with the exception that write/append
    file modes is not allowed.

    :param path: path to file
    :param delimiter: delimiter to split joined text on. if None, defaults to file.readlines()
    :param mode: file open mode
    :param buffering: passed to builtins.open
    :param encoding: passed to builtins.open
    :param errors: passed to builtins.open
    :param newline: passed to builtins.open
    :return: output of file depending on options wrapped in a Sequence via seq
    """
    if not re.match('^[rbt]{1,3}$', mode):
        raise ValueError('mode argument must be only have r, b, and t')
    if delimiter is None:
        return seq(LazyFile(path, mode=mode, buffering=buffering, encoding=encoding, errors=errors,
                            newline=newline))
    else:
        with builtins.open(path, mode=mode, buffering=buffering, encoding=encoding, errors=errors,
                           newline=newline) as data:
            return seq(''.join(data.readlines()).split(delimiter))

def range(*args):
    """
    Additional entry point to Sequence which wraps the builtin range generator.
    seq.range(args) is equivalent to seq(range(args)).
    """
    rng = builtins.range(*args)
    return seq(rng)

seq.open = open
seq.range = range
