from functional.pipeline import Sequence
from functional.util import is_primitive


def seq(*args):
    """
    Alias function for creating a new Sequence. Additionally it also parses various types
    of input to a Sequence as best it can.

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


