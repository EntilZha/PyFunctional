import unittest
import sys
from collections import namedtuple
from functools import reduce
from operator import add

from functional.util import (
    is_namedtuple,
    lazy_parallelize,
    split_every,
    pack,
    unpack,
    compute_partition_size,
    default_value,
)

Data = namedtuple("Data", "x y")


class TestUtil(unittest.TestCase):
    def test_is_namedtuple(self):
        self.assertTrue(is_namedtuple(Data(1, 2)))
        self.assertFalse(is_namedtuple((1, 2, 3)))
        self.assertFalse(is_namedtuple([1, 2, 3]))
        self.assertFalse(is_namedtuple(1))

    # Skipping tests on pypy because of https://github.com/uqfoundation/dill/issues/73
    @unittest.skipIf(
        "__pypy__" in sys.builtin_module_names, "Skip parallel tests on pypy"
    )
    def test_lazy_parallelize(self):
        self.assertListEqual(
            list(range(10)), reduce(add, lazy_parallelize(lambda x: x, range(10)))
        )
        self.assertListEqual(
            list(range(10)),
            list(
                reduce(add, lazy_parallelize(lambda x: x, range(10), processes=10000))
            ),
        )

        def f():
            yield 0

        self.assertListEqual([[0]], list(lazy_parallelize(lambda x: x, f())))

    def test_split_every(self):
        result = iter([1, 2, 3, 4])
        self.assertListEqual(list(split_every(2, result)), [[1, 2], [3, 4]])
        result = iter([1, 2, 3, 4, 5])
        self.assertListEqual(list(split_every(2, result)), [[1, 2], [3, 4], [5]])

    # Skipping tests on pypy because of https://github.com/uqfoundation/dill/issues/73
    @unittest.skipIf(
        "__pypy__" in sys.builtin_module_names, "Skip parallel tests on pypy"
    )
    def test_pack_unpack(self):
        packed = pack(map, [lambda x: x * 2, range(4)])
        self.assertListEqual(unpack(packed), [0, 2, 4, 6])

    def test_compute_partition_size(self):
        result = compute_partition_size([0, 1, 2], 2)
        self.assertEqual(result, 2)
        result = compute_partition_size([0, 1, 2, 3], 2)
        self.assertEqual(result, 2)
        result = compute_partition_size(iter([0, 1, 2, 3]), 2)
        self.assertEqual(result, 1)

    def test_default_value(self):
        result = default_value(True)
        self.assertEqual(result, True)
        result = default_value(False)
        self.assertEqual(result, False)
        result = default_value(None, True)
        self.assertEqual(result, True)
        result = default_value(None, False)
        self.assertEqual(result, False)
        with self.assertRaises(ValueError):
            result = default_value(None, None)
