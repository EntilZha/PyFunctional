# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest
import sys
from collections import namedtuple
from functools import reduce
from operator import add

from functional.util import (ReusableFile, is_namedtuple, lazy_parallelize, split_every, pack,
                             unpack, compute_partition_size, GZFile)


Data = namedtuple('Tuple', 'x y')


class TestUtil(unittest.TestCase):
    def test_reusable_file(self):
        license_file_lf = ReusableFile('LICENSE.txt')
        with open('LICENSE.txt') as license_file:
            self.assertEqual(list(license_file), list(license_file_lf))
        iter_1 = iter(license_file_lf)
        iter_2 = iter(license_file_lf)
        self.assertEqual(list(iter_1), list(iter_2))

    def test_is_namedtuple(self):
        self.assertTrue(is_namedtuple(Data(1, 2)))
        self.assertFalse(is_namedtuple((1, 2, 3)))
        self.assertFalse(is_namedtuple([1, 2, 3]))
        self.assertFalse(is_namedtuple(1))

    # Skipping tests on pypy because of https://github.com/uqfoundation/dill/issues/73
    @unittest.skipIf('__pypy__' in sys.builtin_module_names, 'Skip parallel tests on pypy')
    def test_lazy_parallelize(self):
        self.assertListEqual(list(range(10)), reduce(add, lazy_parallelize(lambda x: x, range(10))))
        self.assertListEqual(list(range(10)), list(
            reduce(add, lazy_parallelize(lambda x: x, range(10), processes=10000))))

        def f():
            yield 0
        self.assertListEqual([[0]], list(lazy_parallelize(lambda x: x, f())))

    def test_split_every(self):
        result = iter([1, 2, 3, 4])
        self.assertListEqual(list(split_every(2, result)), [[1, 2], [3, 4]])
        result = iter([1, 2, 3, 4, 5])
        self.assertListEqual(list(split_every(2, result)), [[1, 2], [3, 4], [5]])

    # Skipping tests on pypy because of https://github.com/uqfoundation/dill/issues/73
    @unittest.skipIf('__pypy__' in sys.builtin_module_names, 'Skip parallel tests on pypy')
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

    def test_gzip_file(self):
        file_name = 'functional/test/data/test.txt.gz'
        expect = [
            'line0\n',
            'line1\n',
            'line2',
        ]
        self.assertListEqual(expect, list(GZFile(file_name, mode='rt', encoding="utf-8")))

        expect = [
            b'line0\n',
            b'line1\n',
            b'line2',
        ]
        self.assertListEqual(expect, list(GZFile(file_name, mode='rb')))
