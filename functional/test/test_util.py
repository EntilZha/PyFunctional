from __future__ import absolute_import

import unittest
from collections import namedtuple

from functional.util import ReusableFile, is_namedtuple

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
