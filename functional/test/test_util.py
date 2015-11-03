import unittest
from ..util import ReusableFile


class TestUtil(unittest.TestCase):
    def test_reusable_file(self):
        license_file_lf = ReusableFile('LICENSE.txt')
        with open('LICENSE.txt') as license_file:
            self.assertEqual(list(license_file), list(license_file_lf))
        iter_1 = iter(license_file_lf)
        iter_2 = iter(license_file_lf)
        self.assertEqual(list(iter_1), list(iter_2))
