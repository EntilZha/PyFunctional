import unittest
from ..util import LazyFile


class TestUtil(unittest.TestCase):
    def test_lazy_file(self):
        license_file = LazyFile('LICENSE.txt')
        self.assertTrue(license_file.file is None)
        iter(license_file)
        handle_0 = license_file.file
        iter(license_file)
        handle_1 = license_file.file
        self.assertTrue(handle_0.closed)
        self.assertFalse(handle_1.closed)
