import unittest
from ..util import LazyFile


class TestUtil(unittest.TestCase):
    def test_lazy_file(self):
        file = LazyFile('LICENSE.txt')
        self.assertTrue(file.file is None)
        iter(file)
        handle_0 = file.file
        iter(file)
        handle_1 = file.file
        self.assertTrue(handle_0.closed)
        self.assertFalse(handle_1.closed)
