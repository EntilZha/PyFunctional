import unittest

from functional.chain import seq


class TestChain(unittest.TestCase):
    def test_eq(self):
        l = [1, 2, 3]
        self.assertEqual(seq(l), seq(l))

    def test_ne(self):
        a = [1, 2, 3]
        b = [1]
        self.assertNotEqual(seq(a), seq(b))

    def test_repr(self):
        l = [1, 2, 3]
        self.assertEqual(repr(l), repr(seq(l)))

    def test_str(self):
        l = [1, 2, 3]
        self.assertEqual(str(l), str(seq(l)))

    def test_len(self):
        l = [1, 2, 3]
        s = seq(l)
        self.assertEqual(len(l), len(s))
        self.assertEqual(len(l), s.count())
        self.assertEqual(len(l), s.size())
        self.assertEqual(len(l), s.len())

    def test_map(self):
        f = lambda x: x * 2
        l = [1, 2, 0, 5]
        expect = [2, 4, 0, 10]
        self.assertEqual(expect, seq(l).map(f))

    def test_filter(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [5, 10]
        s = seq(l)
        self.assertEqual(expect, s.filter(f))

    def test_filter_not(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [0, -1]
        self.assertEqual(expect, seq(l).filter_not(f))

    def test_map_filter(self):
        f = lambda x: x > 0
        g = lambda x: x * 2
        l = [0, -1, 5]
        s = seq(l)
        expect = [10]
        result = s.filter(f).map(g)
        self.assertEqual(expect, result)

    def test_reduce(self):
        f = lambda x, y: x + y
        l = ["a", "b", "c"]
        expect = "abc"
        s = seq(l)
        self.assertEqual(expect, s.reduce(f))

    def test_sum(self):
        l = [1, 2, 3]
        self.assertEqual(6, seq(l).sum())
