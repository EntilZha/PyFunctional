import unittest

from functional.chain import seq, FunctionalSequence


class TestChain(unittest.TestCase):
    def assertType(self, s):
        self.assertTrue(isinstance(s, FunctionalSequence))

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
        result = seq(l).map(f)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_filter(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [5, 10]
        s = seq(l)
        result = s.filter(f)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_filter_not(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [0, -1]
        result = seq(l).filter_not(f)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_map_filter(self):
        f = lambda x: x > 0
        g = lambda x: x * 2
        l = [0, -1, 5]
        s = seq(l)
        expect = [10]
        result = s.filter(f).map(g)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_reduce(self):
        f = lambda x, y: x + y
        l = ["a", "b", "c"]
        expect = "abc"
        s = seq(l)
        self.assertEqual(expect, s.reduce(f))

    def test_sum(self):
        l = [1, 2, 3]
        self.assertEqual(6, seq(l).sum())

    def test_reverse(self):
        l = [1, 2, 3]
        expect = [3, 2, 1]
        s = seq(l)
        result = s.reverse()
        self.assertEqual(expect, result)
        self.assertType(result)
        result = s.__reversed__()
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_distinct(self):
        l = [1, 1, 2, 3, 2, 3]
        expect = [1, 2, 3]
        s = seq(l)
        result = s.distinct()
        for e in result:
            self.assertTrue(e in expect)
        result = s.distinct()
        self.assertEqual(result.size(), len(expect))
        self.assertType(result)

    def test_any(self):
        l = [True, False]
        self.assertTrue(seq(l).any())

    def test_all(self):
        l = [True, False]
        self.assertFalse(seq(l).all())
        l = [True, True]
        self.assertTrue(seq(l).all())

    def test_max(self):
        l = [1, 2, 3]
        self.assertEqual(3, seq(l).max())

    def test_min(self):
        l = [1, 2, 3]
        self.assertEqual(1, seq(l).min())

    def test_find(self):
        l = [1, 2, 3]
        f = lambda x: x == 3
        g = lambda x: False
        self.assertEqual(3, seq(l).find(f))
        self.assertIsNone(seq(l).find(g))

    def test_flat_map(self):
        l = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        f = lambda x: x
        expect = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        result = seq(l).flat_map(f)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_group_by(self):
        l = [(1, 1), (1, 2), (1, 3), (2, 2)]
        f = lambda x: x[0]
        expect = {1: [(1, 1), (1, 2), (1, 3)], 2: [(2, 2)]}
        result = seq(l).group_by(f)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_empty(self):
        self.assertTrue(seq([]).empty())

    def test_non_empty(self):
        self.assertTrue(seq([1]).non_empty())

    def test_string(self):
        l = [1, 2, 3]
        expect1 = "1 2 3"
        expect2 = "1:2:3"
        s = seq(l)
        self.assertEqual(expect1, s.string(" "))
        self.assertEqual(expect2, s.string(":"))

    def test_partition(self):
        l = [-1, -2, -3, 1, 2, 3]
        e2 = [-1, -2, -3]
        e1 = [1, 2, 3]
        f = lambda x: x > 0
        s = seq(l)
        p1, p2 = s.partition(f)
        self.assertEqual(e1, p1)
        self.assertEqual(e2, p2)
        self.assertType(p1)
        self.assertType(p2)

    def test_product(self):
        l = [2, 2, 3]
        self.assertEqual(12, seq(l).product())

    def test_set(self):
        l = [1, 1, 2, 2, 3]
        ls = set(l)
        self.assertEqual(ls, seq(l).set())

    def test_zip(self):
        l1 = [1, 2, 3]
        l2 = [-1, -2, -3]
        e = [(1, -1), (2, -2), (3, -3)]
        result = seq(l1).zip(l2)
        self.assertEqual(e, result)
        self.assertType(result)

    def test_zip_with_index(self):
        l = [2, 3, 4]
        e = [(0, 2), (1, 3), (2, 4)]
        result = seq(l).zip_with_index()
        self.assertEqual(result, e)
        self.assertType(result)

    def test_enumerate(self):
        l = [2, 3, 4]
        e = [(0, 2), (1, 3), (2, 4)]
        result = seq(l).enumerate()
        self.assertEqual(result, e)
        self.assertType(result)

    def test_python_slice(self):
        l = [1, 2, 3]
        self.assertType(seq(l)[0:1])
