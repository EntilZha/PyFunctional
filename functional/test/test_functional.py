# pylint: skip-file
import unittest
import array
from collections import namedtuple, deque
from itertools import product

from functional.pipeline import Sequence, is_iterable, _wrap, extend
from functional.transformations import name
from functional import seq, pseq

Data = namedtuple("Data", "x y")


def pandas_is_installed():
    try:
        global pandas
        import pandas

        return True
    except ImportError:
        return False


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.seq = seq

    def assert_type(self, s):
        self.assertTrue(isinstance(s, Sequence))

    def assert_not_type(self, s):
        self.assertFalse(isinstance(s, Sequence))

    def assertIteratorEqual(self, iter_0, iter_1):
        seq_0 = list(iter_0)
        seq_1 = list(iter_1)
        self.assertListEqual(seq_0, seq_1)

    def test_is_iterable(self):
        self.assertFalse(is_iterable([]))
        self.assertTrue(is_iterable(iter([1, 2])))

    def test_constructor(self):
        self.assertRaises(TypeError, lambda: Sequence(1))

    def test_base_sequence(self):
        l = []
        self.assert_type(self.seq(l))
        self.assert_not_type(self.seq(l).sequence)
        self.assert_type(self.seq(self.seq(l)))
        self.assert_not_type(self.seq(self.seq(l)).sequence)
        self.assert_not_type(self.seq(l)._base_sequence)

    def test_eq(self):
        l = [1, 2, 3]
        self.assertIteratorEqual(self.seq(l).map(lambda x: x), self.seq(l))

    def test_ne(self):
        a = [1, 2, 3]
        b = [1]
        self.assertNotEqual(self.seq(a), self.seq(b))

    def test_repr(self):
        l = [1, 2, 3]
        self.assertEqual(repr(l), repr(self.seq(l)))

    def test_lineage_name(self):
        f = lambda x: x  # noqa: E731
        self.assertEqual(f.__name__, name(f))
        f = "test"
        self.assertEqual("test", name(f))

    def test_str(self):
        l = [1, 2, 3]
        self.assertEqual(str(l), str(self.seq(l)))

    def test_hash(self):
        self.assertRaises(TypeError, lambda: hash(self.seq([1])))

    def test_len(self):
        l = [1, 2, 3]
        s = self.seq(l)
        self.assertEqual(len(l), s.size())
        self.assertEqual(len(l), s.len())

    def test_count(self):
        l = self.seq([-1, -1, 1, 1, 1])
        self.assertEqual(l.count(lambda x: x > 0), 3)
        self.assertEqual(l.count(lambda x: x < 0), 2)

    def test_getitem(self):
        l = [1, 2, [3, 4, 5]]
        s = self.seq(l).map(lambda x: x)
        self.assertEqual(s[1], 2)
        self.assertEqual(s[2], [3, 4, 5])
        self.assert_type(s[2])
        self.assertEqual(s[1:], [2, [3, 4, 5]])
        self.assert_type(s[1:])
        l = [{1, 2}, {2, 3}, {4, 5}]
        s = self.seq(l)
        self.assertIsInstance(s[0], set)
        self.assertEqual(s[0], l[0])

    def test_iter(self):
        l = list(enumerate(self.seq([1, 2, 3])))
        e = list(enumerate([1, 2, 3]))
        self.assertEqual(l, e)
        l = self.seq([1, 2, 3])
        e = [1, 2, 3]
        result = []
        for n in l:
            result.append(n)
        self.assertEqual(result, e)
        self.assert_type(l)

    def test_contains(self):
        string = "abcdef"
        s = self.seq(iter(string)).map(lambda x: x)
        self.assertTrue("c" in s)

    def test_add(self):
        l0 = self.seq([1, 2, 3]).map(lambda x: x)
        l1 = self.seq([4, 5, 6])
        l2 = [4, 5, 6]
        expect = [1, 2, 3, 4, 5, 6]
        self.assertEqual(l0 + l1, expect)
        self.assertEqual(l0 + l2, expect)

    def test_head(self):
        l = self.seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.head(), 1)
        self.assertEqual(l.head(no_wrap=True), 1)
        l = self.seq([[1, 2], 3, 4])
        self.assertEqual(l.head(), [1, 2])
        self.assertEqual(l.head(no_wrap=True), [1, 2])
        self.assert_type(l.head())
        self.assert_not_type(l.head(no_wrap=True))
        l = self.seq([[1, 2], 3, 4], no_wrap=True)
        self.assert_not_type(l.head())
        l = self.seq([])
        with self.assertRaises(IndexError):
            l.head()
        with self.assertRaises(IndexError):
            l.head(no_wrap=True)
        l = self.seq([deque(), deque()]).head()
        self.assert_type(l)
        l = self.seq([deque(), deque()]).head(no_wrap=True)
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).head()
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).head(no_wrap=False)
        self.assert_type(l)

    def test_first(self):
        l = self.seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.first(), 1)
        self.assertEqual(l.first(no_wrap=True), 1)
        l = self.seq([[1, 2], 3, 4]).map(lambda x: x)
        self.assertEqual(l.first(), [1, 2])
        self.assertEqual(l.first(no_wrap=True), [1, 2])
        self.assert_type(l.first())
        self.assert_not_type(l.first(no_wrap=True))
        l = self.seq([[1, 2], 3, 4], no_wrap=True).map(lambda x: x)
        self.assert_not_type(l.first())
        l = self.seq([])
        with self.assertRaises(IndexError):
            l.first()
        with self.assertRaises(IndexError):
            l.first(no_wrap=True)
        l = self.seq([deque(), deque()]).first()
        self.assert_type(l)
        l = self.seq([deque(), deque()]).first(no_wrap=True)
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).first()
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).first(no_wrap=False)
        self.assert_type(l)

    def test_head_option(self):
        l = self.seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.head_option(), 1)
        self.assertEqual(l.head_option(no_wrap=True), 1)
        l = self.seq([[1, 2], 3, 4]).map(lambda x: x)
        self.assertEqual(l.head_option(), [1, 2])
        self.assertEqual(l.head_option(no_wrap=True), [1, 2])
        self.assert_type(l.head_option())
        self.assert_not_type(l.head_option(no_wrap=True))
        l = self.seq([[1, 2], 3, 4], no_wrap=True).map(lambda x: x)
        self.assert_not_type(l.head_option())
        l = self.seq([])
        self.assertIsNone(l.head_option())
        self.assertIsNone(l.head_option(no_wrap=True))
        l = self.seq([deque(), deque()]).head_option()
        self.assert_type(l)
        l = self.seq([deque(), deque()]).head_option(no_wrap=True)
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).head_option()
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).head_option(no_wrap=False)
        self.assert_type(l)

    def test_last(self):
        l = self.seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.last(), 3)
        self.assertEqual(l.last(no_wrap=True), 3)
        l = self.seq([1, 2, [3, 4]]).map(lambda x: x)
        self.assertEqual(l.last(), [3, 4])
        self.assertEqual(l.last(no_wrap=True), [3, 4])
        self.assert_type(l.last())
        self.assert_not_type(l.last(no_wrap=True))
        l = self.seq([1, 2, [3, 4]], no_wrap=True).map(lambda x: x)
        self.assert_not_type(l.last())
        l = self.seq([deque(), deque()]).last()
        self.assert_type(l)
        l = self.seq([deque(), deque()]).last(no_wrap=True)
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).last()
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).last(no_wrap=False)
        self.assert_type(l)

    def test_last_option(self):
        l = self.seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.last_option(), 3)
        self.assertEqual(l.last_option(no_wrap=True), 3)
        l = self.seq([1, 2, [3, 4]]).map(lambda x: x)
        self.assertEqual(l.last_option(), [3, 4])
        self.assertEqual(l.last_option(no_wrap=True), [3, 4])
        self.assert_type(l.last_option())
        self.assert_not_type(l.last_option(no_wrap=True))
        l = self.seq([])
        self.assertIsNone(l.last_option())
        self.assertIsNone(l.last_option(no_wrap=True))
        l = self.seq([deque(), deque()]).last_option()
        self.assert_type(l)
        l = self.seq([deque(), deque()]).last_option(no_wrap=True)
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).last_option()
        self.assert_not_type(l)
        l = self.seq([deque(), deque()], no_wrap=True).last_option(no_wrap=False)
        self.assert_type(l)

    def test_init(self):
        result = self.seq([1, 2, 3, 4]).map(lambda x: x).init()
        expect = [1, 2, 3]
        self.assertIteratorEqual(result, expect)

    def test_tail(self):
        l = self.seq([1, 2, 3, 4]).map(lambda x: x)
        expect = [2, 3, 4]
        self.assertIteratorEqual(l.tail(), expect)

    def test_inits(self):
        l = self.seq([1, 2, 3]).map(lambda x: x)
        expect = [[1, 2, 3], [1, 2], [1], []]
        self.assertIteratorEqual(l.inits(), expect)
        self.assertIteratorEqual(l.inits().map(lambda s: s.sum()), [6, 3, 1, 0])

    def test_tails(self):
        l = self.seq([1, 2, 3]).map(lambda x: x)
        expect = [[1, 2, 3], [2, 3], [3], []]
        self.assertIteratorEqual(l.tails(), expect)
        self.assertIteratorEqual(l.tails().map(lambda s: s.sum()), [6, 5, 3, 0])

    def test_drop(self):
        s = self.seq([1, 2, 3, 4, 5, 6])
        expect = [5, 6]
        result = s.drop(4)
        self.assertIteratorEqual(result, expect)
        self.assert_type(result)
        self.assertIteratorEqual(s.drop(0), s)
        self.assertIteratorEqual(s.drop(-1), s)

    def test_drop_right(self):
        s = self.seq([1, 2, 3, 4, 5]).map(lambda x: x)
        expect = [1, 2, 3]
        result = s.drop_right(2)
        self.assert_type(result)
        self.assertIteratorEqual(result, expect)
        self.assertIteratorEqual(s.drop_right(0), s)
        self.assertIteratorEqual(s.drop_right(-1), s)

        s = seq(1, 2, 3, 4, 5).filter(lambda x: x < 4)
        expect = [1, 2]
        result = s.drop_right(1)
        self.assert_type(result)
        self.assertIteratorEqual(result, expect)

        s = seq(5, 4, 3, 2, 1).sorted()
        expect = [1, 2, 3]
        result = s.drop_right(2)
        self.assert_type(result)
        self.assertIteratorEqual(result, expect)

    def test_drop_while(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8]
        expect = [4, 5, 6, 7, 8]
        result = self.seq(l).drop_while(lambda x: x < 4)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_take(self):
        s = self.seq([1, 2, 3, 4, 5, 6])
        expect = [1, 2, 3, 4]
        result = s.take(4)
        self.assertIteratorEqual(result, expect)
        self.assert_type(result)
        self.assertIteratorEqual(s.take(0), self.seq([]))
        self.assertIteratorEqual(s.take(-1), self.seq([]))

    def test_take_while(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8]
        expect = [1, 2, 3]
        result = self.seq(l).take_while(lambda x: x < 4)
        self.assertIteratorEqual(result, expect)
        self.assert_type(result)

    def test_union(self):
        result = self.seq([1, 1, 2, 3, 3]).union([1, 4, 5])
        expect = [1, 2, 3, 4, 5]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_intersection(self):
        result = self.seq([1, 2, 2, 3]).intersection([2, 3, 4, 5])
        expect = [2, 3]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_difference(self):
        result = self.seq([1, 2, 3]).difference([2, 3, 4])
        expect = [1]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_symmetric_difference(self):
        result = self.seq([1, 2, 3, 3]).symmetric_difference([2, 4, 5])
        expect = [1, 3, 4, 5]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_map(self):
        l = [1, 2, 0, 5]
        expect = [2, 4, 0, 10]
        result = self.seq(l).map(lambda x: x * 2)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_select(self):
        l = [1, 2, 0, 5]
        expect = [2, 4, 0, 10]
        result = self.seq(l).select(lambda x: x * 2)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_starmap(self):
        f = lambda x, y: x * y
        l = [(1, 1), (0, 3), (-3, 3), (4, 2)]
        expect = [1, 0, -9, 8]
        result = self.seq(l).starmap(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)
        result = self.seq(l).smap(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_filter(self):
        l = [0, -1, 5, 10]
        expect = [5, 10]
        s = self.seq(l)
        result = s.filter(lambda x: x > 0)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_where(self):
        def f(x):
            return x > 0

        l = [0, -1, 5, 10]
        expect = [5, 10]
        s = self.seq(l)
        result = s.where(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_filter_not(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [0, -1]
        result = self.seq(l).filter_not(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_map_filter(self):
        f = lambda x: x > 0
        g = lambda x: x * 2
        l = [0, -1, 5]
        s = self.seq(l)
        expect = [10]
        result = s.filter(f).map(g)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_reduce(self):
        f = lambda x, y: x + y
        l = ["a", "b", "c"]
        expect = "abc"
        s = self.seq(l)
        self.assertEqual(expect, s.reduce(f))

        with self.assertRaises(TypeError):
            seq([]).reduce(f)
        with self.assertRaises(ValueError):
            seq([]).reduce(f, 0, 0)

        self.assertEqual(seq([]).reduce(f, 1), 1)
        self.assertEqual(seq([0, 2]).reduce(f, 1), 3)

    def test_accumulate(self):
        f = lambda x, y: x + y
        l_char = ["a", "b", "c"]
        expect_char = ["a", "ab", "abc"]
        l_num = [1, 2, 3]
        expect_num = [1, 3, 6]
        self.assertEqual(seq(l_char).accumulate(), expect_char)
        self.assertEqual(seq(l_num).accumulate(), expect_num)

    def test_aggregate(self):
        f = lambda current, next_element: current + next_element
        l = self.seq([1, 2, 3, 4])
        self.assertEqual(l.aggregate(f), 10)
        self.assertEqual(l.aggregate(0, f), 10)
        self.assertEqual(l.aggregate(0, f, lambda x: 2 * x), 20)
        l = self.seq(["a", "b", "c"])
        self.assertEqual(l.aggregate(f), "abc")
        self.assertEqual(l.aggregate("", f), "abc")
        self.assertEqual(l.aggregate("", f, lambda x: x.upper()), "ABC")
        self.assertEqual(l.aggregate(f), "abc")
        self.assertEqual(l.aggregate("z", f), "zabc")
        self.assertEqual(l.aggregate("z", f, lambda x: x.upper()), "ZABC")
        with self.assertRaises(ValueError):
            l.aggregate()
        with self.assertRaises(ValueError):
            l.aggregate(None, None, None, None)

    def test_fold_left(self):
        f = lambda current, next_element: current + next_element
        l = self.seq([1, 2, 3, 4])
        self.assertEqual(l.fold_left(0, f), 10)
        self.assertEqual(l.fold_left(-10, f), 0)
        l = self.seq(["a", "b", "c"])
        self.assertEqual(l.fold_left("", f), "abc")
        self.assertEqual(l.fold_left("z", f), "zabc")
        f = lambda x, y: x + [y]
        self.assertEqual(l.fold_left([], f), ["a", "b", "c"])
        self.assertEqual(l.fold_left(["start"], f), ["start", "a", "b", "c"])

    def test_fold_right(self):
        f = lambda next_element, current: current + next_element
        l = self.seq([1, 2, 3, 4])
        self.assertEqual(l.fold_right(0, f), 10)
        self.assertEqual(l.fold_right(-10, f), 0)
        l = self.seq(["a", "b", "c"])
        self.assertEqual(l.fold_right("", f), "cba")
        self.assertEqual(l.fold_right("z", f), "zcba")
        f = lambda next_element, current: current + [next_element]
        self.assertEqual(l.fold_right([], f), ["c", "b", "a"])
        self.assertEqual(l.fold_right(["start"], f), ["start", "c", "b", "a"])

    def test_sorted(self):
        s = self.seq([1, 3, 2, 5, 4])
        r = s.sorted()
        self.assertIteratorEqual([1, 2, 3, 4, 5], r)
        self.assert_type(r)

    def test_order_by(self):
        s = self.seq([(2, "a"), (1, "b"), (4, "c"), (3, "d")])
        r = s.order_by(lambda x: x[0])
        self.assertIteratorEqual([(1, "b"), (2, "a"), (3, "d"), (4, "c")], r)
        self.assert_type(r)

    def test_reverse(self):
        l = [1, 2, 3]
        expect = [4, 3, 2]
        s = self.seq(l).map(lambda x: x + 1)
        result = s.reverse()
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)
        result = s.__reversed__()
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_distinct(self):
        l = [1, 3, 1, 2, 2, 3]
        expect = [1, 3, 2]
        s = self.seq(l)
        result = s.distinct()
        self.assertEqual(result.size(), len(expect))
        for er in zip(expect, result):
            self.assertEqual(
                er[0], er[1], "Order was not preserved after running distinct!"
            )
        for e in result:
            self.assertTrue(e in expect)
        self.assert_type(result)

    def test_distinct_by(self):
        s = self.seq(Data(1, 2), Data(1, 3), Data(2, 0), Data(3, -1), Data(1, 5))
        expect = {Data(1, 2), Data(2, 0), Data(3, -1)}
        result = s.distinct_by(lambda data: data.x)
        self.assertSetEqual(set(result), expect)
        self.assert_type(result)

    def test_slice(self):
        s = self.seq([1, 2, 3, 4])
        result = s.slice(1, 2)
        self.assertIteratorEqual(result, [2])
        self.assert_type(result)
        result = s.slice(1, 3)
        self.assertIteratorEqual(result, [2, 3])
        self.assert_type(result)

    def test_any(self):
        l = [True, False]
        self.assertTrue(self.seq(l).any())

    def test_all(self):
        l = [True, False]
        self.assertFalse(self.seq(l).all())
        l = [True, True]
        self.assertTrue(self.seq(l).all())

    def test_enumerate(self):
        l = [2, 3, 4]
        e = [(0, 2), (1, 3), (2, 4)]
        result = self.seq(l).enumerate()
        self.assertIteratorEqual(result, e)
        self.assert_type(result)

    def test_inner_join(self):
        l0 = [("a", 1), ("b", 2), ("c", 3)]
        l1 = [("a", 2), ("c", 4), ("d", 5)]
        result0 = self.seq(l0).inner_join(l1)
        result1 = self.seq(l0).join(l1, "inner")
        e = [("a", (1, 2)), ("c", (3, 4))]
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(e))
        self.assertDictEqual(dict(result1), dict(e))

        result0 = self.seq(l0).inner_join(self.seq(l1))
        result1 = self.seq(l0).join(self.seq(l1), "inner")
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(e))
        self.assertDictEqual(dict(result1), dict(e))

    def test_left_join(self):
        left = [("a", 1), ("b", 2)]
        right = [("a", 2), ("c", 3)]
        result0 = self.seq(left).left_join(right)
        result1 = self.seq(left).join(right, "left")
        expect = [("a", (1, 2)), ("b", (2, None))]
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

        result0 = self.seq(left).left_join(self.seq(right))
        result1 = self.seq(left).join(self.seq(right), "left")
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

    def test_right_join(self):
        left = [("a", 1), ("b", 2)]
        right = [("a", 2), ("c", 3)]
        result0 = self.seq(left).right_join(right)
        result1 = self.seq(left).join(right, "right")
        expect = [("a", (1, 2)), ("c", (None, 3))]

        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

        result0 = self.seq(left).right_join(self.seq(right))
        result1 = self.seq(left).join(self.seq(right), "right")
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

    def test_outer_join(self):
        left = [("a", 1), ("b", 2)]
        right = [("a", 2), ("c", 3)]
        result0 = self.seq(left).outer_join(right)
        result1 = self.seq(left).join(right, "outer")
        expect = [("a", (1, 2)), ("b", (2, None)), ("c", (None, 3))]

        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

        result0 = self.seq(left).outer_join(self.seq(right))
        result1 = self.seq(left).join(self.seq(right), "outer")
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

    def test_join(self):
        with self.assertRaises(TypeError):
            self.seq([(1, 2)]).join([(2, 3)], "").to_list()

    def test_max(self):
        l = [1, 2, 3]
        self.assertEqual(3, self.seq(l).max())

    def test_min(self):
        l = [1, 2, 3]
        self.assertEqual(1, self.seq(l).min())

    def test_max_by(self):
        l = ["aa", "bbbb", "c", "dd"]
        self.assertEqual("bbbb", self.seq(l).max_by(len))

    def test_min_by(self):
        l = ["aa", "bbbb", "c", "dd"]
        self.assertEqual("c", self.seq(l).min_by(len))

    def test_find(self):
        l = [1, 2, 3]
        f = lambda x: x == 3
        g = lambda x: False
        self.assertEqual(3, self.seq(l).find(f))
        self.assertIsNone(self.seq(l).find(g))

    def test_flatten(self):
        l = [[1, 1, 1], [2, 2, 2], [[3, 3], [4, 4]]]
        expect = [1, 1, 1, 2, 2, 2, [3, 3], [4, 4]]
        result = self.seq(l).flatten()
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_flat_map(self):
        l = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        f = lambda x: x
        expect = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        result = self.seq(l).flat_map(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_group_by(self):
        l = [(1, 1), (1, 2), (1, 3), (2, 2)]
        f = lambda x: x[0]
        expect = {1: [(1, 1), (1, 2), (1, 3)], 2: [(2, 2)]}
        result = self.seq(l).group_by(f)
        result_comparison = {}
        for kv in result:
            result_comparison[kv[0]] = kv[1]
        self.assertIteratorEqual(expect, result_comparison)
        self.assert_type(result)

    def test_group_by_key(self):
        l = [("a", 1), ("a", 2), ("a", 3), ("b", -1), ("b", 1), ("c", 10), ("c", 5)]
        e = {"a": [1, 2, 3], "b": [-1, 1], "c": [10, 5]}.items()
        result = self.seq(l).group_by_key()
        self.assertEqual(result.len(), len(e))
        for e0, e1 in zip(result, e):
            self.assertIteratorEqual(e0, e1)
        self.assert_type(result)

    def test_grouped(self):
        l = self.seq([1, 2, 3, 4, 5, 6, 7, 8])
        expect = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.assertIteratorEqual(map(list, l.grouped(2)), expect)
        expect = [[1, 2, 3], [4, 5, 6], [7, 8]]
        self.assertIteratorEqual(map(list, l.grouped(3)), expect)

    def test_grouped_returns_list(self):
        l = self.seq([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(is_iterable(l.grouped(2)))
        self.assertTrue(is_iterable(l.grouped(3)))

    def test_grouped_returns_list_of_lists(self):
        test_inputs = [
            [i for i in "abcdefghijklmnop"],
            [None for i in range(10)],
            [i for i in range(10)],
            [[i] for i in range(10)],
            [{i} for i in range(10)],
            [{i, i + 1} for i in range(10)],
            [[i, i + 1] for i in range(10)],
        ]

        def gen_test(collection, group_size):
            expected_type = type(collection[0])

            types_after_grouping = (
                seq(collection)
                .grouped(group_size)
                .flatten()
                .map(lambda item: type(item))
            )

            err_msg = f"Typing was not maintained after grouping. An input of {collection} yielded output types of {set(types_after_grouping)} and not {expected_type} as expected."

            return types_after_grouping.for_all(lambda t: t == expected_type), err_msg

        for test_input in test_inputs:
            for group_size in [1, 2, 4, 7]:
                all_sub_collections_are_lists, err_msg = gen_test(
                    test_input, group_size
                )
                self.assertTrue(all_sub_collections_are_lists, msg=err_msg)

    def test_sliding(self):
        l = self.seq([1, 2, 3, 4, 5, 6, 7])
        expect = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        self.assertIteratorEqual(l.sliding(2), expect)
        l = self.seq([1, 2, 3])
        expect = [[1, 2], [3]]
        self.assertIteratorEqual(l.sliding(2, 2), expect)
        expect = [[1, 2]]
        self.assertIteratorEqual(l.sliding(2, 3), expect)

    def test_empty(self):
        self.assertTrue(self.seq([]).empty())
        self.assertEqual(self.seq(), self.seq([]))

    def test_non_empty(self):
        self.assertTrue(self.seq([1]).non_empty())

    def test_non_zero_bool(self):
        self.assertTrue(bool(self.seq([1])))
        self.assertFalse(bool(self.seq([])))

    def test_make_string(self):
        l = [1, 2, 3]
        expect1 = "123"
        expect2 = "1:2:3"
        s = self.seq(l)
        self.assertEqual(expect1, s.make_string(""))
        self.assertEqual(expect2, s.make_string(":"))
        s = self.seq([])
        self.assertEqual("", s.make_string(""))
        self.assertEqual("", s.make_string(":"))

    def test_partition(self):
        l = [-1, -2, -3, 1, 2, 3]
        e2 = [-1, -2, -3]
        e1 = [1, 2, 3]
        f = lambda x: x > 0
        s = self.seq(l)
        p1, p2 = s.partition(f)
        self.assertIteratorEqual(e1, list(p1))
        self.assertIteratorEqual(e2, list(p2))
        self.assert_type(p1)
        self.assert_type(p2)

        result = self.seq([[1, 2, 3], [4, 5, 6]]).flatten().partition(lambda x: x > 2)
        expect = [[3, 4, 5, 6], [1, 2]]
        self.assertIteratorEqual(expect, list(result))
        self.assert_type(result)

    def test_cartesian(self):
        result = seq.range(3).cartesian(range(3)).list()
        self.assertListEqual(result, list(product(range(3), range(3))))

        result = seq.range(3).cartesian(range(3), range(2)).list()
        self.assertListEqual(result, list(product(range(3), range(3), range(2))))

        result = seq.range(3).cartesian(range(3), range(2), repeat=2).list()
        self.assertListEqual(
            result, list(product(range(3), range(3), range(2), repeat=2))
        )

    def test_product(self):
        l = [2, 2, 3]
        self.assertEqual(12, self.seq(l).product())
        self.assertEqual(96, self.seq(l).product(lambda x: x * 2))
        s = self.seq([])
        self.assertEqual(1, s.product())
        self.assertEqual(2, s.product(lambda x: x * 2))
        s = self.seq([5])
        self.assertEqual(5, s.product())
        self.assertEqual(10, s.product(lambda x: x * 2))

    def test_sum(self):
        l = [1, 2, 3]
        self.assertEqual(6, self.seq(l).sum())
        self.assertEqual(12, self.seq(l).sum(lambda x: x * 2))

    def test_average(self):
        l = [1, 2]
        self.assertEqual(1.5, self.seq(l).average())
        self.assertEqual(4.5, self.seq(l).average(lambda x: x * 3))

    def test_set(self):
        l = [1, 1, 2, 2, 3]
        ls = set(l)
        self.assertIteratorEqual(ls, self.seq(l).set())

    def test_zip(self):
        l1 = [1, 2, 3]
        l2 = [-1, -2, -3]
        e = [(1, -1), (2, -2), (3, -3)]
        result = self.seq(l1).zip(l2)
        self.assertIteratorEqual(e, result)
        self.assert_type(result)

    def test_zip_with_index(self):
        l = [2, 3, 4]
        e = [(2, 0), (3, 1), (4, 2)]
        result = self.seq(l).zip_with_index()
        self.assertIteratorEqual(result, e)
        self.assert_type(result)
        e = [(2, 5), (3, 6), (4, 7)]
        result = self.seq(l).zip_with_index(5)
        self.assertIteratorEqual(result, e)
        self.assert_type(result)

    def test_to_list(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        result = self.seq(l).to_list()
        self.assertIteratorEqual(result, l)
        self.assertTrue(isinstance(result, list))
        result = self.seq(iter([0, 1, 2])).to_list()
        self.assertIsInstance(result, list)
        result = self.seq(l).list(n=2)
        self.assertEqual(result, [1, 2])

    def test_list(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        result = self.seq(l).list()
        self.assertEqual(result, l)
        self.assertTrue(isinstance(result, list))
        result = self.seq(iter([0, 1, 2])).to_list()
        self.assertIsInstance(result, list)
        result = self.seq(l).list(n=2)
        self.assertEqual(result, [1, 2])

    def test_for_each(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        result = []

        def f(e):
            result.append(e)

        self.seq(l).for_each(f)
        self.assertEqual(result, l)

    def test_peek(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        result = []

        def f(e):
            result.append(e)

        result_iter = self.seq(l).peek(f).list()
        self.assertIteratorEqual(result_iter, l)
        self.assertEqual(result, l)

    def test_exists(self):
        l = ["aaa", "BBB", "ccc"]
        self.assertTrue(self.seq(l).exists(str.islower))
        self.assertTrue(self.seq(l).exists(str.isupper))
        self.assertFalse(self.seq(l).exists(lambda s: "d" in s))

    def test_for_all(self):
        l = ["aaa", "bbb", "ccc"]
        self.assertTrue(self.seq(l).for_all(str.islower))
        self.assertFalse(self.seq(l).for_all(str.isupper))

    def test_to_dict(self):
        l = [(1, 2), (2, 10), (7, 2)]
        d = {1: 2, 2: 10, 7: 2}
        result = self.seq(l).to_dict()
        self.assertDictEqual(result, d)
        self.assertTrue(isinstance(result, dict))
        result = self.seq(l).to_dict(default=lambda: 100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)
        result = self.seq(l).dict(default=100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)

    def test_dict(self):
        l = [(1, 2), (2, 10), (7, 2)]
        d = {1: 2, 2: 10, 7: 2}
        result = self.seq(l).dict()
        self.assertDictEqual(result, d)
        self.assertTrue(isinstance(result, dict))
        result = self.seq(l).dict(default=lambda: 100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)
        result = self.seq(l).dict(default=100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)

    def test_reduce_by_key(self):
        l = [("a", 1), ("a", 2), ("a", 3), ("b", -1), ("b", 1), ("c", 10), ("c", 5)]
        e = {"a": 6, "b": 0, "c": 15}.items()
        result = self.seq(l).reduce_by_key(lambda x, y: x + y)
        self.assertEqual(result.len(), len(e))
        for e0, e1 in zip(result, e):
            self.assertEqual(e0, e1)
        self.assert_type(result)

    def test_count_by_key(self):
        l = [
            ("a", 1),
            ("a", 2),
            ("a", 3),
            ("b", -1),
            ("b", 1),
            ("c", 10),
            ("c", 5),
            ("d", 1),
        ]
        e = {"a": 3, "b": 2, "c": 2, "d": 1}.items()
        result = self.seq(l).count_by_key()
        self.assertEqual(result.len(), len(e))
        for e0, e1 in zip(result, e):
            self.assertEqual(e0, e1)
        self.assert_type(result)

    def test_count_by_value(self):
        l = ["a", "a", "a", "b", "b", "c", "d"]
        e = {"a": 3, "b": 2, "c": 1, "d": 1}.items()
        result = self.seq(l).count_by_value()
        self.assertEqual(result.len(), len(e))
        for e0, e1 in zip(result, e):
            self.assertEqual(e0, e1)
        self.assert_type(result)

    def test_wrap(self):
        self.assert_type(_wrap([1, 2]))
        self.assert_type(_wrap((1, 2)))
        self.assert_not_type(_wrap(1))
        self.assert_not_type(_wrap(1.0))
        self.assert_not_type(_wrap("test"))
        self.assert_not_type(_wrap(True))
        self.assert_not_type(_wrap(Data(1, 2)))

    def test_wrap_objects(self):
        class A(object):
            a = 1

        l = [A(), A(), A()]
        self.assertIsInstance(_wrap(A()), A)
        self.assert_type(self.seq(l))

    @unittest.skipUnless(
        pandas_is_installed(), "Skip pandas tests if pandas is not installed"
    )
    def test_wrap_pandas(self):
        df1 = pandas.DataFrame({"name": ["name1", "name2"], "value": [1, 2]})
        df2 = pandas.DataFrame({"name": ["name1", "name2"], "value": [3, 4]})
        result = seq([df1, df2]).reduce(lambda x, y: pandas.concat([x, y]))
        self.assertEqual(result.len(), 4)
        self.assertEqual(result[0].to_list(), ["name1", 1])
        self.assertEqual(result[1].to_list(), ["name2", 2])
        self.assertEqual(result[2].to_list(), ["name1", 3])
        self.assertEqual(result[3].to_list(), ["name2", 4])

    def test_iterator_consumption(self):
        sequence = self.seq([1, 2, 3])
        first_transform = sequence.map(lambda x: x)
        second_transform = first_transform.map(lambda x: x)
        first_list = list(first_transform)
        second_list = list(second_transform)
        expect = [1, 2, 3]
        self.assertIteratorEqual(first_list, expect)
        self.assertIteratorEqual(second_list, expect)

    def test_single_call(self):
        if self.seq is pseq:
            raise self.skipTest("pseq doesn't support functions with side-effects")
        counter = []

        def counter_func(x):
            counter.append(1)
            return x

        list(self.seq([1, 2, 3, 4]).map(counter_func))
        self.assertEqual(len(counter), 4)

    def test_seq(self):
        self.assertIteratorEqual(self.seq([1, 2, 3]), [1, 2, 3])
        self.assertIteratorEqual(self.seq(1, 2, 3), [1, 2, 3])
        self.assertIteratorEqual(self.seq(1), [1])
        self.assertIteratorEqual(self.seq(iter([1, 2, 3])), [1, 2, 3])
        self.assertIteratorEqual(self.seq(), [])

    def test_lineage_repr(self):
        s = self.seq(1).map(lambda x: x).filter(lambda x: True)
        self.assertEqual(
            repr(s._lineage), "Lineage: sequence -> map(<lambda>) -> filter(<lambda>)"
        )

    def test_cache(self):
        if self.seq is pseq:
            raise self.skipTest("pseq doesn't support functions with side-effects")
        calls = []
        func = calls.append
        result = self.seq(1, 2, 3).map(func).cache().map(lambda x: x).to_list()
        self.assertEqual(len(calls), 3)
        self.assertEqual(result, [None, None, None])
        result = self.seq(1, 2, 3).map(lambda x: x).cache()
        self.assertEqual(
            repr(result._lineage), "Lineage: sequence -> map(<lambda>) -> cache"
        )
        result = self.seq(1, 2, 3).map(lambda x: x).cache(delete_lineage=True)
        self.assertEqual(repr(result._lineage), "Lineage: sequence")

    def test_tabulate(self):
        sequence = seq([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(sequence.show(), None)
        self.assertNotEqual(sequence._repr_html_(), None)
        result = sequence.tabulate()
        self.assertEqual(result, "-  -  -\n1  2  3\n4  5  6\n-  -  -")

        sequence = seq(1, 2, 3)
        self.assertEqual(sequence.tabulate(), None)

        class NotTabulatable(object):
            pass

        sequence = seq(NotTabulatable(), NotTabulatable(), NotTabulatable())
        self.assertEqual(sequence.tabulate(), None)

        long_data = seq([(i, i + 1) for i in range(30)])
        self.assertTrue("Showing 10 of 30 rows" in long_data.tabulate(n=10))
        self.assertTrue("Showing 10 of 30 rows" in long_data._repr_html_())
        self.assertTrue(
            "Showing 10 of 30 rows" not in long_data.tabulate(n=10, tablefmt="plain")
        )

    def test_tabulate_namedtuple(self):
        sequence_tabulated = seq([Data(1, 2), Data(6, 7)]).tabulate()
        self.assertEqual(sequence_tabulated, "  x    y\n---  ---\n  1    2\n  6    7")

    def test_repr_max_lines(self):
        sequence = seq.range(200)
        self.assertEqual(len(repr(sequence)), 395)
        sequence._max_repr_items = None
        self.assertEqual(len(repr(sequence)), 890)


class TestExtend(unittest.TestCase):
    def test_custom_functions(self):
        @extend(aslist=True)
        def my_zip(it):
            return zip(it, it)

        result = seq.range(3).my_zip().list()
        expected = list(zip(range(3), range(3)))
        self.assertEqual(result, expected)

        result = seq.range(3).my_zip().my_zip().list()
        expected = list(zip(expected, expected))
        self.assertEqual(result, expected)

        @extend
        def square(it):
            return [i**2 for i in it]

        result = seq.range(100).square().list()
        expected = [i**2 for i in range(100)]
        self.assertEqual(result, expected)

        name = "PARALLEL_SQUARE"

        @extend(parallel=True, name=name)
        def square_parallel(it):
            return [i**2 for i in it]

        result = seq.range(100).square_parallel()
        self.assertEqual(result.sum(), sum(expected))
        self.assertEqual(
            repr(result._lineage), f"Lineage: sequence -> extended[{name}]"
        )

        @extend
        def my_filter(it, n=10):
            return (i for i in it if i > n)

        # test keyword args
        result = seq.range(20).my_filter(n=10).list()
        expected = list(filter(lambda x: x > 10, range(20)))
        self.assertEqual(result, expected)

        # test args
        result = seq.range(20).my_filter(10).list()
        self.assertEqual(result, expected)

        # test final
        @extend(final=True)
        def toarray(it):
            return array.array("f", it)

        result = seq.range(10).toarray()
        expected = array.array("f", range(10))
        self.assertEqual(result, expected)

        result = seq.range(10).map(lambda x: x**2).toarray()
        expected = array.array("f", [i**2 for i in range(10)])
        self.assertEqual(result, expected)

        # a more complex example combining all above
        @extend()
        def sum_pair(it):
            return (i[0] + i[1] for i in it)

        result = (
            seq.range(100).my_filter(85).my_zip().sum_pair().square_parallel().toarray()
        )

        expected = array.array(
            "f",
            list(
                map(
                    lambda x: (x[0] + x[1]) ** 2,
                    map(lambda x: (x, x), filter(lambda x: x > 85, range(100))),
                )
            ),
        )
        self.assertEqual(result, expected)


class TestParallelPipeline(TestPipeline):
    def setUp(self):
        self.seq = pseq
