#pylint: skip-file
from __future__ import absolute_import

import unittest
from collections import namedtuple
from functional.pipeline import Sequence, is_iterable, _wrap
from functional.transformations import name
from functional import seq

Data = namedtuple('Data', 'x y')


class TestPipeline(unittest.TestCase):
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
        self.assert_type(seq(l))
        self.assert_not_type(seq(l).sequence)
        self.assert_type(seq(seq(l)))
        self.assert_not_type(seq(seq(l)).sequence)
        self.assert_not_type(seq(l)._base_sequence)

    def test_eq(self):
        l = [1, 2, 3]
        self.assertIteratorEqual(seq(l).map(lambda x: x), seq(l))

    def test_ne(self):
        a = [1, 2, 3]
        b = [1]
        self.assertNotEqual(seq(a), seq(b))

    def test_repr(self):
        l = [1, 2, 3]
        self.assertEqual(repr(l), repr(seq(l)))

    def test_lineage_name(self):
        f = lambda x: x
        self.assertEqual(f.__name__, name(f))
        f = "test"
        self.assertEqual("test", name(f))

    def test_str(self):
        l = [1, 2, 3]
        self.assertEqual(str(l), str(seq(l)))

    def test_hash(self):
        self.assertRaises(TypeError, lambda: hash(seq([1])))

    def test_len(self):
        l = [1, 2, 3]
        s = seq(l)
        self.assertEqual(len(l), s.size())
        self.assertEqual(len(l), s.len())

    def test_count(self):
        l = seq([-1, -1, 1, 1, 1])
        self.assertEqual(l.count(lambda x: x > 0), 3)
        self.assertEqual(l.count(lambda x: x < 0), 2)

    def test_getitem(self):
        l = [1, 2, [3, 4, 5]]
        s = seq(l).map(lambda x: x)
        self.assertEqual(s[1], 2)
        self.assertEqual(s[2], [3, 4, 5])
        self.assert_type(s[2])
        self.assertEqual(s[1:], [2, [3, 4, 5]])
        self.assert_type(s[1:])
        l = [{1, 2}, {2, 3}, {4, 5}]
        s = seq(l)
        self.assertIsInstance(s[0], set)
        self.assertEqual(s[0], l[0])

    def test_iter(self):
        l = list(enumerate(seq([1, 2, 3])))
        e = list(enumerate([1, 2, 3]))
        self.assertEqual(l, e)
        l = seq([1, 2, 3])
        e = [1, 2, 3]
        result = []
        for n in l:
            result.append(n)
        self.assertEqual(result, e)
        self.assert_type(l)

    def test_contains(self):
        string = "abcdef"
        s = seq(iter(string)).map(lambda x: x)
        self.assertTrue("c" in s)

    def test_add(self):
        l0 = seq([1, 2, 3]).map(lambda x: x)
        l1 = seq([4, 5, 6])
        l2 = [4, 5, 6]
        expect = [1, 2, 3, 4, 5, 6]
        self.assertEqual(l0 + l1, expect)
        self.assertEqual(l0 + l2, expect)

    def test_head(self):
        l = seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.head(), 1)
        l = seq([[1, 2], 3, 4])
        self.assertEqual(l.head(), [1, 2])
        self.assert_type(l.head())
        l = seq([])
        with self.assertRaises(IndexError):
            l.head()

    def test_first(self):
        l = seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.first(), 1)
        l = seq([[1, 2], 3, 4]).map(lambda x: x)
        self.assertEqual(l.first(), [1, 2])
        self.assert_type(l.first())
        l = seq([])
        with self.assertRaises(IndexError):
            l.head()

    def test_head_option(self):
        l = seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.head_option(), 1)
        l = seq([[1, 2], 3, 4]).map(lambda x: x)
        self.assertEqual(l.head_option(), [1, 2])
        self.assert_type(l.head_option())
        l = seq([])
        self.assertIsNone(l.head_option())

    def test_last(self):
        l = seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.last(), 3)
        l = seq([1, 2, [3, 4]]).map(lambda x: x)
        self.assertEqual(l.last(), [3, 4])
        self.assert_type(l.last())

    def test_last_option(self):
        l = seq([1, 2, 3]).map(lambda x: x)
        self.assertEqual(l.last_option(), 3)
        l = seq([1, 2, [3, 4]]).map(lambda x: x)
        self.assertEqual(l.last_option(), [3, 4])
        self.assert_type(l.last_option())
        l = seq([])
        self.assertIsNone(l.last_option())

    def test_init(self):
        result = seq([1, 2, 3, 4]).map(lambda x: x).init()
        expect = [1, 2, 3]
        self.assertIteratorEqual(result, expect)

    def test_tail(self):
        l = seq([1, 2, 3, 4]).map(lambda x: x)
        expect = [2, 3, 4]
        self.assertIteratorEqual(l.tail(), expect)

    def test_inits(self):
        l = seq([1, 2, 3]).map(lambda x: x)
        expect = [[1, 2, 3], [1, 2], [1], []]
        self.assertIteratorEqual(l.inits(), expect)
        self.assertIteratorEqual(l.inits().map(lambda s: s.sum()), [6, 3, 1, 0])

    def test_tails(self):
        l = seq([1, 2, 3]).map(lambda x: x)
        expect = [[1, 2, 3], [2, 3], [3], []]
        self.assertIteratorEqual(l.tails(), expect)
        self.assertIteratorEqual(l.tails().map(lambda s: s.sum()), [6, 5, 3, 0])

    def test_drop(self):
        s = seq([1, 2, 3, 4, 5, 6])
        expect = [5, 6]
        result = s.drop(4)
        self.assertIteratorEqual(result, expect)
        self.assert_type(result)
        self.assertIteratorEqual(s.drop(0), s)
        self.assertIteratorEqual(s.drop(-1), s)

    def test_drop_right(self):
        s = seq([1, 2, 3, 4, 5]).map(lambda x: x)
        expect = [1, 2, 3]
        result = s.drop_right(2)
        self.assert_type(result)
        self.assertIteratorEqual(result, expect)
        self.assertIteratorEqual(s.drop_right(0), s)
        self.assertIteratorEqual(s.drop_right(-1), s)

    def test_drop_while(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8]
        f = lambda x: x < 4
        expect = [4, 5, 6, 7, 8]
        result = seq(l).drop_while(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_take(self):
        s = seq([1, 2, 3, 4, 5, 6])
        expect = [1, 2, 3, 4]
        result = s.take(4)
        self.assertIteratorEqual(result, expect)
        self.assert_type(result)
        self.assertIteratorEqual(s.take(0), seq([]))
        self.assertIteratorEqual(s.take(-1), seq([]))

    def test_take_while(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8]
        f = lambda x: x < 4
        expect = [1, 2, 3]
        result = seq(l).take_while(f)
        self.assertIteratorEqual(result, expect)
        self.assert_type(result)

    def test_union(self):
        result = seq([1, 1, 2, 3, 3]).union([1, 4, 5])
        expect = [1, 2, 3, 4, 5]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_intersection(self):
        result = seq([1, 2, 2, 3]).intersection([2, 3, 4, 5])
        expect = [2, 3]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_difference(self):
        result = seq([1, 2, 3]).difference([2, 3, 4])
        expect = [1]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_symmetric_difference(self):
        result = seq([1, 2, 3, 3]).symmetric_difference([2, 4, 5])
        expect = [1, 3, 4, 5]
        self.assert_type(result)
        self.assertSetEqual(result.set(), set(expect))

    def test_map(self):
        f = lambda x: x * 2
        l = [1, 2, 0, 5]
        expect = [2, 4, 0, 10]
        result = seq(l).map(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_select(self):
        f = lambda x: x * 2
        l = [1, 2, 0, 5]
        expect = [2, 4, 0, 10]
        result = seq(l).select(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_filter(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [5, 10]
        s = seq(l)
        result = s.filter(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_where(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [5, 10]
        s = seq(l)
        result = s.where(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_filter_not(self):
        f = lambda x: x > 0
        l = [0, -1, 5, 10]
        expect = [0, -1]
        result = seq(l).filter_not(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_map_filter(self):
        f = lambda x: x > 0
        g = lambda x: x * 2
        l = [0, -1, 5]
        s = seq(l)
        expect = [10]
        result = s.filter(f).map(g)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_reduce(self):
        f = lambda x, y: x + y
        l = ["a", "b", "c"]
        expect = "abc"
        s = seq(l)
        self.assertEqual(expect, s.reduce(f))

    def test_aggregate(self):
        f = lambda current, next_element: current + next_element
        l = seq([1, 2, 3, 4])
        self.assertEqual(l.aggregate(f), 10)
        self.assertEqual(l.aggregate(0, f), 10)
        self.assertEqual(l.aggregate(0, f, lambda x: 2 * x), 20)
        l = seq(['a', 'b', 'c'])
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
        l = seq([1, 2, 3, 4])
        self.assertEqual(l.fold_left(0, f), 10)
        self.assertEqual(l.fold_left(-10, f), 0)
        l = seq(['a', 'b', 'c'])
        self.assertEqual(l.fold_left("", f), "abc")
        self.assertEqual(l.fold_left("z", f), "zabc")
        f = lambda x, y: x + [y]
        self.assertEqual(l.fold_left([], f), ['a', 'b', 'c'])
        self.assertEqual(l.fold_left(['start'], f), ['start', 'a', 'b', 'c'])

    def test_fold_right(self):
        f = lambda next_element, current: current + next_element
        l = seq([1, 2, 3, 4])
        self.assertEqual(l.fold_right(0, f), 10)
        self.assertEqual(l.fold_right(-10, f), 0)
        l = seq(['a', 'b', 'c'])
        self.assertEqual(l.fold_right("", f), "cba")
        self.assertEqual(l.fold_right("z", f), "zcba")
        f = lambda next_element, current: current + [next_element]
        self.assertEqual(l.fold_right([], f), ['c', 'b', 'a'])
        self.assertEqual(l.fold_right(['start'], f), ['start', 'c', 'b', 'a'])

    def test_sorted(self):
        s = seq([1, 3, 2, 5, 4])
        r = s.sorted()
        self.assertIteratorEqual([1, 2, 3, 4, 5], r)
        self.assert_type(r)

    def test_order_by(self):
        s = seq([(2, 'a'), (1, 'b'), (4, 'c'), (3, 'd')])
        r = s.order_by(lambda x: x[0])
        self.assertIteratorEqual([(1, 'b'), (2, 'a'), (3, 'd'), (4, 'c')], r)
        self.assert_type(r)

    def test_reverse(self):
        l = [1, 2, 3]
        expect = [3, 2, 1]
        s = seq(l)
        result = s.reverse()
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)
        result = s.__reversed__()
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_distinct(self):
        l = [1, 1, 2, 3, 2, 3]
        expect = [1, 2, 3]
        s = seq(l)
        result = s.distinct()
        for e in result:
            self.assertTrue(e in expect)
        result = s.distinct()
        self.assertEqual(result.size(), len(expect))
        self.assert_type(result)

    def test_distinct_by(self):
        s = seq(Data(1, 2), Data(1, 3), Data(2, 0), Data(3, -1), Data(1, 5))
        expect = {Data(1, 2), Data(2, 0), Data(3, -1)}
        result = s.distinct_by(lambda data: data.x)
        self.assertSetEqual(set(result), expect)
        self.assert_type(result)

    def test_slice(self):
        s = seq([1, 2, 3, 4])
        result = s.slice(1, 2)
        self.assertIteratorEqual(result, [2])
        self.assert_type(result)
        result = s.slice(1, 3)
        self.assertIteratorEqual(result, [2, 3])
        self.assert_type(result)

    def test_any(self):
        l = [True, False]
        self.assertTrue(seq(l).any())

    def test_all(self):
        l = [True, False]
        self.assertFalse(seq(l).all())
        l = [True, True]
        self.assertTrue(seq(l).all())

    def test_enumerate(self):
        l = [2, 3, 4]
        e = [(0, 2), (1, 3), (2, 4)]
        result = seq(l).enumerate()
        self.assertIteratorEqual(result, e)
        self.assert_type(result)

    def test_inner_join(self):
        l0 = [('a', 1), ('b', 2), ('c', 3)]
        l1 = [('a', 2), ('c', 4), ('d', 5)]
        result0 = seq(l0).inner_join(l1)
        result1 = seq(l0).join(l1, 'inner')
        e = [('a', (1, 2)), ('c', (3, 4))]
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(e))
        self.assertDictEqual(dict(result1), dict(e))

        result0 = seq(l0).inner_join(seq(l1))
        result1 = seq(l0).join(seq(l1), 'inner')
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(e))
        self.assertDictEqual(dict(result1), dict(e))

    def test_left_join(self):
        left = [('a', 1), ('b', 2)]
        right = [('a', 2), ('c', 3)]
        result0 = seq(left).left_join(right)
        result1 = seq(left).join(right, 'left')
        expect = [('a', (1, 2)), ('b', (2, None))]
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

        result0 = seq(left).left_join(seq(right))
        result1 = seq(left).join(seq(right), 'left')
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

    def test_right_join(self):
        left = [('a', 1), ('b', 2)]
        right = [('a', 2), ('c', 3)]
        result0 = seq(left).right_join(right)
        result1 = seq(left).join(right, 'right')
        expect = [('a', (1, 2)), ('c', (None, 3))]

        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

        result0 = seq(left).right_join(seq(right))
        result1 = seq(left).join(seq(right), 'right')
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

    def test_outer_join(self):
        left = [('a', 1), ('b', 2)]
        right = [('a', 2), ('c', 3)]
        result0 = seq(left).outer_join(right)
        result1 = seq(left).join(right, 'outer')
        expect = [('a', (1, 2)), ('b', (2, None)), ('c', (None, 3))]

        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

        result0 = seq(left).outer_join(seq(right))
        result1 = seq(left).join(seq(right), 'outer')
        self.assert_type(result0)
        self.assert_type(result1)
        self.assertDictEqual(dict(result0), dict(expect))
        self.assertDictEqual(dict(result1), dict(expect))

    def test_join(self):
        with self.assertRaises(TypeError):
            seq([(1, 2)]).join([(2, 3)], '').to_list()

    def test_max(self):
        l = [1, 2, 3]
        self.assertEqual(3, seq(l).max())

    def test_min(self):
        l = [1, 2, 3]
        self.assertEqual(1, seq(l).min())

    def test_max_by(self):
        l = ["aa", "bbbb", "c", "dd"]
        self.assertEqual("bbbb", seq(l).max_by(len))

    def test_min_by(self):
        l = ["aa", "bbbb", "c", "dd"]
        self.assertEqual("c", seq(l).min_by(len))

    def test_find(self):
        l = [1, 2, 3]
        f = lambda x: x == 3
        g = lambda x: False
        self.assertEqual(3, seq(l).find(f))
        self.assertIsNone(seq(l).find(g))

    def test_flatten(self):
        l = [[1, 1, 1], [2, 2, 2], [[3, 3], [4, 4]]]
        expect = [1, 1, 1, 2, 2, 2, [3, 3], [4, 4]]
        result = seq(l).flatten()
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_flat_map(self):
        l = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        f = lambda x: x
        expect = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        result = seq(l).flat_map(f)
        self.assertIteratorEqual(expect, result)
        self.assert_type(result)

    def test_group_by(self):
        l = [(1, 1), (1, 2), (1, 3), (2, 2)]
        f = lambda x: x[0]
        expect = {1: [(1, 1), (1, 2), (1, 3)], 2: [(2, 2)]}
        result = seq(l).group_by(f)
        result_comparison = {}
        for kv in result:
            result_comparison[kv[0]] = kv[1]
        self.assertIteratorEqual(expect, result_comparison)
        self.assert_type(result)

    def test_group_by_key(self):
        l = [('a', 1), ('a', 2), ('a', 3), ('b', -1), ('b', 1), ('c', 10), ('c', 5)]
        e = {"a": [1, 2, 3], "b": [-1, 1], "c": [10, 5]}.items()
        result = seq(l).group_by_key()
        self.assertEqual(result.len(), len(e))
        for e0, e1 in zip(result, e):
            self.assertIteratorEqual(e0, e1)
        self.assert_type(result)

    def test_grouped(self):
        l = seq([1, 2, 3, 4, 5, 6, 7, 8])
        expect = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.assertIteratorEqual(l.grouped(2), expect)
        expect = [[1, 2, 3], [4, 5, 6], [7, 8]]
        self.assertIteratorEqual(l.grouped(3), expect)

    def test_sliding(self):
        l = seq([1, 2, 3, 4, 5, 6, 7])
        expect = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        self.assertIteratorEqual(l.sliding(2), expect)
        l = seq([1, 2, 3])
        expect = [[1, 2], [3]]
        self.assertIteratorEqual(l.sliding(2, 2), expect)
        expect = [[1, 2]]
        self.assertIteratorEqual(l.sliding(2, 3), expect)

    def test_empty(self):
        self.assertTrue(seq([]).empty())

    def test_non_empty(self):
        self.assertTrue(seq([1]).non_empty())

    def test_non_zero_bool(self):
        self.assertTrue(bool(seq([1])))
        self.assertFalse(bool(seq([])))

    def test_make_string(self):
        l = [1, 2, 3]
        expect1 = "123"
        expect2 = "1:2:3"
        s = seq(l)
        self.assertEqual(expect1, s.make_string(""))
        self.assertEqual(expect2, s.make_string(":"))
        s = seq([])
        self.assertEqual("", s.make_string(""))
        self.assertEqual("", s.make_string(":"))

    def test_partition(self):
        l = [-1, -2, -3, 1, 2, 3]
        e2 = [-1, -2, -3]
        e1 = [1, 2, 3]
        f = lambda x: x > 0
        s = seq(l)
        p1, p2 = s.partition(f)
        self.assertIteratorEqual(e1, list(p1))
        self.assertIteratorEqual(e2, list(p2))
        self.assert_type(p1)
        self.assert_type(p2)

    def test_product(self):
        l = [2, 2, 3]
        self.assertEqual(12, seq(l).product())
        self.assertEqual(96, seq(l).product(lambda x: x * 2))
        s = seq([])
        self.assertEqual(1, s.product())
        self.assertEqual(2, s.product(lambda x: x * 2))
        s = seq([5])
        self.assertEqual(5, s.product())
        self.assertEqual(10, s.product(lambda x: x * 2))

    def test_sum(self):
        l = [1, 2, 3]
        self.assertEqual(6, seq(l).sum())
        self.assertEqual(12, seq(l).sum(lambda x: x * 2))

    def test_average(self):
        l = [1, 2]
        self.assertEqual(1.5, seq(l).average())
        self.assertEquals(4.5, seq(l).average(lambda x: x * 3))

    def test_set(self):
        l = [1, 1, 2, 2, 3]
        ls = set(l)
        self.assertIteratorEqual(ls, seq(l).set())

    def test_zip(self):
        l1 = [1, 2, 3]
        l2 = [-1, -2, -3]
        e = [(1, -1), (2, -2), (3, -3)]
        result = seq(l1).zip(l2)
        self.assertIteratorEqual(e, result)
        self.assert_type(result)

    def test_zip_with_index(self):
        l = [2, 3, 4]
        e = [(2, 0), (3, 1), (4, 2)]
        result = seq(l).zip_with_index()
        self.assertIteratorEqual(result, e)
        self.assert_type(result)
        e = [(2, 5), (3, 6), (4, 7)]
        result = seq(l).zip_with_index(5)
        self.assertIteratorEqual(result, e)
        self.assert_type(result)

    def test_to_list(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        result = seq(l).to_list()
        self.assertIteratorEqual(result, l)
        self.assertTrue(isinstance(result, list))
        result = seq(iter([0, 1, 2])).to_list()
        self.assertIsInstance(result, list)

    def test_list(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        result = seq(l).list()
        self.assertEqual(result, l)
        self.assertTrue(isinstance(result, list))
        result = seq(iter([0, 1, 2])).to_list()
        self.assertIsInstance(result, list)

    def test_for_each(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        result = []
        def f(e):
            result.append(e)
        seq(l).for_each(f)
        self.assertEqual(result, l)

    def test_exists(self):
        l = ["aaa", "BBB", "ccc"]
        self.assertTrue(seq(l).exists(str.islower))
        self.assertTrue(seq(l).exists(str.isupper))
        self.assertFalse(seq(l).exists(lambda s: "d" in s))

    def test_for_all(self):
        l = ["aaa", "bbb", "ccc"]
        self.assertTrue(seq(l).for_all(str.islower))
        self.assertFalse(seq(l).for_all(str.isupper))

    def test_to_dict(self):
        l = [(1, 2), (2, 10), (7, 2)]
        d = {1: 2, 2: 10, 7: 2}
        result = seq(l).to_dict()
        self.assertDictEqual(result, d)
        self.assertTrue(isinstance(result, dict))
        result = seq(l).to_dict(default=lambda: 100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)
        result = seq(l).dict(default=100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)

    def test_dict(self):
        l = [(1, 2), (2, 10), (7, 2)]
        d = {1: 2, 2: 10, 7: 2}
        result = seq(l).dict()
        self.assertDictEqual(result, d)
        self.assertTrue(isinstance(result, dict))
        result = seq(l).dict(default=lambda: 100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)
        result = seq(l).dict(default=100)
        self.assertTrue(1 in result)
        self.assertFalse(3 in result)
        self.assertEqual(result[4], 100)

    def test_reduce_by_key(self):
        l = [('a', 1), ('a', 2), ('a', 3), ('b', -1), ('b', 1), ('c', 10), ('c', 5)]
        e = {"a": 6, "b": 0, "c": 15}.items()
        result = seq(l).reduce_by_key(lambda x, y: x + y)
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
        self.assert_type(seq(l))

    def test_iterator_consumption(self):
        sequence = seq([1, 2, 3])
        first_transform = sequence.map(lambda x: x)
        second_transform = first_transform.map(lambda x: x)
        first_list = list(first_transform)
        second_list = list(second_transform)
        expect = [1, 2, 3]
        self.assertIteratorEqual(first_list, expect)
        self.assertIteratorEqual(second_list, expect)

    def test_single_call(self):
        counter = []

        def counter_func(x):
            counter.append(1)
            return x

        list(seq([1, 2, 3, 4]).map(counter_func))
        self.assertEqual(len(counter), 4)

    def test_seq(self):
        self.assertIteratorEqual(seq([1, 2, 3]), [1, 2, 3])
        self.assertIteratorEqual(seq(1, 2, 3), [1, 2, 3])
        self.assertIteratorEqual(seq(1), [1])
        self.assertIteratorEqual(seq(iter([1, 2, 3])), [1, 2, 3])
        with self.assertRaises(TypeError):
            seq()

    def test_lineage_repr(self):
        s = seq(1).map(lambda x: x).filter(lambda x: True)
        self.assertEqual(repr(s._lineage), 'Lineage: sequence -> map(<lambda>) -> filter(<lambda>)')

    def test_cache(self):
        calls = []
        func = lambda x: calls.append(x)
        result = seq(1, 2, 3).map(func).cache().map(lambda x: x).to_list()
        self.assertEqual(len(calls), 3)
        self.assertEqual(result, [None, None, None])
        result = seq(1, 2, 3).map(lambda x: x).cache()
        self.assertEqual(repr(result._lineage), 'Lineage: sequence -> map(<lambda>) -> cache')
        result = seq(1, 2, 3).map(lambda x: x).cache(delete_lineage=True)
        self.assertEqual(repr(result._lineage), 'Lineage: sequence')
