import unittest
from collections import namedtuple

from functional.chain import seq, FunctionalSequence, _wrap


class TestChain(unittest.TestCase):
    def assertType(self, s):
        self.assertTrue(isinstance(s, FunctionalSequence))

    def assertNotType(self, s):
        self.assertFalse(isinstance(s, FunctionalSequence))

    def test_constructor(self):
        self.assertRaises(TypeError, lambda: FunctionalSequence(1))

    def test_base_sequence(self):
        l = []
        self.assertType(seq(l))
        self.assertNotType(seq(l).sequence)
        self.assertType(seq(seq(l)))
        self.assertNotType(seq(seq(l)).sequence)
        l = seq([])
        l.sequence = seq([])
        self.assertNotType(l._get_base_sequence())

    def test_get_attr(self):
        CustomTuple = namedtuple("CustomTuple", 'x y')
        t = CustomTuple(1, 2)
        s = seq(t)
        self.assertType(s)
        self.assertEqual(s.sum(), 3)
        self.assertEqual(s.x, 1)
        self.assertEqual(s.y, 2)

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

    def test_hash(self):
        self.assertRaises(TypeError, lambda: hash(seq([1])))
        t = (1, 2)
        self.assertEqual(hash(t), hash(seq(t)))

    def test_len(self):
        l = [1, 2, 3]
        s = seq(l)
        self.assertEqual(len(l), len(s))
        self.assertEqual(len(l), s.size())
        self.assertEqual(len(l), s.len())

    def test_count(self):
        l = seq([-1, -1, 1, 1, 1])
        self.assertEqual(l.count(lambda x: x > 0), 3)
        self.assertEqual(l.count(lambda x: x < 0), 2)

    def test_getitem(self):
        l = [1, 2, [3, 4, 5]]
        s = seq(l)
        self.assertEqual(s[1], 2)
        self.assertEqual(s[2], [3, 4, 5])
        self.assertType(s[2])
        self.assertEqual(s[1:], [2, [3, 4, 5]])
        self.assertTrue(s[1:])
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
        self.assertType(l)

    def test_contains(self):
        string = "abcdef"
        s = seq(string)
        self.assertTrue("c" in s)

    def test_add(self):
        l0 = seq([1, 2, 3])
        l1 = seq([4, 5, 6])
        l2 = [4, 5, 6]
        expect = [1, 2, 3, 4, 5, 6]
        self.assertEqual(l0 + l1, expect)
        self.assertEqual(l0 + l2, expect)

    def test_head(self):
        l = seq([1, 2, 3])
        self.assertEqual(l.head(), 1)
        l = seq([[1, 2], 3, 4])
        self.assertEqual(l.head(), [1, 2])
        self.assertType(l.head())
        l = seq([])
        with self.assertRaises(IndexError):
            l.head()

    def test_first(self):
        l = seq([1, 2, 3])
        self.assertEqual(l.first(), 1)
        l = seq([[1, 2], 3, 4])
        self.assertEqual(l.first(), [1, 2])
        self.assertType(l.first())
        l = seq([])
        with self.assertRaises(IndexError):
            l.head()

    def test_head_option(self):
        l = seq([1, 2, 3])
        self.assertEqual(l.head_option(), 1)
        l = seq([[1, 2], 3, 4])
        self.assertEqual(l.head_option(), [1, 2])
        self.assertType(l.head_option())
        l = seq([])
        self.assertIsNone(l.head_option())

    def test_last(self):
        l = seq([1, 2, 3])
        self.assertEqual(l.last(), 3)
        l = seq([1, 2, [3, 4]])
        self.assertEqual(l.last(), [3, 4])
        self.assertType(l.last())

    def test_last_option(self):
        l = seq([1, 2, 3])
        self.assertEqual(l.last_option(), 3)
        l = seq([1, 2, [3, 4]])
        self.assertEqual(l.last_option(), [3, 4])
        self.assertType(l.last_option())
        l = seq([])
        self.assertIsNone(l.last_option())

    def test_init(self):
        l = seq([1, 2, 3, 4])
        expect = [1, 2, 3]
        self.assertSequenceEqual(l.init(), expect)

    def test_tail(self):
        l = seq([1, 2, 3, 4])
        expect = [2, 3, 4]
        self.assertSequenceEqual(l.tail(), expect)

    def test_inits(self):
        l = seq([1, 2, 3])
        expect = [[1, 2, 3], [1, 2], [1], []]
        self.assertSequenceEqual(l.inits(), expect)
        self.assertEqual(l.inits().map(lambda s: s.sum()), [6, 3, 1, 0])

    def test_tails(self):
        l = seq([1, 2, 3])
        expect = [[1, 2, 3], [2, 3], [3], []]
        self.assertSequenceEqual(l.tails(), expect)
        self.assertEqual(l.tails().map(lambda s: s.sum()), [6, 5, 3, 0])

    def test_drop(self):
        l = [1, 2, 3, 4, 5, 6]
        expect = [5, 6]
        result = seq(l).drop(4)
        self.assertEqual(result, expect)
        self.assertTrue(result)

    def test_drop_right(self):
        s = seq([1, 2, 3, 4])
        expect = [1, 2]
        result = s.drop_right(2)
        self.assertType(result)
        self.assertEqual(result, expect)

    def test_drop_while(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8]
        f = lambda x: x < 4
        expect = [4, 5, 6, 7, 8]
        result = seq(l).drop_while(f)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_take(self):
        l = [1, 2, 3, 4, 5, 6]
        expect = [1, 2, 3, 4]
        result = seq(l).take(4)
        self.assertEqual(result, expect)
        self.assertTrue(result)

    def test_take_while(self):
        l = [1, 2, 3, 4, 5, 6, 7, 8]
        f = lambda x: x < 4
        expect = [1, 2, 3]
        result = seq(l).take_while(f)
        self.assertEqual(expect, result)
        self.assertType(result)

    def test_union(self):
        result = seq([1, 1, 2, 3, 3]).union([1, 4, 5])
        expect = [1, 2, 3, 4, 5]
        self.assertType(result)
        self.assertEqual(result, expect)

    def test_intersection(self):
        result = seq([1, 2, 2, 3]).intersection([2, 3, 4, 5])
        expect = [2, 3]
        self.assertType(result)
        self.assertEqual(result, expect)

    def test_difference(self):
        result = seq([1, 2, 3]).difference([2, 3, 4])
        expect = [1]
        self.assertType(result)
        self.assertEqual(result, expect)

    def test_symmetric_difference(self):
        result = seq([1, 2, 3, 3]).symmetric_difference([2, 4, 5])
        expect = [1, 3, 4, 5]
        self.assertType(result)
        self.assertEqual(result, expect)

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

    def test_fold_left(self):
        f = lambda x, y: y + x
        l = seq([1, 2, 3, 4])
        self.assertEqual(l.fold_left(0, f), 10)
        self.assertEqual(l.fold_left(-10, f), 0)
        l = seq(['a', 'b', 'c'])
        self.assertEqual(l.fold_left("", f), "abc")
        self.assertEqual(l.fold_left("z", f), "zabc")
        f = lambda x, y: y + [x]
        self.assertEqual(l.fold_left([], f), ['a', 'b', 'c'])
        self.assertEqual(l.fold_left(['z'], f), ['z', 'a', 'b', 'c'])

    def test_fold_right(self):
        f = lambda x, y: y + x
        l = seq([1, 2, 3, 4])
        self.assertEqual(l.fold_right(0, f), 10)
        self.assertEqual(l.fold_right(-10, f), 0)
        l = seq(['a', 'b', 'c'])
        self.assertEqual(l.fold_right("", f), "cba")
        self.assertEqual(l.fold_right("z", f), "zcba")
        f = lambda x, y: y + [x]
        self.assertEqual(l.fold_right([], f), ['c', 'b', 'a'])

    def test_sorted(self):
        s = seq([1, 3, 2, 5, 4])
        r = s.sorted()
        self.assertEqual([1, 2, 3, 4, 5], r)
        self.assertType(r)

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

    def test_slice(self):
        s = seq([1, 2, 3, 4])
        result = s.slice(1, 2)
        self.assertEqual(result, [2])
        self.assertType(result)
        result = s.slice(1, 3)
        self.assertEqual(result, [2, 3])
        self.assertType(result)

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
        self.assertEqual(result, e)
        self.assertType(result)

    def test_join(self):
        l0 = [('a', 1), ('b', 2), ('c', 3)]
        l1 = [('a', 2), ('c', 4), ('d', 5)]
        result = seq(l0).join(l1)
        e = [('a', (1, 2)), ('c', (3, 4))]
        self.assertType(result)
        self.assertSequenceEqual(dict(result), dict(e))
        result = seq(l0).join(seq(l1))
        self.assertType(result)
        self.assertSequenceEqual(dict(result), dict(e))

    def test_left_join(self):
        left = [('a', 1), ('b', 2)]
        right = [('a', 2), ('c', 3)]
        result = seq(left).left_join(right)
        expect = [('a', (1, 2)), ('b', (2, None))]
        self.assertType(result)
        self.assertEqual(dict(result), dict(expect))
        result = seq(left).left_join(seq(right))
        self.assertType(result)
        self.assertEqual(dict(result), dict(expect))

    def test_right_join(self):
        left = [('a', 1), ('b', 2)]
        right = [('a', 2), ('c', 3)]
        result = seq(left).right_join(right)
        expect = [('a', (1, 2)), ('c', (None, 3))]
        self.assertType(result)
        self.assertEqual(dict(result), dict(expect))
        result = seq(left).right_join(seq(right))
        self.assertType(result)
        self.assertEqual(dict(result), dict(expect))

    def test_outer_join(self):
        left = [('a', 1), ('b', 2)]
        right = [('a', 2), ('c', 3)]
        result = seq(left).outer_join(right)
        expect = [('a', (1, 2)), ('b', (2, None)), ('c', (None, 3))]
        self.assertType(result)
        self.assertEqual(dict(result), dict(expect))
        result = seq(left).outer_join(seq(right))
        self.assertType(result)
        self.assertEqual(dict(result), dict(expect))

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
        self.assertEqual(expect, result)
        self.assertType(result)

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
        result_comparison = {}
        for kv in result:
            result_comparison[kv[0]] = kv[1]
        self.assertEqual(expect, result_comparison)
        self.assertType(result)

    def test_group_by_key(self):
        l = [('a', 1), ('a', 2), ('a', 3), ('b', -1), ('b', 1), ('c', 10), ('c', 5)]
        e = {"a": [1, 2, 3], "b": [-1, 1], "c": [10, 5]}.items()
        result = seq(l).group_by_key()
        self.assertEqual(len(result), len(e))
        for e0, e1 in zip(result, e):
            self.assertEqual(e0, e1)
        self.assertType(result)

    def test_grouped(self):
        l = seq([1, 2, 3, 4, 5, 6, 7, 8])
        expect = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.assertEqual(l.grouped(2), expect)
        expect = [[1, 2, 3], [4, 5, 6], [7, 8]]
        self.assertEqual(l.grouped(3), expect)

    def test_empty(self):
        self.assertTrue(seq([]).empty())

    def test_non_empty(self):
        self.assertTrue(seq([1]).non_empty())

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
        self.assertEqual(e1, p1)
        self.assertEqual(e2, p2)
        self.assertType(p1)
        self.assertType(p2)

    def test_product(self):
        l = [2, 2, 3]
        self.assertEqual(12, seq(l).product())

    def test_sum(self):
        l = [1, 2, 3]
        self.assertEqual(6, seq(l).sum())

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

    def test_to_list(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        self.assertEqual(seq(l).to_list(), l)

    def test_list(self):
        l = [1, 2, 3, "abc", {1: 2}, {1, 2, 3}]
        self.assertEqual(seq(l).list(), l)

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
        self.assertEqual(seq(l).to_dict(), d)

    def test_dict(self):
        l = [(1, 2), (2, 10), (7, 2)]
        d = {1: 2, 2: 10, 7: 2}
        self.assertEqual(seq(l).dict(), d)

    def test_reduce_by_key(self):
        l = [('a', 1), ('a', 2), ('a', 3), ('b', -1), ('b', 1), ('c', 10), ('c', 5)]
        e = {"a": 6, "b": 0, "c": 15}.items()
        result = seq(l).reduce_by_key(lambda x, y: x + y)
        self.assertEqual(len(result), len(e))
        for e0, e1 in zip(result, e):
            self.assertEqual(e0, e1)
        self.assertType(result)

    def test_wrap(self):
        self.assertType(_wrap([1, 2]))
        self.assertType(_wrap((1, 2)))
        self.assertNotType(_wrap(1))
        self.assertNotType(_wrap(1.0))
        self.assertNotType("test")
        self.assertNotType(True)

    def test_wrap_objects(self):
        class A(object):
            a = 1
        l = [A(), A(), A()]
        self.assertIsInstance(_wrap(A()), A)
        self.assertType(seq(l))
