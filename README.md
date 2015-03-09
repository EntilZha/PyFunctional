# Motivation
Having programmed functionally in Scala and now using Python I missed the sytax/style for it from Scala. Most of that can be summed up by comparing the Scala style vs Python style for taking a list, filtering on a criteria, mapping a function to it, then reducing it. Below is a comparison of the default Python style and the Scala inspired style that ScalaFunctional uses.

```python
l = [1, 2, -1, -2]
f = lambda x: x > 0
g = lambda x: x * 2
q = lambda x, y: 2 * x + y

# Python style
reduce(q, map(g, filter(f, l)))

# Python list comprehension
reduce(q, [g(x) for x in l if f(x)])

# ScalaFunction style
from functional import seq
seq(l).filter(f).map(g).reduce(q)
```

# Usage
To use ScalaFunctional, you need only include: `from functional.chain import seq`. `seq` is a function which takes as argument a list and returns a wrapper on that list that provides the extensions for functional programming using Scala style. It also provides some common functions which Python doesn't provide such as group by and flat map. The wrapper class `FunctionalSequence` also implements operations such as equals, repr, not equal, str etc by deferring them to the passed sequence.

#Examples
```python
seq([1, 2, 3, 0, -1, -2, 3]).filter(lambda x: x > 0).filter(lambda x: x < 2)
# -> [1]
seq([1, 1, 2, -2, 5]).distinct()
# -> [1, 2, -5, -2]
seq([[1, 1], [2, 3], [5, -1]]).flat_map(lambda x: x).sum()
# -> 11
seq([("a", 1), ("b", 2), ("c", 3), ("a", 2), ("c", 5)]).group_by(lambda x: x[0])
# -> {'a': [('a', 1), ('a', 2)], 'c': [('c', 3), ('c', 5)], 'b': [('b', 2)]}
p1, p2 = seq([1, 2, 3, -1, -2, -3]).partition(lambda x: x > 0)
p1.reduce(lambda x, y: x * y)
# -> 6
p2.reduce(lambda x, y: x + y)
# -> -6
seq([2, 1, 3]).sorted()
# -> [1, 2, 3]
seq([1, 2, 3])[0]
# -> 1
seq([1, 2, 3])[-1]
# -> 3
```

## List of supported functions
* head, first: get first element
* last, tail: get last element
* drop: drop first n elements
* take: take first n elements
* map: map f onto sequence
* filter: filter sequence by f
* filter_not: filter sequence by not f
* reduce: reduce sequence using f
* count, len, size: get count of sequence
* reverse: reverse sequence
* distinct: return set of unique/distinct elements
* any
* all
* enumerate, zip_with_index
* max
* min
* find
* flat_map
* group_by
* empty
* non_empty
* string: similar to mkString
* partition
* slice
* sum
* set
* zip
* sorted
* list

## Future work
* Add more robust support and testing for dictionary manipulation
* Add concurrent version of `FunctionalSequence`, probably using `ThreadPoolExecutor` from Python futures.
* Continue to find bugs and make improvements in general
* Continue adding tests

## Contributing and Bug Fixes
This project is very, very new, so any feedback would be great!
