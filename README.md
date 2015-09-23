# ScalaFunctional
[<img src="https://travis-ci.org/EntilZha/ScalaFunctional.svg?branch=master"/>](https://travis-ci.org/EntilZha/ScalaFunctional)
[![Coverage Status](https://coveralls.io/repos/EntilZha/ScalaFunctional/badge.svg?branch=master)](https://coveralls.io/r/EntilZha/ScalaFunctional?branch=master)
[![ReadTheDocs](https://readthedocs.org/projects/scalafunctional/badge/?version=latest)](https://readthedocs.org/projects/scalafunctional/)
[![Latest Version](https://badge.fury.io/py/scalafunctional.svg)](https://pypi.python.org/pypi/scalafunctional/)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/EntilZha/ScalaFunctional?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# Motivation
[Blog post about ScalaFunctional](http://entilzha.github.io/blog/2015/03/14/functional-programming-collections-python/)

`ScalaFunctional` exists to make functional programming with collections easy and intuitive in Python. It borrows the collections/RDD APIs from Scala and Apache Spark to provide access to a rich and declarative way of defining data pipelines.

To demonstrate the different style of Python map/filter/reduce, list comprehensions, and `ScalaFunctional`, the code block below does the same thing in all three: manipulate a list of numbers to compute a result. There is a bonus
at the end using the `_` operator from [`fn.py`](https://github.com/kachayev/fn.py) (note: `ipython`'s underscore clashes with this, so its not helpful in interactive `ipython` sessions).

```python
l = [1, 2, -1, -2]

# Python style
reduce(lambda x, y: x * y, map(lambda x: 2 * x, filter(lambda x: x > 0, l)))

# Python list comprehension
reduce(lambda x, y: x * y, [2 * x for x in l if x > 0])

# ScalaFunctional style
from functional import seq
from fn import _
seq(l).filter(lambda x: x > 0).map(lambda x: 2 * x).reduce(lambda x, y: x * y)
seq(l).filter(_ > 0).map(2 * _).reduce(_ * _)
```

Although a trivial example, the real power of `ScalaFunctional` is composing transformations not available natively in Python. For example, the very common word count example is easy:
```python
# ScalaFunctional word count
l = seq("the why the what of word counting of english".split(" "))
l.map(lambda word: (word, 1)).reduce_by_key(lambda x, y: x + y)
# [('what', 1), ('word', 1), ('of', 2), ('english', 1), ('the', 2), ('counting', 1), ('why', 1)]
```

# Installation and Usage
## Installation
```bash
# Install from command line
$ pip install scalafunctional
```
## Usage
To use ScalaFunctional, you need only include: `from functional import seq`. `seq` is a function which takes as argument a list and returns a wrapper on that list that provides access to the functions in the API table below. The API consists of a combination of common Python APIs, the Scala Collections API, and the Apache Spark RDD API.

For detailed documentation and more usage examples, refer to [readthedocs](http://scalafunctional.readthedocs.org/en/latest/functional.html#module-functional.pipeline)

# Scala Functional API
## Transformations and Actions
Fundamentally, there are two types of functions in data pipelines: transformations and actions. After reading/creating data, functions such as `map`, `filter`, `flat_map` and much more do not need to be evaluated immediately. They can be lazily evaluated and only evaluated when the result is required. This helps to boost performance for more intensive work.

The second type of function is an action. These force `functional` to compute the result of a sequence of transformations. Functions such as `reduce`, `to_list`, `to_dict`, `str`, and `repr` are common examples.

For example, in the code `seq(1, 2, 3).map(lambda x: x * 2).reduce(lambda x, y: x + y)`, `map` is a transformation and `reduce` is an action.

## ScalaFunctional API
Function | Description | Type
 ------- | -----------  | ----
 `map(func)` | Maps `func` onto elements of sequence | transformation
 `filter(func)` | Filters elements of sequence to only those where `func(element)` is `True` | transformation
 `filter_not(func)` | Filters elements of sequence to only those where `func(element)` is `False` | transformation
 `flatten()` | Flattens sequence of lists to a single sequence | transformation
 `flat_map(func)` | `func` must return an iterable. Maps `func` to each element, then merges the result to one flat sequence | transformation
 `group_by(func)` | Groups sequence into `(key, value)` pairs where `key=func(element)` and `value` is from the original sequence | transformation
 `group_by_key()` | Groups sequence of `(key, value)` pairs by `key` | transformation
 `reduce_by_key(func)` | Reduces list of `(key, value)` pairs using `func` | transformation
 `union(other)` | Union of unique elements in sequence and `other` | transformation
 `intersection(other)` | Intersection of unique elements in sequence and `other` | transformation
 `difference(other)` | New sequence with unique elements present in sequence but not in `other` | transformation
 `symmetric_difference(other)` | New sequence with unique elements present in sequnce or `other`, but not both | transformation
 `distinct()` | Returns distinct elements of sequence. Elements must be hashable | transformation
 `distinct_by(func)` | Returns distinct elements of sequence using `func` as a key | transformation
 `drop(n)` | Drop the first `n` elements of the sequence | transformation
 `drop_right(n)` | Drop the last `n` elements of the sequence | transformation
 `drop_while(func)` | Drop elements while `func` evaluates to `True`, then returns the rest | transformation
 `take(n)` | Returns sequence of first `n` elements | transformation
 `take_while(func)` | Take elements while `func` evaluates to `True`, then drops the rest | transformation
  `init()` | Returns sequence without the last element | transformation
  `tail()` | Returns sequence without the first element | transformation
  `inits()` | Returns consecutive inits of sequence | transformation
  `tails()` | Returns consecutive tails of sequence | transformation
  `zip(other)` | Zips the sequence with `other` | transformation
  `zip_with_index()` | Zips the sequence with the index starting at zero on the left side | transformation
  `enumerate(start=0)` | Zips the sequence with the index starting at `start` on the left side | transformation
  `inner_join(other)` | Returns inner join of sequence with other. Must be a sequence of `(key, value)` pairs | transformation
  `outer_join(other)` | Returns outer join of sequence with other. Must be a sequence of `(key, value)` pairs | transformation
  `left_join(other)` | Returns left join of sequence with other. Must be a sequence of `(key, value)` pairs | transformation
  `right_join(other)` | Returns right join of sequence with other. Must be a sequence of `(key, value)` pairs | transformation
  `join(other, join_type='inner')` | Returns join of sequence with other as specified by `join_type`. Must be a sequence of `(key, value)` pairs | transformation
  `partition(func)` | Partitions the sequence into elements which satisfy `func(element)` and those that don't | transformation
  `grouped(size)` | Partitions the elements into groups of size `size` | transformation
  `sorted(key=None, reverse=False)` | Returns elements sorted according to python `sorted` | transformation
  `reverse()` | Returns the reversed sequence | transformation
  `slice(start, until)` | Sequence starting at `start` and including elements up to `until` | transformation
 `head()` / `first()` | Returns first element in sequence | action
 `head_option()` | Returns first element in sequence or `None` if its empty | action
 `last()` | Returns last element in sequence | action
 `last_option()` | Returns last element in sequence or `None` if its empty | action
 `len()` / `size()` | Returns length of sequence | action
 `count(func)` | Returns count of elements in sequence where `func(element)` is True | action
 `empty()` | Returns `True` if the sequence has zero length | action
 `non_empty()` | Returns `True` if sequence has non-zero length | action
`all()` | Returns `True` if all elements in sequence are truthy | action
`exists(func)` | Returns `True` if `func(element)` for any element in the sequence is `True` | action
`for_all(func)` | Returns `True` if `func(element)` is `True` for all elements in the sequence | action
`find(func)` | Returns the element that first evaluates `func(element)` to `True` | action
`any()` | Returns `True` if any element in sequence is truthy | action
`max()` | Returns maximal element in sequence | action
`min()` | Returns minimal element in sequence | action
`max_by(func)` | Returns element with maximal value `func(element)` | action
`min_by(func)` | Returns element with minimal value `func(element)` | action
`sum()` | Returns the sum of elements | action
`product()` | Returns the product of elements | action
`fold_left(zero_value, func)` | Reduces element from left to right using `func` and initial value `zero_value` | action
`fold_right(zero_value, func)` | Reduces element from right to left using `func` and initial value `zero_value` | action
`make_string(separator)` | Returns string with `separator` between each `str(element)` | action
`dict(default=None)` / `to_dict(default=None)` | Converts a sequence of `(Key, Value)` pairs to a `dictionary`. If `default` is not None, it must be a value or zero argument callable which will be used to create a `collections.defaultdict` | action
`list()` / `to_list()` | Converts sequence to a list | action
`set() / to_set()` | Converts sequence to a set | action
`cache()` | Forces evaluation of sequence immediately and caches the result | action
`for_each(func)` | Executes `func` on each element of the sequence | action

## Road Map
### Version `0.4.0`
Implement new ways to have `ScalaFunctional` ingest data and write it out. Principally this means implementing reading from files (csv, json, etc) and writing back to them natively. This is being implemented in `functional.streams` if you would like to check progress

### Past next release
* Parallel execution engine for users wanting to run operations in parallel
* Decide if package is stable enough to prepare a `1.0` release

## Contributing and Bug Fixes
Any contributions or bug reports are welcome. Thus far, there is a 100% acceptance rate for pull requests and contributors have offered valuable feedback and critique on code. It is also great to hear from users of the package, especially what it is used for, what works well, and what could be improved.
