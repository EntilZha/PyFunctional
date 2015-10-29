# ScalaFunctional
[<img src="https://travis-ci.org/EntilZha/ScalaFunctional.svg?branch=master"/>](https://travis-ci.org/EntilZha/ScalaFunctional)
[![Coverage Status](https://coveralls.io/repos/EntilZha/ScalaFunctional/badge.svg?branch=master&service=github)](https://coveralls.io/r/EntilZha/ScalaFunctional?branch=master)
[![ReadTheDocs](https://readthedocs.org/projects/scalafunctional/badge/?version=latest)](https://readthedocs.org/projects/scalafunctional/)
[![Latest Version](https://badge.fury.io/py/scalafunctional.svg)](https://pypi.python.org/pypi/scalafunctional/)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/EntilZha/ScalaFunctional?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Usage
`ScalaFunctional` exists to make functional programming with collections easy and intuitive in Python.
It implements the Scala collections and Apache Spark RDD APIs in Python to provide access to a rich and
declarative way of defining data pipelines.

Below is a comparison of using Python builtins, list comprehensions, and `ScalaFunctional`

```python
l = [1, 2, -1, -2]

# Python style
reduce(lambda x, y: x * y, map(lambda x: 2 * x, filter(lambda x: x > 0, l)))

# Python list comprehension
reduce(lambda x, y: x * y, [2 * x for x in l if x > 0])

# ScalaFunctional style
from functional import seq
seq(l).filter(lambda x: x > 0).map(lambda x: 2 * x).reduce(lambda x, y: x * y)
```

`ScalaFunctional` also makes other tasks much easier than using only python's built in utilities. Below are examples of several tasks such as word count and merging data from two streams.

```python
# ScalaFunctional word count
l = seq("I dont want to believe I want to know".split(" "))
l.map(lambda word: (word, 1)).reduce_by_key(lambda x, y: x + y)
# [('dont', 1), ('I', 2), ('to', 2), ('know', 1), ('want', 2), ('believe', 1)]

# List of key value pairs, where the key is an ID and we want values joined
names = [(1, 'spark'), (2, 'hadoop'), (3, 'django')]
languages = [(1, 'scala'), (2, 'java'), (3, 'python')]
joined = seq(names).inner_join(languages).to_dict()
# {1: ('spark', 'scala'), 2: ('hadoop', 'java'), 3: ('django', 'python')}
```

[More examples and motivation.](http://entilzha.github.io/blog/2015/03/14/functional-programming-collections-python/)

## Installation
`ScalaFunctional` is available on [pypi](https://pypi.python.org/pypi/ScalaFunctional) and can be installed by running:
```bash
# Install from command line
$ pip install scalafunctional
```

Or by
```bash
git clone git@github.com:EntilZha/ScalaFunctional.git
python setup.py install
```

Then import the package using: `from functional import seq`

## Documentation
Full documentation can be found at [scalafunctional.readthedocs.org](http://scalafunctional.readthedocs.org/en/latest/functional.html#module-functional.pipeline).

### Summary of Streams, Transformations and Actions
`ScalaFunctional` has three types of functions:

1. Streams read data for use by the collections API. In `0.3.1` the only stream function is `seq`, however in `0.4.0` this is getting expanded to read data from text, csv, json, and jsonl files.
2. Transformations: These mutate data from streams with functions such as `map`, `flat_map`, and `filter`
3. Actions: These cause a series of transformations to evaluate to a concrete value. For example, `to_list`, `reduce`, and `to_dict` are examples of actions.

To summarize, suppose we have: `seq(1, 2, 3).map(lambda x: x * 2).reduce(lambda x, y: x + y)`, `seq` is the stream, `map` is the transformation, and  `reduce` is the action.

### Streams (`seq`) API
The primary entrypoint to using `ScalaFunctional` is through `functional.seq`. `seq` can take any iterable as input and returns a `functional.Sequence` which exposes the collections API described in the table below. `seq` can be called in various ways demonstrated below:

```python
# Passing a list
seq([1, 1, 2, 3]).to_set()
# [1, 2, 3]

# Passing direct arguments
seq(1, 1, 2, 3).map(lambda x: x).to_list()
# [1, 1, 2, 3]

# Passing a single value
seq(1).map(lambda x: -x).to_list()
# [-1]
```

### Collections (transformations and actions) API
Below is the complete list of functions which can be called on the object created by `seq` otherwise known as a `functional.Sequence`.

Function | Description | Type
 ------- | -----------  | ----
`map(func)/select(func)` | Maps `func` onto elements of sequence | transformation
`filter(func)/where(func)` | Filters elements of sequence to only those where `func(element)` is `True` | transformation
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
`sorted(key=None, reverse=False)/order_by(func)` | Returns elements sorted according to python `sorted` | transformation
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
`aggregate(func)/aggregate(seed, func)/aggregate(seed, func, result_map)` | Aggregate using `func` starting with `seed` or first element of list then apply `result_map` to the result | action
`fold_left(zero_value, func)` | Reduces element from left to right using `func` and initial value `zero_value` | action
`fold_right(zero_value, func)` | Reduces element from right to left using `func` and initial value `zero_value` | action
`make_string(separator)` | Returns string with `separator` between each `str(element)` | action
`dict(default=None)` / `to_dict(default=None)` | Converts a sequence of `(Key, Value)` pairs to a `dictionary`. If `default` is not None, it must be a value or zero argument callable which will be used to create a `collections.defaultdict` | action
`list()` / `to_list()` | Converts sequence to a list | action
`set() / to_set()` | Converts sequence to a set | action
`cache()` | Forces evaluation of sequence immediately and caches the result | action
`for_each(func)` | Executes `func` on each element of the sequence | action

### Tips
Another python package named `fn` is also helpful. It can be installed via `pip install fn` and can remove the need for direct `lambda`s.

```python
from functional import seq
from fn import _

seq(1, 2, 3).map(_ * 2).reduce(_ + _)
# 12
```

## Road Map
### Version `0.4.0`
Implement new ways to have `ScalaFunctional` ingest data and write it out. Principally this means implementing reading from files (csv, json, etc) and writing back to them natively. This is being implemented in `functional.streams` if you would like to check progress

### Past next release
* Parallel execution engine for users wanting to run operations in parallel
* Decide if package is stable enough to prepare a `1.0` release

## Contributing and Bug Fixes
Any contributions or bug reports are welcome. Thus far, there is a 100% acceptance rate for pull requests and contributors have offered valuable feedback and critique on code. It is great to hear from users of the package, especially what it is used for, what works well, and what could be improved.
