<div style="display:inline-block;">
<div><b><a href="https://travis-ci.org/EntilZha/ScalaFunctional">Travis CI Build</a></b></div>
<div>
<a href="https://travis-ci.org/EntilZha/ScalaFunctional"><img src="https://travis-ci.org/EntilZha/ScalaFunctional.svg?branch=master"/></a>
</div>
</div>
<div style="display:inline-block;>
<div><b><a href="https://readthedocs.org/projects/scalafunctional/">ReadTheDocs Build</a></b></div>
<div>
<a href="https://readthedocs.org/projects/scalafunctional/"><img src="https://readthedocs.org/projects/scalafunctional/badge/?version=latest">
</a>
</div>
</div>

# Motivation
Having programmed functionally in Scala and now using Python I missed the syntax/style for it from Scala. Most of that can be summed up by comparing the Scala style vs Python style for taking a list, filtering on a criteria, mapping a function to it, then reducing it. Below is a comparison of the default Python style and the Scala inspired style that ScalaFunctional uses.

```python
l = [1, 2, -1, -2]
f = lambda x: x > 0
g = lambda x: x * 2
q = lambda x, y: 2 * x + y

# Python style
reduce(q, map(g, filter(f, l)))

# Python list comprehension
reduce(q, [g(x) for x in l if f(x)])

# ScalaFunctional style
from functional import seq
seq(l).filter(f).map(g).reduce(q)

# ScalaFunctional word count
l = seq("the why the what of word counting of english".split(" "))
l.map(lambda word: (word, 1)).reduce_by_key(lambda x, y: x + y)
# [('what', 1), ('word', 1), ('of', 2), ('english', 1), ('the', 2), ('counting', 1), ('why', 1)]
```

# Inspiration
Almost all inspiration is either from [scala docs](http://www.scala-lang.org/api/current/#scala.Array) or [spark docs](https://spark.apache.org/docs/latest/programming-guide.html#transformations)

# Usage
To use ScalaFunctional, you need only include: `from functional import seq`. `seq` is a function which takes as argument a list and returns a wrapper on that list that provides the extensions for functional programming using Scala style. It also provides some common functions which Python doesn't provide such as group by and flat map. The wrapper class `FunctionalSequence` also implements operations such as equals, repr, not equal, str etc by deferring them to the passed sequence.

# Installation
For common use, you can install via the command line and pip: `pip install scalafunctional`. Then in your code import it via 
`from functional import seq`. For developers, clone the repo then run `python setup.py develop`.

# Examples
## Number twiddling
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
### List to List
* tail: get everything except first element
* drop: drop first n elements
* drop_while: drop first elements using f
* take: take first n elements
* take_while: take first elements using f
* map: map f onto sequence
* filter: filter sequence by f
* filter_not: filter sequence by not f
* reverse: reverse sequence
* distinct: return set of unique/distinct elements
* flatten
* flat_map
* group_by
* enumerate, zip_with_index
* partition
* slice
* zip
* sorted

### List of (Key, Value) to List
* reduce_by_key
* group_by_key

### List to Value
* head, first: get first element
* head_option: get first element or None
* last: get last element
* reduce: reduce sequence using f
* fold_left
* fold_right
* count, len, size: get count of sequence
* any
* all, for_all
* max, max_by
* min, min_by
* find
* empty
* non_empty
* string: similar to mkString
* sum

### Conversion to other types
* set
* list
* to_dict

## Future work
* Continue to find bugs and fix bugs
* Continue adding features from either scala collections or spark (or other methods in the same spirit)
* Continue adding tests

## Contributing and Bug Fixes
This project is very, very new, so any feedback would be great!
