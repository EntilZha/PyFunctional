# PyFunctional
![Build Status](https://github.com/EntilZha/PyFunctional/workflows/Python%20package/badge.svg)
[![Code Coverage](https://codecov.io/gh/EntilZha/PyFunctional/branch/master/graph/badge.svg)](https://codecov.io/gh/EntilZha/PyFunctional)
[![ReadTheDocs](https://readthedocs.org/projects/scalafunctional/badge/?version=latest)](http://docs.pyfunctional.org)
[![PyPI version](https://badge.fury.io/py/PyFunctional.svg)](https://badge.fury.io/py/PyFunctional)

## Features
`PyFunctional` makes creating data pipelines easy by using chained functional operators. Here are a
few examples of what it can do:

* Chained operators: `seq(1, 2, 3).map(lambda x: x * 2).reduce(lambda x, y: x + y)`
* Expressive and feature complete API
* Read and write `text`, `csv`, `json`, `jsonl`, `sqlite`, `gzip`, `bz2`, and `lzma/xz` files
* Parallelize "embarrassingly parallel" operations like `map` easily
* Complete documentation, rigorous unit test suite, 100% test coverage, and CI which provide
robustness

`PyFunctional`'s API takes inspiration from Scala collections, Apache Spark RDDs, and Microsoft
LINQ.

## Table of Contents

1. [Installation](#installation)
2. [Examples](#examples)
    1. [Simple Example](#simple-example)
    2. [Aggregates and Joins](#aggregates-and-joins)
    3. [Reading and Writing SQLite3](#readingwriting-sqlite3)
    4. [Data Interchange with Pandas](https://github.com/EntilZha/PyFunctional/blob/master/examples/PyFunctional-pandas-tutorial.ipynb)
3. [Writing to Files](#writing-to-files)
4. [Parallel Execution](#parallel-execution)
5. [Github Shortform Documentation](#documentation)
    1. [Streams, Transformations, and Actions](#streams-transformations-and-actions)
    2. [Streams API](#streams-api)
    3. [Transformations and Actions APIs](#transformations-and-actions-apis)
    4. [Lazy Execution](#lazy-execution)
6. [Contributing and Bug Fixes](#contributing-and-bug-fixes)
7. [Changelog](https://github.com/EntilZha/PyFunctional/blob/master/CHANGELOG.md)

## Installation
`PyFunctional` is available on [pypi](https://pypi.python.org/pypi/PyFunctional) and can be
installed by running:

```bash
# Install from command line
$ pip install pyfunctional
```

Then in python run: `from functional import seq`

## Examples
`PyFunctional` is useful for many tasks, and can natively open several common file types. Here
are a few examples of what you can do.

### Simple Example
```python
from functional import seq

seq(1, 2, 3, 4)\
    .map(lambda x: x * 2)\
    .filter(lambda x: x > 4)\
    .reduce(lambda x, y: x + y)
# 14

# or if you don't like backslash continuation
(seq(1, 2, 3, 4)
    .map(lambda x: x * 2)
    .filter(lambda x: x > 4)
    .reduce(lambda x, y: x + y)
)
# 14
```

### Streams, Transformations and Actions
`PyFunctional` has three types of functions:

1. Streams: read data for use by the collections API.
2. Transformations: transform data from streams with functions such as `map`, `flat_map`, and
`filter`
3. Actions: These cause a series of transformations to evaluate to a concrete value. `to_list`,
`reduce`, and `to_dict` are examples of actions.

In the expression `seq(1, 2, 3).map(lambda x: x * 2).reduce(lambda x, y: x + y)`, `seq` is the
stream, `map` is the transformation, and `reduce` is the action.

### Filtering a list of account transactions
```python
from functional import seq
from collections import namedtuple

Transaction = namedtuple('Transaction', 'reason amount')
transactions = [
    Transaction('github', 7),
    Transaction('food', 10),
    Transaction('coffee', 5),
    Transaction('digitalocean', 5),
    Transaction('food', 5),
    Transaction('riotgames', 25),
    Transaction('food', 10),
    Transaction('amazon', 200),
    Transaction('paycheck', -1000)
]

# Using the Scala/Spark inspired APIs
food_cost = seq(transactions)\
    .filter(lambda x: x.reason == 'food')\
    .map(lambda x: x.amount).sum()

# Using the LINQ inspired APIs
food_cost = seq(transactions)\
    .where(lambda x: x.reason == 'food')\
    .select(lambda x: x.amount).sum()

# Using PyFunctional with fn
from fn import _
food_cost = seq(transactions).filter(_.reason == 'food').map(_.amount).sum()
```

### Aggregates and Joins
The account transactions example could be done easily in pure python using list comprehensions. To
show some of the things `PyFunctional` excels at, take a look at a couple of word count examples.

```python
words = 'I dont want to believe I want to know'.split(' ')
seq(words).map(lambda word: (word, 1)).reduce_by_key(lambda x, y: x + y)
# [('dont', 1), ('I', 2), ('to', 2), ('know', 1), ('want', 2), ('believe', 1)]
```

In the next example we have chat logs formatted in [json lines (jsonl)](http://jsonlines.org/) which
contain messages and metadata. A typical jsonl file will have one valid json on each line of a file.
Below are a few lines out of `examples/chat_logs.jsonl`.

```json
{"message":"hello anyone there?","date":"10/09","user":"bob"}
{"message":"need some help with a program","date":"10/09","user":"bob"}
{"message":"sure thing. What do you need help with?","date":"10/09","user":"dave"}
```

```python
from operator import add
import re
messages = seq.jsonl('examples/chat_logs.jsonl')

# Split words on space and normalize before doing word count
def extract_words(message):
    return re.sub('[^0-9a-z ]+', '', message.lower()).split(' ')


word_counts = messages\
    .map(lambda log: extract_words(log['message']))\
    .flatten().map(lambda word: (word, 1))\
    .reduce_by_key(add).order_by(lambda x: x[1])

```

Next, lets continue that example but introduce a json database of users from `examples/users.json`.
In the previous example we showed how `PyFunctional` can do word counts, in the next example lets
show how `PyFunctional` can join different data sources.

```python
# First read the json file
users = seq.json('examples/users.json')
#[('sarah',{'date_created':'08/08','news_email':True,'email':'sarah@gmail.com'}),...]

email_domains = users.map(lambda u: u[1]['email'].split('@')[1]).distinct()
# ['yahoo.com', 'python.org', 'gmail.com']

# Join users with their messages
message_tuples = messages.group_by(lambda m: m['user'])
data = users.inner_join(message_tuples)
# [('sarah',
#    (
#      {'date_created':'08/08','news_email':True,'email':'sarah@gmail.com'},
#      [{'date':'10/10','message':'what is a...','user':'sarah'}...]
#    )
#  ),...]

# From here you can imagine doing more complex analysis
```

### CSV, Aggregate Functions, and Set functions
In `examples/camping_purchases.csv` there are a list of camping purchases. Lets do some cost
analysis and compare it the required camping gear list stored in `examples/gear_list.txt`.

```python
purchases = seq.csv('examples/camping_purchases.csv')
total_cost = purchases.select(lambda row: int(row[2])).sum()
# 1275

most_expensive_item = purchases.max_by(lambda row: int(row[2]))
# ['4', 'sleeping bag', ' 350']

purchased_list = purchases.select(lambda row: row[1])
gear_list = seq.open('examples/gear_list.txt').map(lambda row: row.strip())
missing_gear = gear_list.difference(purchased_list)
# ['water bottle','gas','toilet paper','lighter','spoons','sleeping pad',...]
```

In addition to the aggregate functions shown above (`sum` and `max_by`) there are many more.
Similarly, there are several more set like functions in addition to `difference`.

### Reading/Writing SQLite3
`PyFunctional` can read and write to SQLite3 database files. In the example below, users are read
 from `examples/users.db` which stores them as rows with columns `id:Int` and `name:String`.

```python
db_path = 'examples/users.db'
users = seq.sqlite3(db_path, 'select * from user').to_list()
# [(1, 'Tom'), (2, 'Jack'), (3, 'Jane'), (4, 'Stephan')]]

sorted_users = seq.sqlite3(db_path, 'select * from user order by name').to_list()
# [(2, 'Jack'), (3, 'Jane'), (4, 'Stephan'), (1, 'Tom')]
```

Writing to a SQLite3 database is similarly easy

```python
import sqlite3
from collections import namedtuple

with sqlite3.connect(':memory:') as conn:
    conn.execute('CREATE TABLE user (id INT, name TEXT)')
    conn.commit()
    User = namedtuple('User', 'id name')
    
    # Write using a specific query
    seq([(1, 'pedro'), (2, 'fritz')]).to_sqlite3(conn, 'INSERT INTO user (id, name) VALUES (?, ?)')
    
    # Write by inserting values positionally from a tuple/list into named table
    seq([(3, 'sam'), (4, 'stan')]).to_sqlite3(conn, 'user')
    
    # Write by inferring schema from namedtuple
    seq([User(name='tom', id=5), User(name='keiga', id=6)]).to_sqlite3(conn, 'user')
    
    # Write by inferring schema from dict
    seq([dict(name='david', id=7), dict(name='jordan', id=8)]).to_sqlite3(conn, 'user')
    
    # Read everything back to make sure it wrote correctly
    print(list(conn.execute('SELECT * FROM user')))
    
    # [(1, 'pedro'), (2, 'fritz'), (3, 'sam'), (4, 'stan'), (5, 'tom'), (6, 'keiga'), (7, 'david'), (8, 'jordan')]
```

## Writing to files
Just as `PyFunctional` can read from `csv`, `json`, `jsonl`, `sqlite3`, and text files, it can
also write them. For complete API documentation see the collections API table or the official docs.

### Compressed Files
`PyFunctional` will auto-detect files compressed with `gzip`, `lzma/xz`, and `bz2`. This is done
by examining the first several bytes of the file to determine if it is compressed so therefore
requires no code changes to work.

To write compressed files, every `to_` function has a parameter `compression` which can be set to
the default `None` for no compression, `gzip` or `gz` for gzip compression, `lzma` or `xz` for lzma
compression, and `bz2` for bz2 compression.

### Parallel Execution
The only change required to enable parallelism is to import `from functional import pseq` instead of
`from functional import seq` and use `pseq` where you would use `seq`. The following
operations are run in parallel with more to be implemented in a future release:

* `map`/`select`
* `filter`/`filter_not`/`where`
* `flat_map`

Parallelization uses python `multiprocessing` and squashes chains of embarrassingly parallel
operations to reduce overhead costs. For example, a sequence of maps and filters would be executed
all at once rather than in multiple loops using `multiprocessing`

## Documentation
Shortform documentation is below and full documentation is at
[docs.pyfunctional.org](http://docs.pyfunctional.org/en/latest/functional.html).

### Streams API
All of `PyFunctional` streams can be accessed through the `seq` object. The primary way to create
a stream is by calling `seq` with an iterable. The `seq` callable is smart and is able to accept
multiple types of parameters as shown in the examples below.

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

`seq` also provides entry to other streams as attribute functions as shown below.

```python
# number range
seq.range(10)

# text file
seq.open('filepath')

# json file
seq.json('filepath')

# jsonl file
seq.jsonl('filepath')

# csv file
seq.csv('filepath')
seq.csv_dict_reader('filepath')

# sqlite3 db and sql query
seq.sqlite3('filepath', 'select * from data')
```

For more information on the parameters that these functions can take, reference the
[streams documentation](http://docs.pyfunctional.org/en/latest/functional.html#module-functional.streams)

### Transformations and Actions APIs
Below is the complete list of functions which can be called on a stream object from `seq`. For
complete documentation reference
[transformation and actions API](http://docs.pyfunctional.org/en/latest/functional.html#module-functional.pipeline).

Function | Description | Type
 ------- | ----------- | ----
`map(func)/select(func)` | Maps `func` onto elements of sequence | transformation
`starmap(func)/smap(func)` | Apply `func` to sequence with `itertools.starmap` | transformation
`filter(func)/where(func)` | Filters elements of sequence to only those where `func(element)` is `True` | transformation
`filter_not(func)` | Filters elements of sequence to only those where `func(element)` is `False` | transformation
`flatten()` | Flattens sequence of lists to a single sequence | transformation
`flat_map(func)` | `func` must return an iterable. Maps `func` to each element, then merges the result to one flat sequence | transformation
`group_by(func)` | Groups sequence into `(key, value)` pairs where `key=func(element)` and `value` is from the original sequence | transformation
`group_by_key()` | Groups sequence of `(key, value)` pairs by `key` | transformation
`reduce_by_key(func)` | Reduces list of `(key, value)` pairs using `func` | transformation
`count_by_key()` | Counts occurrences of each `key` in list of `(key, value)` pairs | transformation
`count_by_value()` | Counts occurrence of each value in a list | transformation
`union(other)` | Union of unique elements in sequence and `other` | transformation
`intersection(other)` | Intersection of unique elements in sequence and `other` | transformation
`difference(other)` | New sequence with unique elements present in sequence but not in `other` | transformation
`symmetric_difference(other)` | New sequence with unique elements present in sequence or `other`, but not both | transformation
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
`zip_with_index(start=0)` | Zips the sequence with the index starting at `start` on the right side | transformation
`enumerate(start=0)` | Zips the sequence with the index starting at `start` on the left side | transformation
`cartesian(*iterables, repeat=1)` | Returns cartesian product from itertools.product | transformation
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
`sum()/sum(projection)` | Returns the sum of elements possibly using a projection | action
`product()/product(projection)` | Returns the product of elements possibly using a projection | action
`average()/average(projection)` | Returns the average of elements possibly using a projection | action
`aggregate(func)/aggregate(seed, func)/aggregate(seed, func, result_map)` | Aggregate using `func` starting with `seed` or first element of list then apply `result_map` to the result | action
`fold_left(zero_value, func)` | Reduces element from left to right using `func` and initial value `zero_value` | action
`fold_right(zero_value, func)` | Reduces element from right to left using `func` and initial value `zero_value` | action
`make_string(separator)` | Returns string with `separator` between each `str(element)` | action
`dict(default=None)` / `to_dict(default=None)` | Converts a sequence of `(Key, Value)` pairs to a `dictionary`. If `default` is not None, it must be a value or zero argument callable which will be used to create a `collections.defaultdict` | action
`list()` / `to_list()` | Converts sequence to a list | action
`set() / to_set()` | Converts sequence to a set | action
`to_file(path)` | Saves the sequence to a file at path with each element on a newline | action
`to_csv(path)` | Saves the sequence to a csv file at path with each element representing a row | action
`to_jsonl(path)` | Saves the sequence to a jsonl file with each element being transformed to json and printed to a new line | action
`to_json(path)` | Saves the sequence to a json file. The contents depend on if the json root is an array or dictionary | action
`to_sqlite3(conn, tablename_or_query, *args, **kwargs)` | Save the sequence to a SQLite3 db. The target table must be created in advance. | action
`to_pandas(columns=None)` | Converts the sequence to a pandas DataFrame | action
`cache()` | Forces evaluation of sequence immediately and caches the result | action
`for_each(func)` | Executes `func` on each element of the sequence | action

### Lazy Execution
Whenever possible, `PyFunctional` will compute lazily. This is accomplished by tracking the list
of transformations that have been applied to the sequence and only evaluating them when an action is
called. In `PyFunctional` this is called tracking lineage. This is also responsible for the
ability for `PyFunctional` to cache results of computation to prevent expensive re-computation.
This is predominantly done to preserve sensible behavior and used sparingly. For example, calling
`size()` will cache the underlying sequence. If this was not done and the input was an iterator,
then further calls would operate on an expired iterator since it was used to compute the length.
Similarly, `repr` also caches since it is most often used during interactive sessions where its
undesirable to keep recomputing the same value. Below are some examples of inspecting lineage.

```python
def times_2(x):
    print(x)
    return 2 * x
elements = seq(1, 1, 2, 3, 4).map(times_2).distinct()
elements._lineage
# Lineage: sequence -> map(times_2) -> distinct

l_elements = elements.to_list()
# Prints: 1
# Prints: 1
# Prints: 2
# Prints: 3
# Prints: 4

elements._lineage
# Lineage: sequence -> map(times_2) -> distinct -> cache

l_elements = elements.to_list()
# The cached result is returned so times_2 is not called and nothing is printed
```

Files are given special treatment if opened through the `seq.open` and related APIs.
`functional.util.ReusableFile` implements a wrapper around the standard python file to support
multiple iteration over a single file object while correctly handling iteration termination and
file closing.

## Road Map Idea
* SQL based query planner and interpreter
* `_` lambda operator

## Contributing and Bug Fixes
Any contributions or bug reports are welcome. Thus far, there is a 100% acceptance rate for pull
requests and contributors have offered valuable feedback and critique on code. It is great to hear
from users of the package, especially what it is used for, what works well, and what could be
improved.

To contribute, create a fork of `PyFunctional`, make your changes, then make sure that they pass.
In order to be merged, all pull requests must:

* Pass all the unit tests
* Pass all the pylint tests, or ignore warnings with explanation of why its correct to do so
* Not significantly reduce covrage without a good reason [coveralls.io](coveralls.io/github/EntilZha/PyFunctional))
* Edit the `CHANGELOG.md` file in the `Next Release` heading with changes

## Contact
[Gitter for chat](https://gitter.im/EntilZha/PyFunctional)

## Supported Python Versions
* `PyFunctional` 1.4 and above supports and is tested against Python 3.6, Python 3.7, and PyPy3
* `PyFunctional` 1.4 and above does not support python 2.7
* `PyFunctional` 1.4 and above works in Python 3.5, but is not tested against it
* `PyFunctional` 1.4 and above partially works in 3.8, parallel processing currently has issues, but other feature work fine
* `PyFunctional` 1.3 and below supports and was tested against Python 2.7, Python 3.5, Python 3.6, PyPy2, and PyPy3


## Changelog
[Changelog](https://github.com/EntilZha/PyFunctional/blob/master/CHANGELOG.md)

## About me
To learn more about me (the author) visit my webpage at
[pedro.ai](https://www.pedro.ai).

I created `PyFunctional` while using Python extensivel, and finding that I missed the
ease of use for manipulating data that Spark RDDs and Scala collections have. The project takes the
best ideas from these APIs as well as LINQ to provide an easy way to manipulate data when using
Scala is not an option or PySpark is overkill.

## Contributors
These people have generously contributed their time to improving `PyFunctional`

* [versae](https://github.com/versae)
* [adrian17](https://github.com/adrian17)
* [lucidfrontier45](https://github.com/lucidfrontier45)
* [Digenis](https://github.com/Digenis)
* [ChuyuHsu](https://github.com/ChuyuHsu)
* [jsemric](https://github.com/jsemric)
