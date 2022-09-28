# Changelog
## Next Release
* Allow empty sequence expressions `seq()`, `pseq()` (#159)
* Add `raw` option to `head()`, `head_option()`, `first()`, `last()` and `last_option()`

## Release 1.3.0
* added precompute attribute to reverse transformation (#137)
* Update setup.py dill to requirements.txt (#138)
* Docstring of tail fixed (#140)
* adding extend feature (#144)

## Release 1.2.0
* Fix Broken link in readme
* Loosen version requirements #129
* Fix lint errors
* Fix StopIteration errors for Python 3.7 #132
* Drop support for python 3.4

## Release 1.1.3
* Fix bug in `partition` https://github.com/EntilZha/PyFunctional/issues/124

## Release 1.1.0

* Implemented optimized version of `reduce_by_key`
* Implemented `count_by_key`
* Implemented `count_by_value`
* Implemented `accumulate` https://github.com/EntilZha/PyFunctional/pull/104
* Fix bug in `grouped` https://github.com/EntilZha/PyFunctional/pull/123
* Fix bug in `to_csv` https://github.com/EntilZha/PyFunctional/pull/123
* Fix bug with incorrect wrapping of pandas dataframes https://github.com/EntilZha/PyFunctional/pull/122
* Allow variance on versions of certain packages: https://github.com/EntilZha/PyFunctional/pull/117 and https://github.com/EntilZha/PyFunctional/pull/116
* Various typo fixes
* Various CI fixes
* Fix issue with `first/head` evaluating entire sequence https://github.com/EntilZha/PyFunctional/commit/fb8f3686cf94f072f4e6ed23a361952de1447dc8
* Drop CI testing and official support for Python 3.3
* Make import much faster by loading pandas more lazily https://github.com/EntilZha/PyFunctional/issues/99

## Release 1.0.0

Reaching `1.0` primarily means that API stability has been reached so I don't expect to run into many new breaking changes.

### New Features

* Added optional initial value for `reduce` (https://github.com/EntilZha/PyFunctional/issues/86)
* Added table of contents to readme (https://github.com/EntilZha/PyFunctional/issues/88)
* Added data interchange tutorial with pandas (https://github.com/EntilZha/PyFunctional/blob/master/examples/PyFunctional-pandas-tutorial.ipynb)
* Implemented `itertools.starmap` as `Sequence.starmap` and `Sequence.smap` (https://github.com/EntilZha/PyFunctional/issues/90)
* Added interface to `csv.DictReader` via `seq.csv_dict_reader` (https://github.com/EntilZha/PyFunctional/issues/92)
* Improved `_html_repr_`, `show` and `tabulate` by auto detecting named tuples as column names (https://github.com/EntilZha/PyFunctional/issues/91)
* Improved `_html_repr_` and `show` to tell the user 10 of N rows are being shown if there are more than 10 rows (https://github.com/EntilZha/PyFunctional/issues/94)

### Dependencies and Supported Python Versions
* Bumped version dependencies (https://github.com/EntilZha/PyFunctional/issues/89)
* Added Python 3.6 via Travis CI testing


## Release 0.8.0
### New Features

* Implemented pretty html repr for Jupyter
* Implemented proper parsing of pandas DataFrames
* Detect when its possible to pretty print a table and do so
* `list`/`to_list` have a parameter `n` to limit number of results

### Bug Fixes

* Fixed bug where `grouped` unnecessarily forces precomputation of sequence
* Remove package installations from default requirements that sometimes break installation on barebones systems in python 2.7

## Release 0.7.0
### New Features
* Auto parallelization by using `pseq` instead of `seq`. Details at https://github.com/EntilZha/PyFunctional/issues/47
* Parallel functions: `map`, `select`, `filter`, `filter_not`, `where`, `flatten`, and `flat_map`
* Compressed file IO support for `gzip`/`lzma`/`bz2` as detailed at https://github.com/EntilZha/PyFunctional/issues/54
* Cartesian product from `itertools.product` implemented as `Pipeline.cartesian`
* Website at [pyfunctional.pedro.ai](http://pyfunctional.pedro.ai) and docs at [docs.pyfunctional.pedro.ai](http://docs.pyfunctional.pedro.ai)

### Bug Fixes
* No option for encoding in `to_json` https://github.com/EntilZha/PyFunctional/issues/70

### Internal Changes
* Pinned versions of all dependencies

### Contributors
* Thanks to [versae](https://github.com/versae) for implementing most of the `pseq` feature!
* Thanks to [ChuyuHsu](https://github.com/ChuyuHsu) for implemented large parts of the compression feature!

## Release 0.6.0
### New Features
* Added support for reading to and from SQLite databases
* Change project name to `PyFunctional` from `ScalaFunctional`
* Added `to_pandas` call integration

### Internal Changes
* Changed code quality check service


## Release 0.5.0
### New Features
* Added delimiter option on `to_file`
* `Sequence.sliding` to create a sliding window from a list of elements

### Internal Changes
* Changed all relative imports to absolute imports with `__future__.absolute_import`

### Bug Fixes
* Fixed case where `_wrap` is changing named tuples to arrays when it should preserve them
* Fixed documentation on `to_file` which incorrectly copied from `seq.open` delimiter parameter
* Fixed `Sequence.zip_with_index` behavior. used to mimic `enumerate` by zipping on the left size
while scala and spark do zip on the right side. This introduces different behavior and more flexible
behavior in combination with `enumerate` A start parameter was also added like in `enumerate`

## Release 0.4.1
Fix python 3 build error due to wheel installation of enum34. Package no longer depends on enum34

## Release 0.4.0
### New Features
* Official and tested support for python 3.5. Thus `ScalaFunctional` is tested on Python 2.7, 3.3,
3.4, 3.5, pypy, and pypy3
* `aggregate` from LINQ
* `order_by` from LINQ
* `where` from LINQ
* `select` from LINQ
* `average` from LINQ
* `sum` modified to allow LINQ projected sum
* `product` modified to allow LINQ projected product
* `seq.jsonl` to read jsonl files
* `seq.json` to read json files
* `seq.open` to read files
* `seq.csv` to read csv files
* `seq.range` to create range sequences
* `Sequence.to_jsonl` to save jsonl files
* `Sequence.to_json` to save json files
* `Sequence.to_file` to save files
* `Sequence.to_csv` to save csv files
* Improved documentation with more examples and mention LINQ explicitly
* Change PyPi keywords to improve discoverability
* Created [Google groups mailing list](https://groups.google.com/forum/#!forum/scalafunctional)

### Bug Fixes
* `fold_left` and `fold_right` had incorrect order of arguments for passed function
