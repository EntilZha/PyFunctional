# Changelog
## Next Release

## Release 0.7.0
### New Features
* Auto parallelization by using `pseq` instead of `seq`. Details at https://github.com/EntilZha/PyFunctional/issues/47
* Parallel functions: `map`, `select`, `filter`, `filter_not`, `where`, `flatten`, and `flat_map`
* Compressed file IO support for `gzip`/`lzma`/`bz2` as detailed at https://github.com/EntilZha/PyFunctional/issues/54
* Cartesian product from `itertools.product` implemented as `Pipeline.cartesian`
* Website at [pyfunctional.org](http://www.pyfunctional.org) and docs at [docs.pyfunctional.org](http://docs.pyfunctional.org)

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
