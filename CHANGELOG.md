# Changelog
## Release 0.4.0
### New Features
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


### Bug Fixes
* `fold_left` and `fold_right` had incorrect order of arguments for passed function
