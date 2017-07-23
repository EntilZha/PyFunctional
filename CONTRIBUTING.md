# Contributing and Bug Fixes
Any contributions or bug reports are welcome. Thus far, there is a 100% acceptance rate for pull
requests and contributors have offered valuable feedback and critique on code. It is great to hear
from users of the package, especially what it is used for, what works well, and what could be
improved.

To contribute, create a fork of `PyFunctional`, make your changes, then make sure that they pass
when running on [TravisCI](travis-ci.org) (you may need to sign up for an account and link Github).
In order to be merged, all pull requests must:

* Pass all the unit tests
* Pass all the pylint tests, or ignore warnings with explanation of why its correct to do so
* Must include tests that cover all new code paths
* Must not decrease code coverage (currently at 100% and tested by
[coveralls.io](coveralls.io/github/EntilZha/ScalaFunctional))
* Edit the `CHANGELOG.md` file in the `Next Release` heading with changes
