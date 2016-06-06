# Developer Documentation

## Release Process
For every release, the following process is used.

### Before Release
1. From the project directory, run `./run-tests.sh` to insure unit tests pass (on python 2 and 3),
and that pylint succeeds
2. Push commit which is the candidate release to Github master
3. Wait for tests to pass on [TravisCI](https://travis-ci.org/EntilZha/PyFunctional)
4. Wait for coverage to complete at 100% on [Coveralls](https://coveralls.io/github/EntilZha/PyFunctional)
5. Wait for docs to complete building successfully at [docs.pyfunctional.org/en/latest](http://docs.pyfunctional.org/en/latest/)

### Testing Local Release
1. Run `docker run -it python bash` for clean python installation
2. Clone and install `PyFunctional` with `python setup.py install`
3. Run a python terminal session and insure that `import functional` returns with no errors
4. Repeat steps 6-9 but instead use a python3 docker image

### Building Release
1. Build the source distribution using `python setup.py sdist`
2. Build the wheel distribution using `python bdist_wheel`
3. Assuming a `.pypirc` file like below, double check that `dist/` has the source and wheel
distributions

### Testing on Test PyPI
1. Run `twine upload -r test dist/*` to upload `PyFunctional` to the test PyPi server
2. Browse to the [pypi test server](testpypi.python.org) and insure the webpage looks correct and
that the upload was done correctly.
3. Run `docker run -it python bash` and install the package using `pip install -i https://testpypi.python.org/pypi pyfunctional`.
4. Install dependencies not on the test PyPI instance: `future`, `six`, `dill`, and `backports.lzma`
5. Test that `functional` is importable
6. Repeat using python 3.

If all these steps run, than the candidate release commit will become the new release which
means uploading to live pypi and tagging the commit as a release.

### Publishing Release on Production PyPI
1. Run `twine upload -r pypi dist/*` to publish `PyFunctional` to the live PyPi server.
2. Repeat install tests from Test PyPI testing
3. Tag the release on git with `git tag -a vX.X.X`. Then run `git push` and `git push --tags`
4. On Github, create/edit the release page to match the changelog and add discussion
5. Celebrate!


### `.pypirc` file
```bash
[distutils]
index-servers =
  pypi
  test

[pypi]
repository: https://pypi.python.org/pypi

[test]
repository: https://testpypi.python.org/pypi
```
