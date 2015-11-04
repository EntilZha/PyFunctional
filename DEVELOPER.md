# Developer Documentation
This file primarily contains useful information for developers of `ScalaFunctional`

## Release Process
For every release, the following process is used.

1. From the project directory, run `./run-tests.sh` to insure unit tests pass (on python 2 and 3),
and that pylint succeeds
2. Push commit which is the candidate release to Github master
3. Wait for tests to pass on [TravisCI](https://travis-ci.org/EntilZha/ScalaFunctional)
4. Wait for coverage to complete at 100% on [Coveralls](https://coveralls.io/github/EntilZha/ScalaFunctional)
5. Wait for docs to complete building successfully at [scalafunctional.readthedocs.org/en/latest](http://scalafunctional.readthedocs.org/en/latest/)
6. Create an empty `virtualenv` by running `virtualenv env_directory` and activate it by running
`source env_directory/bin/activate`
7. Install `ScalaFunctional` into the virtualenv by running `python setup.py install`
8. Run a python terminal session and insure that `import functional` returns with no errors
9. Deactivate the `virtualenv` by running `deactivate`
10. Repeat steps 6-9 but instead use a python3 interpreter
11. Build the source distribution using `python setup.py sdist`
12. Build the wheel distribution using `python bdist_wheel`
13. Assuming a `.pypirc` file like below, double check that `dist/` has the source and wheel
distributions
14. Run `twine upload -r test dist/*` to upload `ScalaFunctional` to the test PyPi server
15. Browse to the [pypi test server](testpypi.python.org) and insure the webpage looks correct and
that the upload was done correctly.
16. Create a new `virtualenv` and install the package using `pip install -i https://testpypi.python.org/pypi scalafunctional`.
Test that the install completes correctly and that `functional` is importable. This may require
installing dependencies from regular `pip` which are not on the test servers like `future`, `six`,
and `enum34` if running python 2.
17. Repeat step 16 using python 3.
18. If all these steps run, than the candidate release commit will become the new release which
means uploading to live pypi and tagging the commit as a release
19. Run `twine upload -r pypi dist/*` to publish `ScalaFunctional` to the live PyPi server.
20. Repeat steps 16 and 17 using the live pip repositories
21. Tag the release on git with `git tag -a vX.X.X`. Then run `git push` and `git push --tags`
22. On Github, create/edit the release page to match the changelog and add discussion
23. Celebrate!


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
