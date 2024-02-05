import pytest

from functional import seq


@pytest.fixture(autouse=True)
def add_seq(doctest_namespace):
    doctest_namespace["seq"] = seq
