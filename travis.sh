#!/usr/bin/env bash

set -e

sudo apt-get install -y pandoc
pip install pylint

if ! [[ $PYPY -eq 1 ]]; then
    pip install pandas
fi

nosetests --with-coverage --cover-package=functional
pylint functional
