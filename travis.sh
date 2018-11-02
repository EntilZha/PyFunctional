#!/usr/bin/env bash

set -e

pip install pylint

if ! [[ $PANDAS -eq 1 ]]; then
    pip install pandas
fi

nosetests --with-coverage --cover-package=functional
pylint functional
