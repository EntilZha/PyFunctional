#!/usr/bin/env bash

set -e

if [[ $PY_36 -eq 1 ]]; then
    pip install git+https://github.com/PyCQA/pylint.git@7daed7b8982480c868b0f642a5251f00ffb253c6
    pip install git+https://github.com/PyCQA/astroid.git@d0b5acdfebcdda5c949584c32a8cbc0f31d5cf25
else
    pip install pylint
fi

if ! [[ $PYPY -eq 1 ]]; then
    pip install pandas
fi

nosetests --with-coverage --cover-package=functional
pylint functional
