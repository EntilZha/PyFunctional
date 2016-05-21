#!/usr/bin/env bash
echo "Running Python 2 Tests"
python -m 'nose' --with-coverage --cover-package=functional --cover-erase
sleep 1
echo "Running Python 3 Tests"
python3 -m 'nose' --with-coverage --cover-package=functional --cover-erase
pylint functional
