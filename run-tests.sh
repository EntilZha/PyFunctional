#!/usr/bin/env bash
python -m 'nose' --with-coverage --cover-package=functional --cover-erase
sleep 1
python3 -m 'nose' --with-coverage --cover-package=functional --cover-erase
pylint functional
