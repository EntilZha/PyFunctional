#!/usr/bin/env bash

if [[ $LINT ]]; then
    nosetests --with-coverage --cover-package=functional && pylint functional
fi
