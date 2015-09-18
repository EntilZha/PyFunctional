#!/usr/bin/env bash
python -V 2> pyversion
version=$(cat pyversion)
rm pyversion
match=$(echo "$version" | grep "2\.7")
if [ "$match" != "" ]
then
    pip install enum34
fi
