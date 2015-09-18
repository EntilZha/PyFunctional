#!/usr/bin/env bash
python -V 2> pyversion
version=$(cat pyversion)
rm pyversion
match=$(echo "$version" | grep -E "(2\.7)|(3\.0)|(3\.1)|(3\.2)|(3\.3)")
if [ "$match" != "" ]
then
    pip install enum34
fi
