#!/bin/bash

echo ----------------------------------
echo removing pycache detritus
echo ----------------------------------
echo
rm -vrf $(find . -type d -name __pycache__)
echo

echo ----------------------------------
echo removing mypy detritus
echo ----------------------------------
echo
rm -vrf .mypy_cache

