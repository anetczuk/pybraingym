#!/bin/bash

set -eu


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"


cd $SCRIPT_DIR


## installing package
pip install -e ./src
# python setup.py install --user
