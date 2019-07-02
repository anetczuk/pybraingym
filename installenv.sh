#!/bin/bash

set -eu


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"


ENV_DIR=$SCRIPT_DIR/venv

echo "Creating virtual environment in $ENV_DIR"

python3 -m venv $$ENV_DIR



## pip install requests==2.10.0

