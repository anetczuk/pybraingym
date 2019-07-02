#!/bin/bash

set -eu


echo "Starting virtual env"

bash -i <<< 'source venv/bin/activate; exec </dev/tty'
