#!/bin/bash


## works both under bash and sh
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")




src_dir=$SCRIPT_DIR/../src


## remove ^M character
find $src_dir -type f -name "*.py" -exec sed -i $'s/\r$//' {} +

