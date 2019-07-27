#!/bin/bash


## works both under bash and sh
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")


tmp_dir=$SCRIPT_DIR/../tmp

prof_file=$tmp_dir/script.prof


python -m cProfile -o $prof_file $@

## in case of missing program install it by: "pip install --user pyprof2calltree"
pyprof2calltree -i $prof_file -k
