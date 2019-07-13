#!/bin/bash

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"


tmpfile=$(mktemp start.venv.XXXXXX.sh --tmpdir)


cat > $tmpfile <<EOL
source $SCRIPT_DIR/venv/bin/activate
if [ \$? -ne 0 ]; then
    echo -e "Unable to activate virtual environment, exiting"
    exit 1
fi 
exec </dev/tty 
EOL


echo "Starting virtual env"

bash -i <<< "source $tmpfile"
