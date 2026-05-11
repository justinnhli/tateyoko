#!/bin/bash

set -euo pipefail

source ~/.local/share/venv/ndl-ocr/bin/activate && cd ~/repos/ndlkotenocr-lite/src && python3 ocr.py $@
