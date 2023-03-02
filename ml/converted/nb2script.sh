#!/bin/bash
# This script is called from Jupyter notebooks in the parent directory to convert
# produce Python scripts that can be called by torch.multiprocessing

nbname="${1:?Error: the notebook name must be specified}" 
scriptname="converted/${nbname%.ipynb}" 
jupyter nbconvert --to script --output "$scriptname" "$nbname"

# Removing converted code not required for multiprocessing. This code is designated
# with the following comment text:
# Remove all code from '# >>$' to the end of the notebook
sed -i '/^# >>\$/,$d' "$scriptname.py"
# Remove all code from '# >>>' to '# <<<'
sed -i '/^# >>>/,/^# <<</d' "$scriptname.py"