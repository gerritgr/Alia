#!/bin/bash

for file in *.ipynb
do
    # Get the file prefix (i.e., strip the .ipynb extension)
    filename="${file%.ipynb}"
    # Convert the notebook to a script
    jupyter nbconvert --to script --output "${filename}.py" "$file"
    mv "${filename}.py.txt" "${filename}.py"
    mv "${filename}.py.py" "${filename}.py"
done

ls
