#!/usr/bin/env bash

if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Please install it using 'pip install gdown'."
    exit 1
fi

mkdir ../data/stl &> /dev/null
gdown https://drive.google.com/uc?id=1dCRYD4IVnDOVjEO4R5Op6p73rw6N2h4n -O ../data/smalltpc.tar.gz
tar -xvf ../data/smalltpc.tar.gz -C ../data/stl
echo "Removing tarball..."
rm ../data/smalltpc.tar.xz