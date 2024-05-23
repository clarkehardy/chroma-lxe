#!/usr/bin/env bash

if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Please install it using 'pip install gdown'."
    exit 1
fi

mkdir ../data/stl &> /dev/null
gdown https://drive.google.com/uc?id=1dLqrSXCod2-ARkKj-hgAGjMm4sxuCku- -O ../data/smalltpc.tar.xz
tar -xvf ../data/smalltpc.tar.xz -C ../data/stl
echo "Removing tarball..."
rm ../data/smalltpc.tar.xz