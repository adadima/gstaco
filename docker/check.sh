#!/usr/bin/env bash

set -e

mkdir -p cmake-build-docker-$1
cmake -G "Unix Makefiles" -H. -Bcmake-build-docker-$1 -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX
cd cmake-build-docker-$1
make -j4 all
make check