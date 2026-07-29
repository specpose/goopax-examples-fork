#!/bin/sh
g++ -D'MODULE_LIBRARY=1' -shared -fPIC -std=c++17 `/usr/bin/python3-config --cflags` `/usr/bin/python3-config --ldflags --embed` -o build/$1`/usr/bin/python-config --extension-suffix` src/$1.cpp
