#!/bin/sh
g++ -std=c++17 `/usr/bin/python3-config --cflags` `/usr/bin/python3-config --ldflags --embed` -o build/$1 src/$1.cpp
