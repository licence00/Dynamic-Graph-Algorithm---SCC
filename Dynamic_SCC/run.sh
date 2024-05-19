#!/bin/bash
nvcc -c load.cpp -o load.o
nvcc -c combine.cu -o combine.o
nvcc -o combine combine.o load.o
./combine