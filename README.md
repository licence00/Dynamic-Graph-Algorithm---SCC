# Dynamic_Graph_Algorithm-SCC
The proposed approach employs an SCC (Strongly Connected Component) tree structure to represent the hierarchical organization of SCCs. This method includes both incremental and decremental components to update the SCC structure efficiently. The incremental component is responsible for handling edge insertions, while the decremental component manages edge deletions. This approach aims to avoid the need to recompute SCCs for the entire graph each time an update occurs.

# Combine Program

This project demonstrates how to read a filename and a float value from command-line arguments, process a file, and use these inputs in a CUDA-enabled C++ program.

## Requirements (Used)

- GCC 9.2.0 
- CUDA Toolkit 11.2

## Files

- load.cpp : Contains the code for loading and processing the file.
- combine.cu : Contains the CUDA code that combines the file processing with additional computations.

## Compilation Instructions

nvcc -c load.cpp -o load.o
nvcc -c combine.cu -o combine.o
nvcc -o combine combine.o load.o

## Execution Instructions

./combine <filename> <percentage_of_updates>

<filename>: The name of the file to be processed.
<percentage_of_updates>: A float value representing the percentage of updates to apply.

## Example
./combine example.txt 0.5\
This command runs the program with example.txt as the input file and 0.5 as the float value for the percentage of updates.
