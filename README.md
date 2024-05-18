# Dynamic_Graph_Algorithm-SCC
The proposed approach employs an SCC (Strongly Connected Component) tree structure to represent the hierarchical organization of SCCs. This method includes both incremental and decremental components to update the SCC structure efficiently. The incremental component is responsible for handling edge insertions, while the decremental component manages edge deletions. This approach aims to avoid the need to recompute SCCs for the entire graph each time an update occurs.

# SCC Combine Program

This project implements a method to efficiently manage updates in a Strongly Connected Component (SCC) structure using a combination of C++ and CUDA. The program provides mechanisms for incremental and decremental updates to the SCC structure, avoiding the need to recompute the entire SCC hierarchy for each update.

## Compilation Instructions

### Prerequisites (Modules Used)

- **GCC**: GCC 9.2.0
- **CUDA Toolkit**: CUDA 11.2

### Steps

1. **Compile the C++ file (`load.cpp`)**:
   ```sh
   nvcc -c load.cpp -o load.o
   ```

2. **Compile the CUDA file (`combine.cu`)**:
   ```sh
   nvcc -c combine.cu -o combine.o
   ```

3. **Link the object files and create the executable (`combine`)**:
   ```sh
   nvcc -o combine load.o combine.o
   ```

## Execution Instructions

To execute the program, use the following command:
```sh
./combine <filename> <percentage_of_updates>
```

- `<filename>`: The name of the file to be processed.
- `<percentage_of_updates>`: A float value representing the percentage of updates to apply.

### Example

```sh
./combine example.txt 0.5
```

This command runs the `combine` program with `example.txt` as the input file and `0.5` as the float value for the percentage of updates.

## Specifications

- **GCC**: GCC 9.2.0
- **CUDA Toolkit**: CUDA 11.2

Ensure that both GCC and CUDA Toolkit are properly installed and configured on your system before compiling and running the program.
