# Parallel SCC Computation in CUDA

This project implements a parallel algorithm in CUDA to compute Strongly Connected Components (SCCs) in a directed graph. The CUDA implementation utilizes the indegree and outdegree of nodes and employs a forward and backward algorithm to identify SCCs starting from nodes with the highest degree.

## Compilation Instructions

### Prerequisites

- **CUDA Toolkit**: Ensure CUDA is properly installed and configured.
- **GCC**: Required for compiling CUDA C++.

### Steps

1. **Compile the CUDA file (`SCC.cu`)**:
   ```sh
   nvcc -c SCC.cu -o SCC.o
   ```

2. **Link the object file and create the executable (`SCC`)**:
   ```sh
   nvcc -o SCC SCC.o
   ```

## Execution Instructions

To execute the program, use the following command:
```sh
./SCC <input_file>
```

- `<input_file>`: The name of the file containing the graph data.

### Example

```sh
./SCC graph.txt
```

This command runs the `SCC` program with `graph.txt` as the input file.

## Output

The program outputs the following:
- Number of SCCs in the graph.
- Vertices present in each SCC.

The SCCs are represented using a device_vector `scc` and `scc_offset` in CSR format.

## Specifications

- **CUDA Toolkit**: CUDA 11.2
- **GCC**: GCC 9.2.0

Ensure that both CUDA and GCC are properly installed and configured on your system before compiling and running the program.

