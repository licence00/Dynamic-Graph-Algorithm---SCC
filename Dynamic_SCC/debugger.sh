nvcc -g -G -c load.cpp -o load.o
nvcc -g -G -c combine.cu -o combine.o
nvcc -g -G -o combine combine.o load.o
cuda-gdb ./combine