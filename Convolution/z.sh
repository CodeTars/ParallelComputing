g++ -O3 convol_cpu.cpp -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv`
./a.out a.jpg a_out.jpg

g++ -O3 -fopenmp convol_omp.cpp -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv`
./a.out a.jpg aa.jpg && diff aa.jpg a_out.jpg

nvcc -c -O3 -arch=sm_30 convol_plain.cu -o device.o
g++ -c -O3 main.cpp -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv`  -o host.o
g++ -O3 host.o device.o -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv` -o a.out -L /usr/local/cuda/lib64  -lcudart
./a.out a.jpg aa.jpg && diff aa.jpg a_out.jpg

nvcc -c -O3 -arch=sm_30 convol_shared.cu -o device.o
g++ -c -O3 main.cpp -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv`  -o host.o
g++ -O3 host.o device.o -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv` -o a.out -L /usr/local/cuda/lib64  -lcudart
./a.out a.jpg aa.jpg && diff aa.jpg a_out.jpg

nvcc -c -O3 -Xcompiler -fopenmp -arch=sm_30 convol_with_omp.cu -o device.o
g++ -c -O3 -fopenmp main_2gpu.cpp -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv` -o host.o
g++ -O3 -fopenmp host.o device.o -I/usr/local/cuda/include  `pkg-config --libs --cflags opencv` -o a.out -L /usr/local/cuda/lib64  -lcudart
./a.out a.jpg aa.jpg && diff aa.jpg a_out.jpg

rm ./aa.jpg
rm ./a_out.jpg
rm ./*.o
rm ./a.out
