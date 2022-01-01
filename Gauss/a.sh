g++ makeEquationData.cpp -O3 && ./a.out
g++ gauss_cpu.cpp -O3 && ./a.out && diff x.txt stdx.txt
g++ gauss_omp.cpp -O3 -fopenmp && ./a.out && diff x.txt stdx.txt
nvcc gauss1.cu -O3 && ./a.out && diff x.txt stdx.txt
nvcc gauss2.1.cu -O3 && ./a.out && diff x.txt stdx.txt
nvcc gauss2.2.cu -O3 && ./a.out && diff x.txt stdx.txt
nvcc gauss2.3.cu -O3 && ./a.out && diff x.txt stdx.txt
nvcc gauss3.cu -O3 && ./a.out && diff x.txt stdx.txt
rm ./a.out