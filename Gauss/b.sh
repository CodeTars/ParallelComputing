g++ makeEquationData.cpp -O3 && ./a.out 30720
nvcc gauss3.cu -O3 && ./a.out > time_30k.txt && diff x.txt stdx.txt
rm ./a.out