#include <omp.h>

#include <iostream>

#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;

void beforeConvol(char imgData[], int height, int width, int widthStep, int imgSize, int tag);

int main(int argc, char* argv[])
{
    if (argc != 3) {
        puts("Input Error!");
        return -1;
    }
    IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_ANYCOLOR);

    double total_time = -omp_get_wtime();
    int upper_lines = img->height / 2; // 上半部图像的行数
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section // 上半部分图像
        {
            // imageData指向上半图内的首行 (0行)
            beforeConvol(img->imageData, upper_lines, img->width, img->widthStep, upper_lines * img->widthStep, 0);
        }
#pragma omp section // 下半部分图像
        {
            char* start_address_of_data = img->imageData + upper_lines * img->widthStep;
            int bottom_lines = img->height - upper_lines;
            // start_address_of_data指向下半图内首行 (0行)
            beforeConvol(start_address_of_data, bottom_lines, img->width, img->widthStep, bottom_lines * img->widthStep, 1);
        }
    }
    total_time += omp_get_wtime();

    cout << endl
         << "====运行convol_2gpu====" << endl
         << "|-图像高宽: " << img->height << " * " << img->width << endl
         << "|-总共耗时: " << total_time << " s" << endl
         << endl;

    cvSaveImage(argv[2], img, 0);
    cvReleaseImage(&img);
    return 0;
}
