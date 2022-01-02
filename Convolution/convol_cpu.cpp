#include <iostream>

#include "mask.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;

void Convol(char* imgData, int height, int width, int widthStep)
{
    getMask();

    /* 建立临时图像数组, 同时给临时数组加上黑边 */
    char* tmpData = new char[(height + 2) * 3 * (width + 2)];
#pragma unrol
    for (int i = 0; i <= height + 1; i++) {
        for (int j = 0; j <= width + 1; j++) {
            if (i >= 1 && i <= height && j >= 1 && j <= width) {
                tmpData[3 * (width + 2) * i + 3 * j + 0] = imgData[widthStep * (i - 1) + 3 * (j - 1) + 0];
                tmpData[3 * (width + 2) * i + 3 * j + 1] = imgData[widthStep * (i - 1) + 3 * (j - 1) + 1];
                tmpData[3 * (width + 2) * i + 3 * j + 2] = imgData[widthStep * (i - 1) + 3 * (j - 1) + 2];
            } else {
                tmpData[3 * (width + 2) * i + 3 * j + 0] = 0;
                tmpData[3 * (width + 2) * i + 3 * j + 1] = 0;
                tmpData[3 * (width + 2) * i + 3 * j + 2] = 0;
            }
        }
    }

#pragma unroll
    /* 对tmpData进行卷积, 并将结果返回给imgData */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int b = 0;
            int g = 0;
            int r = 0;
            /** 此时被卷积的像素点是tmpData[i + 1][j + 1]
             * 与卷积核中心mask[1][1]正对
             */
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    int row = (i + 1) + (k - 1); // 对卷积核某点, 行号相对偏移核中心k - 1
                    int col = (j + 1) + (l - 1); // 对卷积核某点, 列号相对偏移核中心l - 1
                    b += mask[3 * k + l] * tmpData[3 * (width + 2) * row + 3 * col + 0];
                    g += mask[3 * k + l] * tmpData[3 * (width + 2) * row + 3 * col + 1];
                    r += mask[3 * k + l] * tmpData[3 * (width + 2) * row + 3 * col + 2];
                }
            }
            imgData[widthStep * i + 3 * j + 0] = b;
            imgData[widthStep * i + 3 * j + 1] = g;
            imgData[widthStep * i + 3 * j + 2] = r;
        }
    }

    delete[] tmpData;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        puts("Input Error!");
        return -1;
    }
    IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_ANYCOLOR);

    clock_t tic = clock();
    Convol(img->imageData, img->height, img->width, img->widthStep);
    clock_t toc = clock();
    double cpu_time = double(toc - tic) / CLOCKS_PER_SEC;

    cout << endl
         << "====运行"
         << "convol_cpu"
         << "====" << endl
         << "|-图像高宽: " << img->height << " * " << img->width << endl
         << "|-总共耗时: " << cpu_time << " s" << endl
         << endl;

    cvSaveImage(argv[2], img, 0);
    cvReleaseImage(&img);
    return 0;
}