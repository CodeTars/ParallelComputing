#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <iostream>

using namespace std;

void beforeConvol(char imgData[], int height, int width, int widthStep, int imgSize);

int main(int argc, char* argv[])
{
    if (argc != 3) {
        puts("Input Error!");
        return -1;
    }
    IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_ANYCOLOR);

    beforeConvol(img->imageData, img->height, img->width, img->widthStep, img->imageSize);

    cvSaveImage(argv[2], img, 0);
    cvReleaseImage(&img);
    return 0;
}