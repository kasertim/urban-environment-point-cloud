#include <iostream>
#include <string>

#include <cv.h>
#include <highgui.h>

int main(int argc, char **argv) {
  IplImage *img;
  img = cvLoadImage(argv[1],CV_LOAD_IMAGE_UNCHANGED);
  
  return 0;
}