#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main()
{
VideoCapture cap(0);
//check if the file was opened properly
if(!cap.isOpened())
{
cout << "Webcam could not be opened succesfully" << endl;
exit(-1);
}
else
{
cout << "p n" << endl;
}
int w = 640;
int h = 480;
cap.set(CAP_PROP_FRAME_WIDTH, w);
cap.set(CAP_PROP_FRAME_HEIGHT, h);
Mat frame;
cap >>frame;
// converts the image to grayscale
Mat frame_in_gray;
cvtColor(frame, frame_in_gray, COLOR_RGB2GRAY);
// process the Canny algorithm
cout << "processing image with Canny..." << endl;
int threshold1 = 0;
int threshold2 = 80;
//Canny(frame_in_gray, frame_in_gray, threshold2, threshold2);
// saving the images in the files system
cout << "Saving the images..." << endl;
imwrite("captured.jpg", frame);
imwrite("captured_with_edges.jpg", frame_in_gray);
imwrite("opencv.jpg", frame);
cap.release();
return 0;
}
