#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main()
{
VideoCapture cap(-1);
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
imwrite("opencv.jpg", frame);
cap.release();
return 0;
}