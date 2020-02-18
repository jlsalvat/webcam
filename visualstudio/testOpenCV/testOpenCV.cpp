/*
//get a image twice colored and gray and display value of last pixel
#include <iostream>
#include <string>
#include <sstream>
using namespace std;
// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
int main(int argc, const char** argv)
{
	// Read images
	Mat color = imread("Lenna.png");
	Mat gray = imread("Lenna.png", IMREAD_GRAYSCALE);
	// Write images
	imwrite("lenaGray.jpg", gray);
	// Get same pixel with opencv function
	int myRow = color.cols - 1;
	int myCol = color.rows - 1;
	Vec3b pixel = color.at<Vec3b>(myRow, myCol);
	cout << "Pixel value (B,G,R): (" << (int)pixel[0] << "," <<
		(int)pixel[1] << "," << (int)pixel[2] << ")" << endl;
	// show images
	imshow("Lena BGR", color);
	imshow("Lena Gray", gray);
	// wait for any key press
	waitKey(0);
	return 0;
}*/
/*
//read image from webcam and display with a red circle 
#include <iostream>
#include <string>
#include <sstream>
using namespace std;
// OpenCV includes
#include "opencv2/opencv.hpp"
using namespace cv;
#define W 640
#define H 480
int main(int argc, const char** argv)
{
	VideoCapture cap; // open the default camera
	cap.open(0);
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	namedWindow("Video", 1);

	Mat test0 = Mat::zeros(5, 5, CV_8UC3);
	cout << "test0  = " << endl << test0 << endl << endl;
	for (;;)
	{
		Mat frame;
		Mat test1 = Mat::ones(640, 480, CV_8UC3) * 100;
		imshow("test1", test1);
		cap >> frame; // get a new frame from camera
		//add a circle
	 circle(frame,  Point(W / 2, H / 2), 80,
      Scalar( 0, 0, 255 ),
      FILLED,
      LINE_8 );
		imshow("Video", frame);
		if (waitKey(30) >= 0) break;
	}
	// Release the camera or video cap
	cap.release();
	return 0;
}
*/
/*
// play with trackball and mouse and apply filter on image
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::ostringstream
using namespace std;
// OpenCV includes
#include "opencv2/opencv.hpp"
using namespace cv;
// Create a variable to save the position value in track
int blurAmount = 0;
int cpt;
ostringstream sFiltre;
Mat img;

// Trackbar call back function
static void onChange(int pos, void *)
{
	if (pos <= 0)
		return;
	// Aux variable for result
	Mat imgBlur;
	// Get the pointer input image
	// Apply a blur filter
	blur(img, imgBlur, Size(pos, pos));
	// Show the result
	imshow("Lena", imgBlur);
}
void applyFilters() {
	sFiltre.str("");

	switch (cpt) {
	case 0: cvtColor(img, img, COLOR_BGR2GRAY);  sFiltre << "cvtColor COLOR_BGR2GRAY"; break;
	case 1:	blur(img, img, Size(5, 5)); sFiltre << "blur 5x5"; break;
	case 2: Sobel(img, img, CV_8U, 1, 1); sFiltre << "sobel"; break;
	}
	if (cpt > 2) {
		cpt = 0;
		sFiltre << "NO Filter"; 
	}
	else
		cpt++;
	cout << sFiltre.str() << endl;
	putText(img, sFiltre.str(), Point(150, 500), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 0));

}
//Mouse callback
static void onMouse(int event, int x, int y, int, void *)
{
	if (event != EVENT_LBUTTONDOWN)
		return;
	// Get image
	img.release();
	img = imread("Lenna.png");// on recharge l'image 
	// Draw circle
	circle(img, Point(x, y), 10, Scalar(0, 255, 0), 8);
	char caption[30];
	sprintf_s(caption,30, "X:%d, Y=%d", x, y);
	putText(img, caption,Point(x,y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
	applyFilters();
	// Call on change to get blurred image
	imshow("Lena", img);
}

int main(int argc, const char** argv)
{
	// Read images
	img= imread("Lenna.png");
	// Create windows
	namedWindow("Lena");
	// create Buttons
		imshow("Lena", img);
	// create a trackbark
	createTrackbar("Lena", "Lena", &blurAmount, 30, onChange);
	setMouseCallback("Lena", onMouse);

	sFiltre << "NO Filter";
	cout << sFiltre.str() << endl;
	putText(img, sFiltre.str(), Point(150, 500), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 0));
	imshow("Lena", img);
	// Call to onChange to init
	//onChange(blurAmount,NULL);
	// wait app for a key to exit
	waitKey(0);
	// Destroy the windows
	destroyWindow("Lena");
	return 0;
}*/
/*
// display histogram
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::ostringstream
using namespace std;
// OpenCV includes
#include "opencv2/opencv.hpp"
using namespace cv;

Mat img;
int numbins = 256;
/// Set the ranges ( for B,G,R) ), last is not included
const float range[] = { 0, 256 };
const float* histRange = { range };

void showHistoBGR(Mat img, int width, int height)
{
	// Separate image in BRG
	vector<Mat> bgr;
	split(img, bgr);
	// Create the histogram for 256 bins
	// The number of possibles values [0..255]
	int numbins = 256;
	/// Set the ranges ( for B,G,R) ), last is not included
	float range[] = { 0, 256 };
	const float* histRange = { range };
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins,
		&histRange);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins,
		&histRange);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins,
		&histRange);
	// Draw the histogram
	// We go to draw lines for each channel
	// Create image with gray base
	Mat histImage(height, width, CV_8UC3, Scalar(220, 220, 220));
	// Normalize the histograms to height of image
	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);
	int binStep = cvRound((float)width / (float)numbins);
	for (int i = 1; i < numbins; i++)
	{
		line(histImage,
			Point(binStep*(i - 1), height - cvRound(b_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0));
		line(histImage,
			Point(binStep*(i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0));
		line(histImage,
			Point(binStep*(i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255));
	}
	imshow("HistoBGR", histImage);
}
void displayHist(Mat hist, String nameWindows, int width, int height) {

	// Draw the histogram
	// We go to draw lines for each channel
	// Create image with gray base
	Mat histImage(height, width, CV_8UC3, Scalar(220, 220, 220));
	// Normalize the histograms to height of image
	normalize(hist, hist, 0, height, NORM_MINMAX);
	int binStep = cvRound((float)width / (float)numbins);
	for (int i = 1; i < numbins; i++)
	{
		if (hist.at<float>(i - 1) == 0 && hist.at<float>(i) != 0)
			line(histImage,
				Point(binStep*(i - 1), height - cvRound(hist.at<float>(i))),
				Point(binStep*(i), height - cvRound(hist.at<float>(i))),
				Scalar(0, 0, 0));
		if (hist.at<float>(i - 1) != 0 && hist.at<float>(i) == 0)
			line(histImage,
				Point(binStep*(i - 1), height - cvRound(hist.at<float>(i - 1))),
				Point(binStep*(i), height - cvRound(hist.at<float>(i - 1))),
				Scalar(0, 0, 0));
		
		if (hist.at<float>(i - 1) != 0 && hist.at<float>(i) != 0)
			line(histImage,
				Point(binStep*(i - 1), height - cvRound(hist.at<float>(i - 1))),
				Point(binStep*(i), height - cvRound(hist.at<float>(i ))),
				Scalar(0, 0, 0));
		cout << hist.at<float>(i - 1) << " ";
	}
	cout << endl;
	imshow(nameWindows, histImage);
}
void equalizeYCrCB(Mat img)
{
	Mat result;
	Mat hist_before,hist_after;
	// Convert BGR image to YCbCr
	Mat ycrcb;
	cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
	// Split image into channels
	vector<Mat> channels;
	split(ycrcb, channels);
	// Equalize the Y channel only
	calcHist(&channels[0], 1, 0, Mat(), hist_before, 1, &numbins,&histRange);
	displayHist(hist_before, "histoYCrCB before", 256, 100);
	equalizeHist(channels[0], channels[0]);
	calcHist(&channels[0], 1, 0, Mat(), hist_after, 1, &numbins,&histRange);
	displayHist(hist_after, "histoYCrCB after", 256, 100);
	// Merge the result channels
	merge(channels, ycrcb);
	// Convert color ycrcb to BGR
	cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
	// Show image
	imshow("Equalized YCrCb", result);
}
void equalizeHSV(Mat img)
{
	Mat result;
	Mat hist_before, hist_after;
	// Convert BGR image to HSV
	Mat HSV;
	cvtColor(img, HSV, COLOR_BGR2HSV);
	// Split image into channels
	vector<Mat> channels;
	split(HSV, channels);
	// Equalize the Y channel only
	calcHist(&channels[0], 1, 0, Mat(), hist_before, 1, &numbins, &histRange);
	displayHist(hist_before, "histoHSV before", 256, 100);
	equalizeHist(channels[0], channels[0]);
	calcHist(&channels[0], 1, 0, Mat(), hist_after, 1, &numbins, &histRange);
	displayHist(hist_after, "histoHSV after", 256, 100);
	// Merge the result channels
	merge(channels, HSV);
	// Convert color ycrcb to BGR
	cvtColor(HSV, result, COLOR_HSV2BGR);
	// Show image
	imshow("Equalized HSV", result);
}

void lomo(Mat img)
{
	Mat result;
	const double exponential_e = std::exp(1.0);
	// Create Lookup table for color curve effect
	Mat lut(1, 256, CV_8UC1);
	for (int i = 0; i < 256; i++)
	{
		float x = (float)i / 256.0;
		lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(exponential_e,
			-((x - 0.5) / 0.1)))));
	}
	// Split the image channels and apply curve transform only to red	channel
		vector<Mat> bgr;
	split(img, bgr);
	LUT(bgr[2], lut, bgr[2]);
	// merge result
	merge(bgr, result);
	// Create image for halo dark
	Mat halo(img.rows, img.cols, CV_32FC3, Scalar(0.3, 0.3, 0.3));
	// Create circle
	circle(halo, Point(img.cols / 2, img.rows / 2), img.cols / 3,
		Scalar(1, 1, 1), -1);
	blur(halo, halo, Size(img.cols / 3, img.cols / 3));
	// Convert the result to float to allow multiply by 1 factor
	Mat resultf;
	result.convertTo(resultf, CV_32FC3);
	// Multiply our result with halo
	multiply(resultf, halo, resultf);
	// convert to 8 bits
	resultf.convertTo(result, CV_8UC3);
	// show result
	imshow("Lomograpy", result);
}
int main(int argc, const char** argv)
{
	// Read images
	img = imread("Lenna.png");
	// Create windows
	namedWindow("Lena");

	imshow("Lena", img);
	showHistoBGR(img, 256, 100);
	equalizeYCrCB(img);
	lomo(img);
//	equalizeHSV(img);
	// Call to onChange to init
	//onChange(blurAmount,NULL);
	// wait app for a key to exit
	waitKey(0);
	// Destroy the windows
	destroyWindow("Lena");
	return 0;
}
*/



#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::ostringstream
#include <vector>
using namespace std;
// OpenCV includes
#include "opencv2/opencv.hpp"
using namespace cv;

#define DEBUG_TIME
#define SHOW_MOUSE_HSV
#define SIZE_CIRCLE 15
#define FRAME_HEIGHT 480
#define FRAME_WIDTH 640

int WIDTH;
int HEIGHT;
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 10 * 10;
const int MAX_OBJECT_AREA = (int) FRAME_HEIGHT * FRAME_WIDTH / 1.5;

Mat frame, frame2, frame_reduced, frame_contrast, hsvImage, frame_threshold, frame_bitwise,output;
Mat src, frame_erode, frame_dilation;
double dAlpha; /**< Simple contrast control */
int iBeta=1;  /**< Simple brightness control */
int iAlpha = 0;
int iElem = 0;
int iSize = 2;
int iType = 0;
int iIteration = 2;
int iBlur_size = 3;
int iHmin = 5;
int iHmax = 5;
Mat mat_element;
int nb_dilation_erosion=1;
int const max_elem = 2;
int const max_kernel_size = 5;
int const max_iteration = 3;
int const max_blur = 5;
int const max_beta = 100;
int const max_alpha = 200;
int const max_H = 8;
const string trackbarWindowName = "Trackbars";

typedef struct {
	int x;
	int y;
	uint8_t H;
	uint8_t S;
	uint8_t V;
} struct_HSV;

typedef struct {
	int x;
	int y;
	uint8_t B;
	uint8_t G;
	uint8_t R;
} struct_BGR;

enum pos {
	RIGHT,CENTER,LEFT
} ;

vector <struct_BGR> points_BGR;

vector <struct_HSV> points_HSV;

string intToString(int number) {
	std::stringstream ss;
	ss << number;
	return ss.str();
}

Vec3b getHSV(int x, int y, Mat & frame) {
	Mat HSV_frame;
	cvtColor(frame, HSV_frame, COLOR_BGR2HSV);
	Vec3b pixel = HSV_frame.at<Vec3b>(y, x);
	return pixel;
}
Vec3b getBGR(int x, int y, Mat & frame) {
	Vec3b pixel = frame.at<Vec3b>(y, x);
	return pixel;
}
void drawObject(int x, int y, Mat &frame);
void displayObject(Mat & frame, int x, int y, string s);

void displayMire(Mat &frame) {
	rectangle(frame,
		Point(0, 0),
		Point(30, 30),
		Scalar(255, 255, 255),
		-1,
		8);
	rectangle(frame,
		Point(30, 30),
		Point(60, 60),
		Scalar(0, 255, 255),
		-1,
		8);
}

//Mouse callback
static void onMouse(int event, int x, int y, int, void *)
{
	if (event != EVENT_LBUTTONDOWN)
		return;
	// Get image
	Vec3b pixel_HSV = getHSV(x, y,frame);
	Vec3b pixel_GBR = getBGR(x, y,frame);
	struct_HSV point_HSV{ x, y, pixel_HSV[0], pixel_HSV[1], pixel_HSV[2] };
	struct_BGR point_BGR{ x, y, pixel_GBR[0], pixel_GBR[1], pixel_GBR[2] };
	points_HSV.push_back(point_HSV);
	points_BGR.push_back(point_BGR);
	cout << "x=" << x << "y=" << y << " " << "Pixel value (H,S,V):(" << (int)pixel_HSV[0] << "," <<
		(int)pixel_HSV[1] << "," << (int)pixel_HSV[2] << ")" <<
		" (B,G,R):(" << (int)pixel_GBR[0] << "," <<
		(int)pixel_GBR[1] << "," << (int)pixel_GBR[2] << ")" << endl;
	// Call on change to get blurred image

}
void drawMousePoints(Mat & frame) {
	for (int i = 0; i < points_HSV.size(); i++) {
		string s = intToString(points_HSV[i].x) + "," + intToString(points_HSV[i].y) + ":" +
			intToString(points_HSV[i].H) + "," + intToString(points_HSV[i].S) + ":" + intToString(points_HSV[i].V);
		displayObject(frame, points_HSV[i].x, points_HSV[i].y, s);
	}

}
void displayObject(Mat & frame, int x, int y, string s) {
	Scalar sScalar = Scalar(255, 255, 0);
	circle(frame, Point(x, y), SIZE_CIRCLE, sScalar, 2);
	if (y - SIZE_CIRCLE > 0)
		line(frame, Point(x, y), Point(x, y - SIZE_CIRCLE), sScalar, 2);
	else line(frame, Point(x, y), Point(x, 0), sScalar, 2);
	if (y + SIZE_CIRCLE < HEIGHT)
		line(frame, Point(x, y), Point(x, y + SIZE_CIRCLE), sScalar, 2);
	else line(frame, Point(x, y), Point(x, HEIGHT), sScalar, 2);
	if (x - SIZE_CIRCLE > 0)
		line(frame, Point(x, y), Point(x - SIZE_CIRCLE, y), sScalar, 2);
	else line(frame, Point(x, y), Point(0, y), sScalar, 2);
	if (x + SIZE_CIRCLE < WIDTH)
		line(frame, Point(x, y), Point(x + SIZE_CIRCLE, y), sScalar, 2);
	else line(frame, Point(x, y), Point(WIDTH, y), sScalar, 2);
	putText(frame, s, Point(x+20, y),FONT_HERSHEY_SIMPLEX,	0.3, sScalar);
}

void ErosionDilation(int, void*)
{
	int erosion_type = 0;
	switch (iElem) {
		case 0: iType = MORPH_RECT; break;
		case 1: iType = MORPH_CROSS; break;
		case 2: iType = MORPH_ELLIPSE; break;
	}
	mat_element = getStructuringElement(iType,
		Size(2 *iSize + 1, 2 * iSize + 1));
}
void changeContrastBrightness(Mat & frame) {
//	frame_contrast= Mat::zeros(frame.size(), frame.type());
	double alpha;
	alpha = 1.0 + iAlpha / 100.0;
/*	
for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			for (int c = 0; c < 3; c++)
			{
				frame_contrast.at<Vec3b>(j, i)[c] =
					saturate_cast<uchar>(alpha*(frame.at<Vec3b>(j, i)[c]) + iBeta);
			}
		}
	}
*/
	frame.convertTo(frame, -1, alpha, iBeta);// autre solution
}

void createTrackbars() {
	Mat src = Mat::ones(Size(500, 100), CV_8U)*200;
	//create window for trackbars
	namedWindow("Parametres Demo", WINDOW_AUTOSIZE);
	moveWindow("Parametres Demo", 0, 0);
	//resizeWindow("Erosion Dilatation Demo", 300, 150);
	imshow("Parametres Demo", src);
	createTrackbar("iType", "Parametres Demo",
		&iElem, max_elem,
		ErosionDilation);
	createTrackbar("iSize", "Parametres Demo",
		&iSize, max_kernel_size,
		ErosionDilation);
	createTrackbar("iIteration:", "Parametres Demo",
		&iIteration, max_iteration,
		ErosionDilation);
	createTrackbar("iBlur_size:", "Parametres Demo",
		&iBlur_size, max_blur,
		ErosionDilation);
	createTrackbar("alpha:", "Parametres Demo",
		&iAlpha, max_alpha,
		ErosionDilation);
	createTrackbar("beta:", "Parametres Demo",
		&iBeta, max_beta,
		ErosionDilation);
	createTrackbar("HMIN:", "Parametres Demo",
		&iHmin, 10,
		ErosionDilation);
	createTrackbar("HMAX:", "Parametres Demo",
		&iHmax, 10,
		ErosionDilation);
}
static Scalar randomColor(RNG& rng)
{
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}
void ConnectedComponents(Mat &img)
{
	// Use connected components to divide our possibles parts of images
	Mat labels;
	int num_objects = connectedComponents(img, labels);
	// Check the number of objects detected
	if (num_objects < 2) {
		cout << "No objects detected" << endl;
		return;
	}
	else {
		cout << "Number of objects detected: " << num_objects - 1 << endl;
	}
// Create output image coloring the objects
Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
RNG rng(0xFFFFFFFF);
for (int i = 1; i < num_objects; i++) {
	Mat mask = labels == i;
	output.setTo(randomColor(rng), mask);
}
imshow("Result", output);
}

void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-SIZE_CIRCLE,-SIZE_CIRCLE) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - SIZE_CIRCLE > 0)
		line(frame, Point(x, y), Point(x, y - SIZE_CIRCLE), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + SIZE_CIRCLE < FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + SIZE_CIRCLE), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - SIZE_CIRCLE > 0)
		line(frame, Point(x, y), Point(x - SIZE_CIRCLE, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + SIZE_CIRCLE < FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + SIZE_CIRCLE, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}

void reduceMatFromHight(Mat &img, int y) {
	int width = img.cols;
	int height = img.rows - y;
	Rect region_of_interest = Rect(0, y, width, height);
	frame_reduced = img(region_of_interest);
}

void displayTracking(Mat &img, string text, enum pos position) {
	int fontFace = FONT_HERSHEY_COMPLEX;
	double fontScale = 1;
	int thickness = 1;
	int baseline = 0;
	int offset = 10; // offset par rapport au bord haut
	int space = 3; // espace entre le texte et le rectangle
	Size textSize = getTextSize(text, fontFace,
		fontScale, thickness, &baseline);
	baseline += thickness+5;
	// center the text
	Point textOrg((img.cols - textSize.width) / 2,
		(textSize.height)/2 + baseline+ offset);

	if (position == CENTER) {
		// draw the box
	//rectangle(InputOutputArray img, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0);

		rectangle(img, textOrg + Point(-space, baseline + space),
			textOrg + Point(textSize.width + space, -textSize.height - space),
			Scalar::all(255), 3);
		// then put the text itself
		putText(img, text, textOrg, fontFace, fontScale,
			Scalar::all(255), thickness, 8);
	}
	if (position == RIGHT) {
		putText(img, text, Point((img.cols - textSize.width), (textSize.height) / 2 + baseline + offset), fontFace, fontScale,
			Scalar::all(255), thickness, 8);
	}
	if (position == LEFT) {
		putText(img, text, Point(0, (textSize.height) / 2 + baseline + offset), fontFace, fontScale,
			Scalar::all(255), thickness, 8);
	}
}

void ConnectedComponentsStats(Mat &img)
{
	// Use connected components with stats
	Mat labels, stats, centroids;
	int num_objects = connectedComponentsWithStats(img, labels, stats,
		centroids, 8);
	// Check the number of objects detected
	if (num_objects < 2) {
		cout << "No objects detected" << endl;
		return;
	}
	else {
		cout << "Number of objects detected: " << num_objects - 1 << endl;
	}
	// Create output image coloring the objects and show area
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
	cout << "centroids = " << endl << " " << centroids << endl << endl;
	double  *myData = (double*) centroids.data;
	//cout << "*mydata=" << *myData << ":"<<*(myData + 1) << endl;
	RNG rng(0xFFFFFFFF);
	for (int i = 1; i < num_objects; i++) {
//		cout << "Object " << i << " with pos: " << centroids.at<Point2i>(i)<<","<< centroids.at<Point2i>(i);
		cout << "Object " << i << " with pos: "<< *(myData+2*(i)) << ":" << *(myData + 2 * (i)+1);
	cout << " with area " << stats.at<int>(i, CC_STAT_AREA) << endl; 
		Mat mask = labels == i;
		output.setTo(randomColor(rng), mask);
		// draw text with area
		stringstream ss;
		ss << "area: " << stats.at<int>(i, CC_STAT_AREA);
		putText(output,
			ss.str(),
			Point(*(myData + 2 *i), *(myData + 2 *i +1)),
			FONT_HERSHEY_SIMPLEX,
			0.4,
			Scalar(255, 255, 255));
		//draw object location on screen
#ifndef SHOW_MOUSE_HSV
		drawObject((int)*(myData + 2 * i), (int)*(myData + 2 * i + 1), frame);
#endif
	}
	imshow("Result", output);
	moveWindow("Result", 0, HEIGHT);
#ifdef SHOW_MOUSE_HSV
	displayTracking(frame,"MOUSE HSV ON",CENTER);
#else
	displayTracking(frame, " display tracking method 1");
#endif


}



void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	std::vector< std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = (int) hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area > MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = (int)(moment.m10 / area);
					y = (int)(moment.m01 / area);
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				string text = "display trackings method 2";
				displayTracking(frame2, text,CENTER);
				//draw object location on screen
				drawObject(x, y, cameraFeed);
			}

		}
		else displayTracking(frame, "TOO MUCH NOISE",CENTER);
	}
}

int64 DEBUG_TIME_IN() {
#ifdef DEBUG_TIME
	return (getTickCount());
#endif
	return 0;
}
void DEBUG_TIME_OUT(int64 t1,string s) {
#ifdef DEBUG_TIME
	int64 t2 = getTickCount();
	double freq = getTickFrequency();
	double time = (t2 - t1) / freq*1000;
	cout << s << time  << " ms" << endl;
#endif
}

VideoCapture cap; // open the default camera
int main(int argc, char* argv[])
{
	int x, y;
	Scalar outputImage;
	VideoCapture cap(0);
	// If you cannot open the webcam, stop the execution!
	if (!cap.isOpened())
		return -1;
	createTrackbars();
	cap >> frame;
	imshow("frame", frame);
	moveWindow("frame", 0, 0); // Move window to pos. (0, 0)
	string sFPS=intToString(cap.get(CAP_PROP_FPS))+"fps";
	displayTracking(frame, sFPS, RIGHT);
	//pour jouer
//	Mat img_G(Size(100, 50), CV_8UC3, Scalar(0, 255, 0));
//	imshow("GREEN Frame", img_G);
//	resizeWindow("GREEN Frame", 300, 100);
//	waitKey(); // Wait infinitely for key press
	 WIDTH = frame.cols;
	 HEIGHT = frame.rows;
	cout << "WIDTH=" << WIDTH << " HEIGHT" << HEIGHT << endl;
	setMouseCallback("frame", onMouse);
	while (true)	{
		// Initialize the output image before each iteration
		outputImage = Scalar(0, 0, 0);
		// Capture the current frame
		cap >> frame;
		// Check if 'frame' is empty
		if (frame.empty())
			break;
		imshow("frame", frame);
//		int WIDTH = frame.cols;
//		int HEIGHT = frame.rows;
//		cout << "WIDTH=" << WIDTH << " HEIGHT" << HEIGHT << endl;
		setMouseCallback("video", onMouse);

		reduceMatFromHight(frame, 200);
		imshow("frame_reduced", frame_reduced);

		int64 t1;
		t1 = DEBUG_TIME_IN();
		changeContrastBrightness(frame);
		DEBUG_TIME_OUT(t1, "time changeContrastBrightness :");
		t1= DEBUG_TIME_IN();
		medianBlur(frame, frame,iBlur_size*2+1);
		DEBUG_TIME_OUT(t1, "time medianBlur :");
		t1 = DEBUG_TIME_IN();
		frame.copyTo(frame2);



		cvtColor(frame, hsvImage, COLOR_BGR2HSV);
		DEBUG_TIME_OUT(t1, "time cvtColor :");
		// Define the range of "yellow" color in HSV colorspace
		Scalar lowerLimit = Scalar(20+iHmin, 100, 100);
		Scalar upperLimit = Scalar(20+max_H+iHmax, 255, 255);
		// Threshold the HSV image to get only blue color
		t1 = DEBUG_TIME_IN();
		inRange(hsvImage, lowerLimit, upperLimit, frame_threshold);
		DEBUG_TIME_OUT(t1, "time inRange :");
		t1 = DEBUG_TIME_IN();
		erode(frame_threshold, frame_erode, mat_element, Point(-1,-1),iIteration);
		dilate(frame_erode, frame_dilation, mat_element, Point(-1, -1), iIteration);
		DEBUG_TIME_OUT(t1, "erode/dilate time :");
		imshow("frame_threshold", frame_threshold);
		imshow("erode", frame_erode);
		imshow("dilate", frame_dilation);
		Mat frame_gradient = frame_dilation - frame_erode;
		imshow("gradient", frame_gradient);
		t1 = DEBUG_TIME_IN();
		ConnectedComponentsStats(frame_dilation);
		DEBUG_TIME_OUT(t1, "ConnectedComponentsStats time:");
		t1 = DEBUG_TIME_IN();

		trackFilteredObject(x, y, frame_dilation, frame2);
		DEBUG_TIME_OUT(t1, "trackFilteredObject time:");


		drawMousePoints(frame);
		displayMire(frame);
		displayTracking(frame, sFPS, RIGHT);
		imshow("frame", frame);
		imshow("frame2", frame2);
		moveWindow("frame2", WIDTH, 0); // Move window to pos. (WIDTH, 0)

		
		// Get the keyboard input and check if it's 'Esc'
		// 30 -> wait for 30 ms
		// 27 -> ASCII value of 'ESC' key
		int ch = waitKey(30);
		if (ch == 27) {
			break;
		}
	}
	return 1;
}
/*



#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::ostringstream
#include <vector>
using namespace std;
// OpenCV includes
#include "opencv2/opencv.hpp"
using namespace cv;
typedef struct {
	int x;
	int y;
	uint8_t H;
	uint8_t S;
	uint8_t V;
} struct_HSV;

typedef struct {
	int x;
	int y;
	uint8_t B;
	uint8_t G;
	uint8_t R;
} struct_BGR;

vector <struct_BGR> points_BGR;

vector <struct_HSV> points_HSV;

int WIDTH=640;
int HEIGHT=480;
Mat BGR_frame;
Mat HSV_frame;

string intToString(int number) {
	std::stringstream ss;
	ss << number;
	return ss.str();
}
Vec3b getHSV(int x, int y) {
	cvtColor(BGR_frame, HSV_frame, COLOR_BGR2HSV);
	Vec3b pixel = HSV_frame.at<Vec3b>(y,x);
	return pixel;
}
Vec3b getBGR(int x, int y) {
	Vec3b pixel = BGR_frame.at<Vec3b>(y, x);
	return pixel;
}
void displayMire(Mat &frame) {
	rectangle(frame,
		Point(0,0),
		Point(30,30),
		Scalar(255, 255, 255),
		-1,
		8);
	rectangle(frame,
		Point(30, 30),
		Point(60, 60),
		Scalar(0, 255, 255),
		-1,
		8);
}
void displayObject(Mat & frame, int x, int y, string s) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-SIZE_CIRCLE,-SIZE_CIRCLE) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - SIZE_CIRCLE > 0)
		line(frame, Point(x, y), Point(x, y - SIZE_CIRCLE), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + SIZE_CIRCLE < HEIGHT)
		line(frame, Point(x, y), Point(x, y + SIZE_CIRCLE), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, HEIGHT), Scalar(0, 255, 0), 2);
	if (x - SIZE_CIRCLE > 0)
		line(frame, Point(x, y), Point(x - SIZE_CIRCLE, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + SIZE_CIRCLE < WIDTH)
		line(frame, Point(x, y), Point(x + SIZE_CIRCLE, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(WIDTH, y), Scalar(0, 255, 0), 2);
	putText(frame, s, Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
}
//Mouse callback
static void onMouse(int event, int x, int y, int, void *)
{
	if (event != EVENT_LBUTTONDOWN)
		return;
	// Get image
	Vec3b pixel_HSV = getHSV(x, y);
	Vec3b pixel_GBR = getBGR(x, y);
	struct_HSV point_HSV{ x, y, pixel_HSV[0], pixel_HSV[1], pixel_HSV[2] };
	struct_BGR point_BGR{ x, y, pixel_GBR[0], pixel_GBR[1], pixel_GBR[2] };
	points_HSV.push_back(point_HSV);
	points_BGR.push_back(point_BGR);
	cout << "x="<<x<< "y="<<y<<" "<<"Pixel value (H,S,V):(" << (int)pixel_HSV[0] << "," <<
		(int)pixel_HSV[1] << "," << (int)pixel_HSV[2] << ")" <<
		" (B,G,R):(" << (int)pixel_GBR[0] << "," <<
		(int)pixel_GBR[1] << "," << (int)pixel_GBR[2] << ")" << endl;
	// Call on change to get blurred image

}
void drawMousePoints(Mat & frame) {
	for (int i = 0; i < points_HSV.size(); i++) {
		string s = intToString(points_HSV[i].x) + "," + intToString(points_HSV[i].y) + ":" +
			intToString(points_HSV[i].H) + "," + intToString(points_HSV[i].S) + ":" + intToString(points_HSV[i].V);
		displayObject(frame, points_HSV[i].x, points_HSV[i].y, s);
	}

}


int main(int argc, const char** argv)
{
	VideoCapture cap; // open the default camera
	cap.open(0);
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	bool bSuccess = cap.read(BGR_frame); // read a new frame from video
	if (!bSuccess) {
		cout << "Cannot read a frame from video stream" << endl;
		return 1;
	}
	imshow("video", BGR_frame);
	int WIDTH = BGR_frame.cols;
	int HEIGHT = BGR_frame.rows;
	cout << "WIDTH=" << WIDTH << " HEIGHT" << HEIGHT << endl;
	setMouseCallback("video", onMouse);

	for (;;)	{
		cap >> BGR_frame; // get a new frame from camera
		drawMousePoints(BGR_frame);
		displayMire(BGR_frame);
		imshow("video", BGR_frame);
		if (waitKey(30) >= 0) break;
	}
	// Release the camera or video cap
	cap.release();
	return 0;
}
*/

/*
//objectTrackingTutorial.cpp
//https://aishack.in/tutorials/tracking-colored-objects-opencv/
//Written by  Kyle Hounslow 2013

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")
//, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//IN THE SOFTWARE.


#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::ostringstream
using namespace std;
// OpenCV includes
#include "opencv2/opencv.hpp"
using namespace cv;

//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 19;
int H_MAX = 45;
int S_MIN = 69;
int S_MAX = 255;
int V_MIN = 65;
int V_MAX = 255;
//default capture width and height
const int FRAME_WIDTH = 320;//*2;
const int FRAME_HEIGHT = 240;//*2;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS=50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

void on_trackbar( int, void* )
{//This function gets called whenever a
	// trackbar position is changed
}

string intToString(int number){
	std::stringstream ss;
	ss << number;
	return ss.str();
}
void createTrackbars(){
	//create window for trackbars


	namedWindow(trackbarWindowName,0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf( TrackbarName, "H_MIN", H_MIN);
	sprintf( TrackbarName, "H_MAX", H_MAX);
	sprintf( TrackbarName, "S_MIN", S_MIN);
	sprintf( TrackbarName, "S_MAX", S_MAX);
	sprintf( TrackbarName, "V_MIN", V_MIN);
	sprintf( TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->
	createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
	createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
	createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
	createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
	createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
	createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );


}
void drawObject(int x, int y,Mat &frame){

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-SIZE_CIRCLE,-SIZE_CIRCLE) is not within the window!)

	circle(frame,Point(x,y),20,Scalar(0,255,0),2);
	if(y-SIZE_CIRCLE>0)
	line(frame,Point(x,y),Point(x,y-SIZE_CIRCLE),Scalar(0,255,0),2);
	else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
	if(y+SIZE_CIRCLE<FRAME_HEIGHT)
	line(frame,Point(x,y),Point(x,y+SIZE_CIRCLE),Scalar(0,255,0),2);
	else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);
	if(x-SIZE_CIRCLE>0)
	line(frame,Point(x,y),Point(x-SIZE_CIRCLE,y),Scalar(0,255,0),2);
	else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
	if(x+SIZE_CIRCLE<FRAME_WIDTH)
	line(frame,Point(x,y),Point(x+SIZE_CIRCLE,y),Scalar(0,255,0),2);
	else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);

	putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);

}
void morphOps(Mat &thresh){

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));

	erode(thresh,thresh,erodeElement);
	erode(thresh,thresh,erodeElement);


	dilate(thresh,thresh,dilateElement);
	dilate(thresh,thresh,dilateElement);



}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed){

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	std::vector< std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if(numObjects<MAX_NUM_OBJECTS){
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea){
					x = moment.m10/area;
					y = moment.m01/area;
					objectFound = true;
					refArea = area;
				}else objectFound = false;


			}
			//let user know you found an object
			if(objectFound ==true){
				putText(cameraFeed,"Tracking Object",Point(0,50),2,1,Scalar(0,255,0),2);
				//draw object location on screen
				drawObject(x,y,cameraFeed);}

		}else putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
	}
}
int main(int argc, char* argv[])
{
	//some boolean variables for different functionality within this
	//program
	bool trackObjects = true;
	bool useMorphOps = true;
	bool useTrackbars = false;

	//Matrix to store each frame of the webcam feed
	Mat image;
	raspicam::RaspiCam_Cv Camera;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat threshold;
	//x and y values for the location of the object
	int x=0, y=0;
	//create slider bars for HSV filtering
	if(useTrackbars)
		createTrackbars();

	// Setting parameters for the camera

	Camera.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
	Camera.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);

	Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3 );
	Camera.open();

	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop
	while(1){
		//store image to matrix
		//convert frame from BGR to HSV colorspace
		//filter HSV image between values and store filtered image to
		//threshold matrix
		Camera.grab();
		Camera.retrieve(image);
		cvtColor(image,HSV,COLOR_BGR2HSV);
		inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);
		//perform morphological operations on thresholded image to eliminate noise
		//and emphasize the filtered object(s)
		if(useMorphOps)
			morphOps(threshold);
		//pass in thresholded frame to our object tracking function
		//this function will return the x and y coordinates of the
		//filtered object
		if(trackObjects)
			trackFilteredObject(x,y,threshold,image);

		//show frames
		//imshow(windowName2,threshold);
		//imshow(windowName,cameraFeed);
		imshow(windowName, image);
		//imshow(windowName1,HSV);

		//Camera.release();

		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
		waitKey(15);
	}
	Camera.release();






	return 0;
}

*/

