extern "C"
{
    #include "lsd.h"
};
#include "VPDetection.h"
#include<iostream>
#include <math.h> 

using namespace std;
using namespace cv;


// LSD line segment detection
void LineDetect( cv::Mat image, double thLength, std::vector<std::vector<double> > &lines )
{
	cv::Mat grayImage;
	if ( image.channels() == 1 )
		grayImage = image;
	else
		cv::cvtColor(image, grayImage, CV_BGR2GRAY);

	image_double imageLSD = new_image_double( grayImage.cols, grayImage.rows );
	unsigned char* im_src = (unsigned char*) grayImage.data;

	int xsize = grayImage.cols;
	int ysize = grayImage.rows;
	for ( int y = 0; y < ysize; ++y )
	{
		for ( int x = 0; x < xsize; ++x )
		{
			imageLSD->data[y * xsize + x] = im_src[y * xsize + x];
		}
	}

	ntuple_list linesLSD = lsd( imageLSD );
	free_image_double( imageLSD );

	int nLines = linesLSD->size;
	int dim = linesLSD->dim;
	std::vector<double> lineTemp( 4 );
	for ( int i = 0; i < nLines; ++i )
	{
		double x1 = linesLSD->values[i * dim + 0];
		double y1 = linesLSD->values[i * dim + 1];
		double x2 = linesLSD->values[i * dim + 2];
		double y2 = linesLSD->values[i * dim + 3];

		double l = sqrt( ( x1 - x2 ) * ( x1 - x2 ) + ( y1 - y2 ) * ( y1 - y2 ) );
		if ( l > thLength )
		{
			lineTemp[0] = x1;
			lineTemp[1] = y1;
			lineTemp[2] = x2;
			lineTemp[3] = y2;

			lines.push_back( lineTemp );
		}
	}

	free_ntuple_list(linesLSD);
}

void drawClusters( cv::Mat &img, std::vector<std::vector<double> > &lines, std::vector<std::vector<int> > &clusters )
{
	int cols = img.cols;
	int rows = img.rows;

	//draw lines
	std::vector<cv::Scalar> lineColors( 3 );
	lineColors[0] = cv::Scalar( 0, 0, 255 );
	lineColors[1] = cv::Scalar( 0, 255, 0 );
	lineColors[2] = cv::Scalar( 255, 0, 0 );

	for ( int i=0; i<lines.size(); ++i )
	{
		int idx = i;
		cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1]);
		cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3]);
		cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

		cv::line( img, pt_s, pt_e, cv::Scalar(0,0,0), 2, CV_AA );
	}

	for ( int i = 0; i < clusters.size(); ++i )
	{
		for ( int j = 0; j < clusters[i].size(); ++j )
		{
			int idx = clusters[i][j];

			cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1] );
			cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3] );
			cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

			cv::line( img, pt_s, pt_e, lineColors[i], 2, CV_AA );
		}
	}
}

int main()
{
	string inPutImage = "../test.jpg";

	cv::Mat image= cv::imread( inPutImage );
	if ( image.empty() )
	{
		printf( "Load image error : %s\n", inPutImage.c_str());
	}

	// LSD line segment detection
	double thLength = 30.0;
	std::vector<std::vector<double> > lines;
	LineDetect( image, thLength, lines );

	// Camera internal parameters
	//cv::Point2d pp( 307, 251 );        // Principle point (in pixel)
	//double f = 6.053 / 0.009;          // Focal length (in pixel)

	//cv::Point2d pp( image.cols/2, image.rows/2 );        // Principle point (in pixel)
	//double f = 1.2*(std::max(image.cols, image.rows))-1;          // Focal length (in pixel)

	cv::Point2d pp( 1015, 578 );        // Principle point (in pixel)
	double f = 1506;

	printf("Principle point is: %i, %i\n", image.cols/2, image.rows/2);
	printf("focal length is : %f\n", f);

	// Vanishing point detection
	std::vector<cv::Point3d> vps;              // Detected vanishing points (in pixel)
	std::vector<cv::Point2d> vps_2d;              // Detected vanishing points (in pixel)
	std::vector<std::vector<int> > clusters;   // Line segment clustering results of each vanishing point
	VPDetection detector;
	detector.run( lines, pp, f, vps, clusters );

	printf("vps: \n");
	for (int i=0; i < vps.size(); i++) {
    	cout << vps[i] << endl;
	}
	printf("vps in image: \n");
	for (int i=0; i < vps.size(); i++) {
		double X = (vps[i].x*f)/vps[i].z + image.cols/2;
		double Y = (vps[i].y*f)/vps[i].z + image.rows/2;
		cv::Point2d ppp(X, Y);
		vps_2d.push_back(ppp);
    	cout << "[" << X << ", " << Y << "]" << endl;
		if (X>0 && X < image.cols && Y>0 && Y< image.rows)
		{
			circle(image, Point(int(X), int(Y)),20, Scalar(255,255,0), 10);
		}
	}

	double newf;
	newf = sqrt(-1*(vps_2d[0]-pp).dot(vps_2d[1]-pp));
	printf("new focal length is : %f\n", newf);


	// calculate the rotation matrix
	cv::Mat Vpmat = cv::Mat::ones(3, 3, CV_32F);
	Vpmat.at<double>(0, 0) = vps_2d[0].x;
	Vpmat.at<double>(1, 0) = vps_2d[0].y;
	Vpmat.at<double>(0, 1) = vps_2d[1].x;
	Vpmat.at<double>(1, 1) = vps_2d[1].x;
	Vpmat.at<double>(0, 2) = vps_2d[2].x;
	Vpmat.at<double>(1, 2) = vps_2d[2].x;

	cv::Mat Mint = cv::Mat::eyes(3, 3, CV_32F);
	Mint.at<double>(0, 0) = newf;
	Mint.at<double>(1, 1) = newf;
	Mint.at<double>(0, 2) = pp.x;
	Mint.at<double>(1, 2) = pp.y;

	drawClusters( image, lines, clusters );
	imshow("",image);
	imwrite("result.jpg", image);
	cv::waitKey( 0 );
	return 0;
}
