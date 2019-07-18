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

void drawClusters( cv::Mat &img, std::vector<std::vector<double> > &lines, std::vector<std::vector<int> > &clusters, int vpindex[3] )
{
	int cols = img.cols;
	int rows = img.rows;

	//draw lines
	std::vector<cv::Scalar> lineColors( 3 );
	lineColors[vpindex[0]] = cv::Scalar( 0, 0, 255 );
	lineColors[vpindex[1]] = cv::Scalar( 0, 255, 0 );
	lineColors[vpindex[2]] = cv::Scalar( 255, 0, 0 );

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
	// All parameters:

	//string inPutImage = "../raw2.jpg";
	//string inPutImage = "../test.jpg";
	//string inPutImage = "../IMG_1660_half.JPG";
	//string inPutImage = "../IMG_1661_half.JPG";
	string inPutImage = "../lhs.png";

	//Parameters
	double worldcenterproj_x = 970; //992; //529.9345 ; //1100.3393; //654.91; //800;
	double worldcenterproj_y =  795; //797; //1009.13153; //1200.00684; //431.47; //600;	
	double camera_height_in_mm = 9400; //6075.361; //9579.657; //10850; //9400;
	printf("camera_height_in_mm is %f\n", camera_height_in_mm);


	// 40: clusered: 548   X: 258   Y: 119   Z: 171
	// thLength = 42: clusered: 492   X: 234   Y: 108   Z: 150   %:53.216 in noiseratio=0.98
	double thLength = 42.0; // longer segment has more confident to be from linear structure	
	double maxdisplaywidth=4096;
	double x_axis_length_in_mm=50000;
	double axis_length_in_mm=5300; //
	printf("axis_length_in_mm is %f\n", axis_length_in_mm);

	//double f = 1.2*(std::max(image.cols, image.rows))-1;          // Focal length (in pixel)
	double f = 1944;          // Focal length (in pixel)

	cv::Mat camera_matrix_manual = cv::Mat::ones(3, 3, CV_64F);
	camera_matrix_manual.at<double>(0, 0) = 1.94746786e+03;
	camera_matrix_manual.at<double>(1, 0) = 0;
	camera_matrix_manual.at<double>(2, 0) = 0;
	camera_matrix_manual.at<double>(0, 1) = 0;
	camera_matrix_manual.at<double>(1, 1) = 1.94166609e+03;
	camera_matrix_manual.at<double>(2, 1) = 0;
	camera_matrix_manual.at<double>(0, 2) = 9.45510667e+02;
	camera_matrix_manual.at<double>(1, 2) = 4.75102449e+02;
	camera_matrix_manual.at<double>(2, 2) = 1;	

	cv::Mat dist_coefs_manual = cv::Mat::ones(1, 8, CV_64F);
	dist_coefs_manual.at<double>(0, 0) = -6.41206117e+00;
	dist_coefs_manual.at<double>(0, 1) =  2.23841751e+02;
	dist_coefs_manual.at<double>(0, 2) = -1.59456030e-02;
	dist_coefs_manual.at<double>(0, 3) = -1.18112074e-03;
	dist_coefs_manual.at<double>(0, 4) = -9.92272140e+00;
	dist_coefs_manual.at<double>(0, 5) = -6.29789046e+00;
	dist_coefs_manual.at<double>(0, 6) =  2.19709262e+02;
	dist_coefs_manual.at<double>(0, 7) =  1.04513149e+01;


	cv::Mat rawimage= cv::imread( inPutImage );
	if ( rawimage.empty() )
	{
		printf( "Load image error : %s\n", inPutImage.c_str());
	}
	imwrite("rawimage.jpg", rawimage);

	cv::Rect validPixROI;
	cv::Mat camera_matrix_manual_new = cv::getOptimalNewCameraMatrix(camera_matrix_manual, dist_coefs_manual, rawimage.size(), 1, rawimage.size(), &validPixROI);
	cout << "camera_matrix_manual_new: " << camera_matrix_manual_new << endl;
	cout << "validPixROI: " << validPixROI << endl;

	cv::Mat image;
	rawimage.copyTo(image);
	cv::undistort(rawimage, image, camera_matrix_manual, dist_coefs_manual);
	image = image(validPixROI);
	imwrite("rawimage_undistort.jpg", image);

	// LSD line segment detection
	cv::Point2d pp( image.cols/2, image.rows/2 );        // Principle point (in pixel)

	printf("Line length threshold: %fpx\n", thLength);
	std::vector<std::vector<double> > lines;
	LineDetect( image, thLength, lines );
	printf("Numer of line detected: %ld\n", lines.size());

	// Camera internal parameters
	//cv::Point2d pp( 307, 251 );        // Principle point (in pixel)
	//double f = 6.053 / 0.009;          // Focal length (in pixel)



	//cv::Point2d pp( 1015, 578 );        // Principle point (in pixel)
	//double f = 1506; // initial guess??

	printf("Principle point is: %i, %i\n", image.cols/2, image.rows/2);
	printf("focal length is : %f\n", f);

	// Vanishing point detection
	std::vector<cv::Point3d> vps;              // Detected vanishing points (in pixel)
	std::vector<cv::Point2d> vps_2d_raw, vps_2d;              // Detected vanishing points (in pixel)
	std::vector<std::vector<int> > clusters;   // Line segment clustering results of each vanishing point
	VPDetection detector;
	detector.run( lines, pp, f, vps, clusters );
	int vpindex[3] = { 0 }; // defined to be the vp on the road surface

	printf("vps in sphere grid: \n");
	for (int i=0; i < vps.size(); i++) {
    	cout << vps[i] << endl;
	}

	double minx=0, miny=0, maxx=-9999, maxy=-9999;
	printf("vps in image: \n");
	for (int i=0; i < vps.size(); i++) {
		double X = (vps[i].x*f)/vps[i].z + image.cols/2;
		double Y = (vps[i].y*f)/vps[i].z + image.rows/2;
		cv::Point2d ppp(X, Y);

		if(X>maxx) maxx=X;
		if(X<minx) minx=X;
		if(Y>maxy) maxy=Y;
		if(Y<miny) miny=Y;

		vps_2d_raw.push_back(ppp);
    	cout << "[" << X << ", " << Y << "]" << endl;
		if (X>0 && X < image.cols && Y>0 && Y< image.rows)
		{
			circle(image, Point(int(X), int(Y)),20, Scalar(0,0,255), 5);
			line(image, Point(int(X)-40, int(Y)-40), Point(int(X)+40, int(Y)+40),Scalar(0,0,255), 5);
			line(image, Point(int(X)+40, int(Y)-40), Point(int(X)-40, int(Y)+40),Scalar(0,0,255), 5);
			vpindex[0]=i;
		}
		else if(fabs(X)> fabs(Y))
		{
			vpindex[1]=i;
		}
		else
		{
			vpindex[2]=i;
		}
	}


	double raw_width=maxx-minx;
	double raw_height=maxy-miny;
	double ratio = maxdisplaywidth/raw_width;

	printf("minx is %f\n", minx);
	printf("miny is %f\n", miny);
	printf("raw_width is %f\n", raw_width);
	printf("raw_height is %f\n", raw_height);
	printf("ratio is %f\n", ratio);
	printf("firstvpindex is %u\n", vpindex[0]);
	printf("secondvpindex is %u\n", vpindex[1]);
	printf("thirdvpindex is %u\n", vpindex[2]);

	//Rearrange since in the paper, the first vp correspond to 
	for (int i=0; i < vps.size(); i++) {
		vps_2d.push_back(vps_2d_raw[vpindex[i]]);
	}	

	// By equation 18, since R is orthogonal, inner product of 1st and 2nd col yieids:
	double newf;
	newf = sqrt(-1*(vps_2d[0]-pp).dot(vps_2d[1]-pp));
	printf("new focal length is : %f\n", newf);


	// calculate the scales by equation 22
	cv::Mat A = cv::Mat::ones(5, 3, CV_64F);
	A.at<double>(0, 0) = vps_2d[0].x;
	A.at<double>(0, 1) = vps_2d[1].x;
	A.at<double>(0, 2) = vps_2d[2].x;

	A.at<double>(1, 0) = vps_2d[0].y;
	A.at<double>(1, 1) = vps_2d[1].y;
	A.at<double>(1, 2) = vps_2d[2].y;

	A.at<double>(2, 0) = vps_2d[0].x*vps_2d[0].x;
	A.at<double>(2, 1) = vps_2d[1].x*vps_2d[1].x;
	A.at<double>(2, 2) = vps_2d[2].x*vps_2d[2].x;

	A.at<double>(3, 0) = vps_2d[0].y*vps_2d[0].y;
	A.at<double>(3, 1) = vps_2d[1].y*vps_2d[1].y;
	A.at<double>(3, 2) = vps_2d[2].y*vps_2d[2].y;

	A.at<double>(4, 0) = vps_2d[0].x*vps_2d[0].y;
	A.at<double>(4, 1) = vps_2d[1].x*vps_2d[1].y;
	A.at<double>(4, 2) = vps_2d[2].x*vps_2d[2].y;

	cv::Mat B = cv::Mat::ones(5, 1, CV_64F);
	B.at<double>(0, 0) = pp.x;
	B.at<double>(1, 0) = pp.y;
	B.at<double>(2, 0) = newf*newf+pp.x*pp.x;
	B.at<double>(3, 0) = newf*newf+pp.y*pp.y;
	B.at<double>(4, 0) = pp.x*pp.y;

	cv::Mat ScaleMatSol = (A.t()*A).inv()*(A.t()*B);
	double det = cv::determinant(A.t()*A);
	cout << "ATA: " << A.t()*A << endl;
	cout << "det: " << det << endl;
	cout << "ScaleMatSol: " << ScaleMatSol.t() << endl;

	// Calculate back Rotation by equation (19)
	cv::Mat VshMat = cv::Mat::ones(3, 3, CV_64F);
	VshMat.at<double>(0, 0) = vps_2d[0].x;
	VshMat.at<double>(1, 0) = vps_2d[0].y;
	VshMat.at<double>(0, 1) = vps_2d[1].x;
	VshMat.at<double>(1, 1) = vps_2d[1].y;
	VshMat.at<double>(0, 2) = vps_2d[2].x;
	VshMat.at<double>(1, 2) = vps_2d[2].y;

	cv::Mat Mint = cv::Mat::eye(3, 3, CV_64F);
	Mint.at<double>(0, 0) = newf;
	Mint.at<double>(1, 1) = newf;
	Mint.at<double>(0, 2) = pp.x;
	Mint.at<double>(1, 2) = pp.y;


	cv::Mat Rot, Trans;
	int IsFinish=0;
	// Set it be the point with rtk device?

	cout << "====== 8 possibilities of rotation matrix ======" << endl;
	for(int x=-1; x<=1; x+=2)
	{
		for (int y=-1; y<=1; y+=2)
		{
			for (int z=-1; z<=1; z+=2)
			{
				if(true)
				{
					cv::Mat scalemat = cv::Mat::eye(3, 3, CV_64F);
					scalemat.at<double>(0, 0) = x*sqrt(ScaleMatSol.at<double>(0, 0));
					scalemat.at<double>(1, 1) = y*sqrt(ScaleMatSol.at<double>(1, 0));
					scalemat.at<double>(2, 2) = z*sqrt(ScaleMatSol.at<double>(2, 0));

					Rot = Mint.inv()*VshMat*scalemat;
					double det2 = cv::determinant(Rot);
					cout << "det2: " << det2 << endl;

					double diff = cv::sum(Rot.t()-Rot.inv())[0];
					cout << "diff of transpose-inv: " << diff << endl;
					if(fabs(det2-1.0)<0.01 && fabs(diff) < 0.01)
					{
						IsFinish=1;
						cout << "Found the correction rotation matrix" << endl;

						cv::Mat rot_vec = cv::Mat::ones(1, 3, CV_64F);
						cv::Rodrigues(Rot, rot_vec);

						cout << "Rot: " << Rot << endl;
						cout << "rot_vec(in deg): " << (rot_vec*(180.0/3.14159)).t() << endl;		

						//According to equation (11) and (12) of practical...
						//[f  0  u0-u4][tx] = [0]
						//[f  0  v0-v4][ty] = [0]
						//[r13 r23 r33][tz] = [-H]

						// Calculate back Rotation by equation (19)
						cv::Mat Left = cv::Mat::eye(3, 3, CV_64F);
						// Left.at<double>(0, 0) = newf;
						// Left.at<double>(1, 1) = newf;
						// Left.at<double>(0, 2) = pp.x-worldcenterproj_x;
						// Left.at<double>(1, 2) = pp.y-worldcenterproj_y;
						Left.at<double>(0, 0) = camera_matrix_manual_new.at<double>(0, 0);
						Left.at<double>(1, 1) = camera_matrix_manual_new.at<double>(1, 1);
						Left.at<double>(0, 2) = camera_matrix_manual_new.at<double>(0, 2)-worldcenterproj_x;
						Left.at<double>(1, 2) = camera_matrix_manual_new.at<double>(1, 2)-worldcenterproj_y;						
						
						Left.at<double>(2, 0) = Rot.at<double>(0,2);
						Left.at<double>(2, 1) = Rot.at<double>(1,2);
						Left.at<double>(2, 2) = Rot.at<double>(2,2);

						cv::Mat Right = cv::Mat::eye(3, 1, CV_64F);
						Right.at<double>(0, 0) = 0;
						Right.at<double>(1, 0) = 0;
						Right.at<double>(2, 0) = camera_height_in_mm; // camera height is 10.85m

						Trans = Left.inv()*Right;
						cout << "Trans (assuming 4th vp and camera height): " << Trans.t() << endl;											
					}
				}


			}
		}
	}

	if(IsFinish==0)
	{
		cout << "No rotation matrix is correct" << endl;
	}



	//Mat A, w, u, vt;
	//SVD::compute(A, w, u, vt);

	drawClusters( image, lines, clusters,vpindex );
	cout << "Finish drawing segments cluster..." << endl;
	cv::Mat resized;
	resize(image, resized, Size(), ratio, ratio);
	cout << "Size of resized: " << resized.size() << endl;

	Mat final = Mat::zeros(int(raw_height*ratio)+500, int(maxdisplaywidth)+500 , CV_8UC3);
	cout << "Size of final: " << final.size() << endl;
	resized.copyTo(final(Rect(int(-1*minx*ratio), int(-1*miny*ratio), resized.cols, resized.rows)));

	cout << "Finish copying resized frame to final..." << endl;

	for (int i=0; i < vps_2d_raw.size(); i++) {
		double Xr = vps_2d_raw[i].x;
		double Yr = vps_2d_raw[i].y;


		double X = (Xr-minx)*ratio;
		double Y = (Yr-miny)*ratio;
		Scalar color;
		if (i==0) color=Scalar(0,0,255);
		else if (i==1) color=Scalar(0,255,0);
		else color=Scalar(255,0,0);
		circle(final, Point(int(X), int(Y)),20, color, 5);
		line(final, Point(int(X)-40, int(Y)-40), Point(int(X)+40, int(Y)+40),color, 5);
		line(final, Point(int(X)+40, int(Y)-40), Point(int(X)-40, int(Y)+40),color, 5);

	}	

	//print the 3 axis on the ground plane for visualization

	cv::Mat axis = cv::Mat::eye(3, 4, CV_64F);
	axis.at<double>(0, 0) = 0;
	axis.at<double>(1, 0) = 0;
	axis.at<double>(2, 0) = 0;
	axis.at<double>(0, 1) = x_axis_length_in_mm;
	axis.at<double>(1, 1) = 0;
	axis.at<double>(2, 1) = 0;
	axis.at<double>(0, 2) = 0;
	axis.at<double>(1, 2) = axis_length_in_mm;
	axis.at<double>(2, 2) = 0;
	axis.at<double>(0, 3) = 0;
	axis.at<double>(1, 3) = 0;
	axis.at<double>(2, 3) = -axis_length_in_mm;

	std::vector<cv::Point2d> axis_screen;
	cout << "pixel pts of the 3 axis endpt: " << endl;
	for(int i=0; i<4; i++)
	{
		cv::Mat homogen = Mint*(Rot*axis.col(i)+Trans);
		axis_screen.push_back(cv::Point2d(homogen.at<double>(0, 0)/homogen.at<double>(2, 0), homogen.at<double>(1, 0)/homogen.at<double>(2, 0)));
		if(i>0) cout << axis_screen.back() << endl;
	}

	line(image, axis_screen[0], axis_screen[1] ,Scalar(0,0,255), 2);
	line(image, axis_screen[0], axis_screen[2] ,Scalar(0,255,0), 2);
	line(image, axis_screen[0], axis_screen[3] ,Scalar(255,0,0), 2);

	cout << "Camera center in world coord: " << (-1*Rot.t()*Trans).t() << endl;

	imshow("",image);
	imwrite("result.jpg", image);
	imwrite("final.jpg", final);
	cv::waitKey( 0 );
	return 0;
}
