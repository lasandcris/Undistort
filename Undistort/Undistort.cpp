#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

Ptr<FeaturesFinder> finder;



Mat translateImg(Mat &img, int offsetx, int offsety){
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img, img, trans_mat, img.size());
	return trans_mat;
}

int main(void)
{
	Mat img1 = imread("C:/Users/laszlo.rikker/Downloads/6U5K0387.jpg", IMREAD_GRAYSCALE);
	Mat img2 = imread("C:/Users/laszlo.rikker/Downloads/6U5K0386.jpg", IMREAD_GRAYSCALE);

	
	vector<ImageFeatures> features(2);
	finder = makePtr<OrbFeaturesFinder>();

	for (int i = 0; i < 2; ++i)
	{
		Mat img;
		if (i == 0)
			img = img1;
		else
			img = img2;
		(*finder)(img, features[i]);
		features[i].img_idx = i;
		cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;
	}

	//resize(img1, img1, Size(img1.cols / 2, img1.rows / 2));
	//resize(img2, img2, Size(img2.cols / 2, img2.rows / 2));

	//Mat homography;
	//FileStorage fs("C:/Users/laszlo.rikker/Downloads/OpenCV_V3/opencv/sources/samples/data/H1to3p.xml", FileStorage::READ);
	//fs.getFirstTopLevelNode() >> homography;

	std::vector<cv::Point2f> points1, points2;
	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);

	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < nn_match_ratio * dist2) 
		{
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}

	

	float AvgDistY = 0;
	float distHighestY = 0;
	int atYval = 0;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;

	/*for (unsigned i = 0; i < matched1.size(); i++) 
	{
		
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			//inlier_matches.push_back(DMatch(new_i, new_i, 0));

			float x = matched1[i].pt.x;
			float y = matched1[i].pt.y;

			points1.push_back(cv::Point2f(x, y));

			x = matched2[i].pt.x;
			y = matched2[i].pt.y;

			points2.push_back(cv::Point2f(x, y));
		
	}*/

	

	for (unsigned i = 0; i < matched1.size(); i++) 
	{
		Mat col = Mat::ones(3, 1, CV_64F);
		col.at<double>(0) = matched1[i].pt.x;
		col.at<double>(1) = matched1[i].pt.y;

		

		double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) + pow(col.at<double>(1) - matched2[i].pt.y, 2));
		double distX = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2));
		double distY = sqrt(pow(col.at<double>(1) - matched2[i].pt.y, 2));
		//cout << "matched1 X = " << matched1[i].pt.x << ", matched2 X = " << matched2[i].pt.x << "\n";
		//cout << "matched1 Y = " << matched1[i].pt.y << ", matched2 Y = " << matched2[i].pt.y << "\n";
		//cout << "dist = " << dist << "\n";

		if (dist < 1000)
		{
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);

			


			float x = matched1[i].pt.x;
			float y = matched1[i].pt.y;

			points1.push_back(cv::Point2f(x, y));

			x = matched2[i].pt.x;
			y = matched2[i].pt.y;

			points2.push_back(cv::Point2f(x, y));

			good_matches.push_back(DMatch(new_i, new_i, 0));

			if (distY > distHighestY)
			{
				distHighestY = distY;
				atYval = matched2[i].pt.y;

			}

			AvgDistY += distY;
		}
	}

	// Find the homography between image 1 and image 2
	std::vector<uchar> inliers(points1.size(), 0);
	cv::Mat homography = cv::findHomography(
		cv::Mat(points2), // corresponding
		cv::Mat(points1), // points
		inliers,      //  outputted inliers matches
		CV_RANSAC,    // RANSAC method
		1.);          // max distance to reprojection point

	double XX;
	double YY; 
	bool TX = true;
	bool TY = true;
	focalsFromHomography(homography, XX, YY, TX, TY);

	//estimateFocal(matched1, )

	cout << "focal estX: " << XX << endl;
	cout << "focal estY: " << YY << endl;


	cv::Mat result;
	cv::warpPerspective(img2, // input image
		result,         // output image
		homography,      // homography
		cv::Size(img2.cols,
		img2.rows)); // size of output image

	translateImg(result, 0, 0);
	imwrite("img1.jpg", img1);
	imwrite("result1.jpg", result);

	cv::Mat half(result, cv::Rect(0, 0, img1.cols, img1.rows));
	img1.copyTo(half); // copy image2 to image1 roi

	imwrite("result3.jpg", img1);

	
	AvgDistY = AvgDistY / inliers1.size();
	cout << AvgDistY << endl;
	cout << distHighestY << endl;
	cout << atYval << endl;


	Mat map_x, map_y;
	map_x.create(img2.size(), CV_32FC1);
	map_y.create(img2.size(), CV_32FC1);

	int centX = img2.rows / 2;
	int centY = img2.cols / 2;



	for (int y = 0; y < img1.cols; y++)
	{
		for (int x = 0; x < img1.rows; x++)
		{		 
			float Xmult = (float)x / (float)centX;
			if (Xmult > 1.0)
				Xmult = 2.0 - Xmult;

			map_x.at<float>(y, x) = x;
			map_y.at<float>(y, x) = y + (((((AvgDistY / centY)*y) - AvgDistY)*-1) * Xmult);
		}
	}

	Mat remapped;
	remapped = img2.clone();
	remap(img2, remapped, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

	Mat res;
	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);

	imwrite("res.png", remapped);

	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
	cout << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
	cout << "# Matches:                            \t" << matched1.size() << endl;
	cout << "# Inliers:                            \t" << inliers1.size() << endl;
	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
	cout << endl;

	return 0;
}
