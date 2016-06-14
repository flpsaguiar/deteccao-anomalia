
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"


#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    //Mat img_1 = imread("/home/flpsaguiar/Work/img1.jpg", 1);
    //Mat img_2 = imread("/home/flpsaguiar/Work/img2.jpg", 1);

    Mat img_1 = imread("/home/flpsaguiar/Work/Faculdade/002.jpg", 0);
    Mat img_2 = imread("/home/flpsaguiar/Work/Faculdade/001.jpg", 0);


    //padrão: int nfeatures=0, int nOctaveLayers=3, double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6
    //parametros: numeros de pontos chave,
    //cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(0, 3, 0.04, 10, 0.5);

    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(120, 4, 0.0, 10, 0.5);

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    f2d->detect( img_1, keypoints_1 );
    f2d->detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute( img_1, keypoints_1, descriptors_1 );
    f2d->compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using BFMatcher :
    BFMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );


    Scalar keypointColor = Scalar(255, 0, 0);     // Blue keypoints.
    namedWindow("matches", 1);
    Mat img_matches;
    //drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, keypointColor);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    drawKeypoints(img_1, keypoints_1, img_matches);
    imshow("matches", img_matches);
    waitKey(0);


}



