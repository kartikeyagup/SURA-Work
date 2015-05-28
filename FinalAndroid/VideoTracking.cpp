#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void RemoveLightColors(Mat &img)
{
	for (int i=0; i<img.rows ; i++)
	{
		for (int j=0; j<img.cols; j++)
		{
			if (img.at<uchar>(i,j) <= 50)
			{
				img.at<uchar>(i,j) = 0;
			}
		}
	}
}

std::vector<std::pair<int,int>> GetBrightPoints(Mat &img)
{
	std::vector<std::pair<int,int>> ans;
	for (int i=0; i< img.rows ; i++)
	{
		for (int j=0; j<img.cols ; j++)
		{
			if (img.at<uchar>(i,j) > 100)
			{
				std::cout << (int) img.at<uchar>(i,j) <<"\n";
				ans.push_back(std::pair<int,int> (i,j));
			}
		}
	}
	return ans;
}

std::vector<std::pair<double,double>> FilterVector(std::vector<std::pair<int,int>> inpvector)
{
	std::vector<std::pair<double,double>> ansvect;
	std::vector<int> countvect;
	for (int i=0; i<inpvector.size() ; i++)
	{
		int presentx= inpvector[i].first;
		int presenty= inpvector[i].second;

		bool sofar= false;
		for (int j=0; j<ansvect.size() ; j++)
		{
			double newx= ansvect[j].first/countvect[j], newy=ansvect[j].second/countvect[j];
			if (abs(newx-presentx)<10 && abs(newy-presenty)<10)
			{
				countvect[j]+=1;
				ansvect[j] = std::pair<double,double> (ansvect[j].first+newx, ansvect[j].second+newy);
				std::cout << ansvect[j].first <<"\t" << ansvect[j].second <<"\n";
				sofar=true;
				break;
			}
		}
		if (!sofar)
		{
			ansvect.push_back(std::pair<double,double> ((double) inpvector[i].first,(double) inpvector[i].second) );
			countvect.push_back(1);
		}
	}
	for (int i=0; i<ansvect.size(); i++)
	{
		ansvect[i]= std::pair<double,double> (ansvect[i].first/countvect[i],ansvect[i].second/countvect[i]);
	}
	return ansvect;
}

std::vector<std::pair<double,double>> GetSpeeds(std::vector<std::pair<int,int>> prevframe,std::vector<std::pair<int,int>> curframe, double timeelapsed)
{
	std::vector<std::pair<double,double>> ansspeeds;

	if (prevframe.size() != curframe.size())
	{
		std::cout << "Something wrong will happen, sizes of vectors dont match\n";
	}
	else
	{
		for (int i=0; i<prevframe.size(); i++)
		{
			ansspeeds.push_back(std::pair<double,double> ( (curframe[i].first-prevframe[i].first)/timeelapsed,(curframe[i].second-prevframe[i].second)/timeelapsed));
		}
	}
	return ansspeeds;
}

std::vector<Mat> GetFrames(cv::VideoCapture &cap)
{
	std::vector<Mat> ansvect;
	for(int i=0;;i++)
	{
		//std::cout << i <<"\n";
		cv::Mat frame;
		if (int(cap.get(CV_CAP_PROP_POS_FRAMES)) == int(cap.get(CV_CAP_PROP_FRAME_COUNT)))
			break;
		//std::cout << cap.get(CV_CAP_PROP_POS_FRAMES) <<"\t"<<cap.get(CV_CAP_PROP_FRAME_COUNT) <<"\n";
		if (!cap.read(frame))             
			break;
		ansvect.push_back(frame);
		//cv::imshow("window", frame);
		//char key = cvWaitKey(0);
	}
	return ansvect;
}



int main( int argc, char** argv )
{
    if( argc != 3)
    {
		cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
		int temp;
		cin>>temp;
		return temp;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); // Read the file
	
    if(! image.data ) // Check for invalid input
    {
        cout << "Could not open or find the image\t" <<argv[1] <<std::endl ;
        int temp;
		cin>>temp;
		return temp;
    }
	
	RemoveLightColors(image);
	std::vector<std::pair<double,double>> ans1= FilterVector(GetBrightPoints(image));
	
	for (int i=0; i<ans1.size() ; i++)
	{
		std::cout << "(" << ans1[i].first <<"," << ans1[i].second <<")\t";
	}

	//cout << image <<"\n";

	//image.
	cv::VideoCapture cap(argv[2]);
	/*if (!cap.isOpened())
	{
	    std::cout << "!!! Failed to open file: " << argv[2] << std::endl;
	    return -1;
	}
	std::cout <<cap.get(CV_CAP_PROP_FPS)<<" itne fps\n";
	cv::Mat frame;
	for(;;)
	{
		if (int(cap.get(CV_CAP_PROP_POS_FRAMES)) == int(cap.get(CV_CAP_PROP_FRAME_COUNT)))
			break;
		std::cout << cap.get(CV_CAP_PROP_POS_FRAMES) <<"\t"<<cap.get(CV_CAP_PROP_FRAME_COUNT) <<"\n";
		if (!cap.read(frame))             
			break;
		cv::imshow("window", frame);
		char key = cvWaitKey(0);
	}*/
	
	std::cout << GetFrames(cap).size() <<" these many frames in video\n"; 

    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image ); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}