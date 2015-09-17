#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <unistd.h>
using namespace cv;
using namespace std;


Mat src_gray,image,src_gray_prev,mask;
int maxCorners = 10000;
RNG rng(12345);
vector<Point2f> corners,corners_prev;
double qualityLevel = 0.03;
double minDistance = 2;
int blockSize = 7;
bool useHarrisDetector = false;
double k = 0.04;
vector<uchar> status;
int fno=0;
vector<float> err;
bool pointsrecv=true;
Size winSize(15,15);
TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03);



struct CorrData
{
    Point2f PointData;
    int CameraId;
};

vector< vector<pair<Point2f,Point2f> > > Corrpointpairs;
vector<vector<CorrData> > CorrPoints;

string GetCorresString(vector<CorrData> &CorrLine)
{
    string ans="";
    for (int i=0; i<CorrLine.size(); i++)
    {
        ans+=","+to_string(CorrLine[i].CameraId)+","+to_string(CorrLine[i].PointData.x)+","+to_string(CorrLine[i].PointData.y);
    }
    return ans.substr(1);
}

string GetBigStringForCorres(vector<vector<CorrData> > &CorrDataTotal)
{
    string bigans="";
    for (int k=0;k<CorrDataTotal.size(); k++)
    {
        bigans+=GetCorresString(CorrDataTotal[k])+"\n";
    }
    return bigans;
}

void getCorrPoints(vector<Point2f> &previouspoints, vector<Point2f> &presentpoints, vector<uchar> &statusarr)
{
     cerr<<"Starting String corr"<<endl;
    vector< pair<Point2f, Point2f> > ans;
    for (int i=0;i<statusarr.size() ; i++)
    {
        if (statusarr[i]!=0)
        {
            ans.push_back(pair<Point2f, Point2f> (previouspoints[i],presentpoints[i]));
        }
    }
    Corrpointpairs.push_back(ans);
     cerr<<"AnsSize:"<<ans.size()<<"\nDone String corr"<<endl;
}

void UpdateCorrespondance()
{
     for(int i=0;i<Corrpointpairs[0].size();i++)
     {
          CorrData temp1;
          temp1.PointData=Corrpointpairs[0][i].first;
          temp1.CameraId=0;

          CorrData temp2;
          temp2.PointData=Corrpointpairs[0][i].second;
          temp2.CameraId=1;

          vector<CorrData> temp3;
          temp3.push_back(temp1);
          temp3.push_back(temp2);

          CorrPoints.push_back(temp3);

     }

     for(int k=1;k<Corrpointpairs.size();k++)
     {
          vector<pair<Point2f,Point2f> > FramePoints=Corrpointpairs[k];
          int previndex=0;
          for (int i=0;i<FramePoints.size();i++)
         {
             pair<Point2f, Point2f> present=FramePoints[i];
             while(previndex<CorrPoints.size()) {
                 if (CorrPoints[previndex].back().CameraId == k) {
                     if (present.first == CorrPoints[previndex].back().PointData) {
                         CorrData n1;
                         n1.PointData = present.second;
                         n1.CameraId = k+1;
                         CorrPoints[previndex].push_back(n1);
                         break;
                     }
                 }
                 previndex+=1;
             }
             if (previndex>=CorrPoints.size())
             {
                 for (int t=i;t<FramePoints.size(); t++) {
                     CorrData n1;
                     n1.PointData = FramePoints[t].first;
                     n1.CameraId = k;
                     CorrData n2;
                     n2.PointData = FramePoints[t].second;
                     n2.CameraId = k + 1;
                     vector <CorrData> newvect;
                     newvect.push_back(n1);
                     newvect.push_back(n2);
                     CorrPoints.push_back(newvect);
                 }
                 break;
             }
        }
    }
}

Mat KLTracker(Mat &img)
{
     fno+=1;
     cerr<<"Starting Tracking:"<<fno<<endl;
     Mat gray;
     cvtColor(img, gray, CV_RGBA2GRAY);
     resize(gray,src_gray,Size(),0.5,0.5);
     if (pointsrecv)
     {
          goodFeaturesToTrack( src_gray,corners,maxCorners,qualityLevel,minDistance,mask,blockSize,useHarrisDetector,k);
          src_gray.copyTo(src_gray_prev);
          corners_prev = corners;
          if(corners_prev.size()>0)
          {   
               pointsrecv=false;
          }
     }

    else
    {
          string s=to_string(corners_prev.size());
          if (corners_prev.size()>0)
          {
               calcOpticalFlowPyrLK(src_gray_prev, src_gray, corners_prev, corners, status, err,winSize, 2, termcrit, 0, 0.001);
               src_gray.copyTo(src_gray_prev);
               cerr<<"CreatingCorrPoints"<<endl;
               getCorrPoints(corners_prev,corners,status);
               cerr<<"DoneCorrPoints"<<endl;
               corners_prev.clear();
               for(int i=0;i<status.size();i++)
               {
                   if(status[i]!=0)
                   {
                       corners_prev.push_back(corners[i]);
                   }
               }
          }
    }

    if ((fno%10==0) )
    {
          cv::Mat mask= cv::Mat::ones(src_gray.size(), CV_8UC1);
          for (int i=0; i<corners_prev.size(); i++)
          {
              cv::circle(mask, Point2f(corners_prev[i].x,corners_prev[i].y), 3, cv::Scalar( 0 ), -1 );
          }
          vector<Point2f> newcorners;
          goodFeaturesToTrack(src_gray,newcorners,maxCorners,qualityLevel,minDistance,mask,blockSize,useHarrisDetector,k);
          corners_prev.insert(corners_prev.end(),newcorners.begin(),newcorners.end());
          corners.insert(corners.end(),newcorners.begin(),newcorners.end());
    }


    for( size_t i = 0; i < corners_prev.size(); i++ )
    {
          Point2f temp;
          temp.x=2*corners_prev[i].x;
          temp.y=2*corners_prev[i].y;
          cv::circle( gray, temp, 3, cv::Scalar( 255. ), -1 );
    }
    cerr<<"Done"<<endl;
    return gray;
}


int main( int argc, const char** argv )
{
     cv::VideoCapture cap(argv[1]);
     if (!cap.isOpened())
     {
         std::cout << "!!! Failed to open file: " << argv[1] << std::endl;
          return -1;
     }

    cv::Mat frame;
    for(;;)
    {

        if (!cap.read(frame))             
            break;

        if(frame.rows>0 && frame.cols>0)
          frame=KLTracker(frame);
        else
          continue;

        cv::imshow("window", frame);
        usleep(100000);
        char key = cvWaitKey(10);
        if (key == 27)
            break;
    }

    UpdateCorrespondance();

    std::ofstream out("data.csv");
    string strtowrite = GetBigStringForCorres(CorrPoints);
    out<<strtowrite;
    out.close();

    return 0;

}

