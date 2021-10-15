#include <opencv2/opencv.hpp>
#include <raspicam_cv.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <wiringPi.h>

using namespace std;
using namespace cv;
using namespace raspicam;

RaspiCam_Cv camera;

Mat frame;
Mat displayTranformation;
Mat framPers;
Mat frameGray;
Mat frameThresh;
Mat frameEdge;
Mat frameFinal;
Mat ROILane;
Mat frameFinalDuplicate;

int iWidth = 520;
int iHeight = 320;
int leftLanePosition;
int rightLanePosition;
int roadCenter;
int frameCenter;
int deviationFromCenterLine;

stringstream stringStream;

vector<int> histogramLane;


//Machine learning

CascadeClassifier stopCascade;
Mat stopFrame;
Mat stopSignRegionOfIntrest;
Mat stopGrey;
vector<Rect> stop;
int distanceToStopsign;

CascadeClassifier objectCascade;
Mat objectFrame;
Mat objectRegionOfIntrest;
Mat objectGrey;
vector<Rect> object;
int distanceToObject;

CascadeClassifier trafficLightCascade;
Mat trafficLightFrame;
Mat trafficLightRegionOfIntrest;
Mat trafficLightGrey;
vector<Rect> trafficLight;
int distanceToTrafficLight;


//points of interest, change the values based on the region required
Point2f boxInFrame[] = {Point2f(25, 250), Point2f(490, 250), Point2f(0, 300), Point2f(520, 300)};
Point2f boxTo[] = {Point2f(60, 0), Point2f(460, 0), Point2f(60, 320), Point2f(460, 320)};


// camera setup. Change camera settings if needed
 void setup ( int argc,char **argv, RaspiCam_Cv &camera ) {
	 
    camera.set ( CAP_PROP_FRAME_WIDTH,  ( "-w",argc,argv,iWidth ) );
    camera.set ( CAP_PROP_FRAME_HEIGHT,  ( "-h",argc,argv,iHeight ) );
    camera.set ( CAP_PROP_BRIGHTNESS, ( "-br",argc,argv,60) );
    camera.set ( CAP_PROP_CONTRAST ,( "-co",argc,argv,60 ) );
    camera.set ( CAP_PROP_SATURATION,  ( "-sa",argc,argv,60 ) );
    camera.set ( CAP_PROP_GAIN,  ( "-g",argc,argv ,50 ) );
    camera.set ( CAP_PROP_FPS,  ( "-fps",argc,argv,100));
}


// get the zoomed in box of the road, this is where we will find the lanes from
void perspective() {
	
	line(frame, boxInFrame[0], boxInFrame[1], Scalar(0,0,255), 2);
	line(frame, boxInFrame[1], boxInFrame[3], Scalar(0,0,255), 2);
	line(frame, boxInFrame[2], boxInFrame[0], Scalar(0,0,255), 2);
	line(frame, boxInFrame[3], boxInFrame[2], Scalar(0,0,255), 2);

	displayTranformation = getPerspectiveTransform(boxInFrame, boxTo);
	
	warpPerspective(frame, framPers, displayTranformation, Size(iWidth,iHeight));
}


// make the perspective black and white and do the canny edge detection
void threshold() {
	
	cvtColor(framPers, frameGray, COLOR_RGB2GRAY);
	
	// change values based on what workes
	// white intencity, black intencity. White should be higher outdoors/ when it is bright 
	inRange(frameGray, 60, 255, frameThresh);
	

	// change the treshhold for what workes 
	Canny(frameGray, frameEdge, 300, 700, 3, false);
	
	add(frameThresh, frameEdge, frameFinal);
	
	cvtColor(frameFinal, frameFinal, COLOR_GRAY2RGB);
	cvtColor(frameFinal, frameFinalDuplicate, COLOR_RGB2BGR);
}


// divide the screen into x number of bars and put the color intensity of each of those bars into an array. Black 0, White 255
void histogram() {
	
	histogramLane.resize(iWidth);
	histogramLane.clear();
	
	for(int i = 0; i < frame.size().width; i++) {
		
		ROILane = frameFinalDuplicate(Rect(i, 200 , 1, 100));	
		divide(255, ROILane, ROILane);
		
		histogramLane.push_back((int)(sum(ROILane)[0]));
	}
}


// find where the array and the bars changes from black to white, aka the lines
void laneFinder() {
	
	vector<int>:: iterator leftPtr;
	leftPtr = max_element(histogramLane.begin()+10, histogramLane.begin() + frame.size().width / 2);
	leftLanePosition = distance(histogramLane.begin(), leftPtr);
	
	vector<int>:: iterator rightPtr;
	rightPtr = max_element(histogramLane.begin() + frame.size().width / 2, histogramLane.end());
	rightLanePosition = distance(histogramLane.begin(), rightPtr);
	
	line(frameFinal, Point2f(leftLanePosition, 0), Point2f(leftLanePosition, 320), Scalar(0, 255, 0), 2);
	line(frameFinal, Point2f(rightLanePosition, 0), Point2f(rightLanePosition, 320), Scalar(0, 255, 0), 2);
}


// find the center of the road and the center of the frame
void centerOfRoad() {
	
	roadCenter = (rightLanePosition-leftLanePosition) / 2 + leftLanePosition;
	frameCenter = (frame.size().width) / 2;
	
	line(frameFinal, Point2f(roadCenter, 0), Point2f(roadCenter, 320), Scalar(255, 0, 255), 3);
	
	// should overlap, so make changes to the frameCenter
	line(frameFinal, Point2f(frameCenter, 0), Point2f(frameCenter, 320), Scalar(255, 165, 0), 3);
	
	// calculate how much the center of the frame is from the center of the road
	// this is the one that will to determine how much the car should steer
	deviationFromCenterLine = roadCenter-frameCenter;
}


void capture() {
	camera.grab();
    camera.retrieve(frame);
	cvtColor(frame, frame, COLOR_BGR2RGB);
	cvtColor(frame, stopFrame, COLOR_BGR2RGB);
}

//machine learning

void stopDetection() {
	
	if(!stopCascade.load("")) //insert the path of your cascade classifier file here for stop sign detection
	{
		
		printf("FAIL in opening cascade file");
	}
	
	stopSignRegionOfIntrest = stopFrame(Rect(0, 0, iWidth, iHeight));
	cvtColor(stopSignRegionOfIntrest, stopGrey, COLOR_RGB2GRAY);
	equalizeHist(stopGrey, stopGrey);
	stopCascade.detectMultiScale(stopGrey, stop);
	
	for( int i = 0; i < stop.size(); i++) {
		
		Point p1(stop[i].x, stop[i].y);
		Point p2(stop[i].x + stop[i].width, stop[i].y + stop[i].height);
		rectangle(stopSignRegionOfIntrest, p1, p2, Scalar(0, 0, 255), 2);
		putText(stopSignRegionOfIntrest, "Stop", p1, FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255, 255), 2);
		
		// y = mx+c
		distanceToStopsign = (-1)*((-1.07)*(p2.x-p1.x) + 102.597);
		
		stringStream.str(" ");
		stringStream.clear();
		
		stringStream<<"Distance: "<<distanceToStopsign<<"cm";
		putText(stopFrame, stringStream.str(), Point2f(1, 130), 0, 1, Scalar(255, 165, 0), 1);
	}
}

void objectDetection() {
	
	if(!objectCascade.load("")) //insert the path of your cascade classifier file here for object detection
	{
		
		printf("FAIL in opening cascade file");
	}
	
	objectRegionOfIntrest = objectFrame(Rect(0, 0, iWidth, iHeight));
	cvtColor(objectRegionOfIntrest, objectGrey, COLOR_RGB2GRAY);
	equalizeHist(objectGrey, objectGrey);
	objectCascade.detectMultiScale(objectGrey, object);
	
	for( int i = 0; i < object.size(); i++) {
		
		Point p1(object[i].x, object[i].y);
		Point p2(object[i].x + object[i].width, object[i].y + object[i].height);
		rectangle(objectRegionOfIntrest, p1, p2, Scalar(0, 0, 255), 2);
		putText(objectRegionOfIntrest, "Object", p1, FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255, 255), 2);
		
		// y = mx+c
		distanceToObject = (-1)*((-1.07)*(p2.x-p1.x) + 102.597);
		
		stringStream.str(" ");
		stringStream.clear();
		
		stringStream<<"Distance: "<<distanceToObject<<"pixels";
		putText(objectFrame, stringStream.str(), Point2f(1, 130), 0, 1, Scalar(255, 165, 0), 1);
	}
}

void trafficLightDetection() {
	
	if(!trafficLightCascade.load("")) //insert the path of your cascade classifier file here for traffic light detection
	{
		
		printf("FAIL in opening cascade file");
	}
	
	trafficLightRegionOfIntrest = trafficLightFrame(Rect(0, 0, iWidth, iHeight ));
	cvtColor(trafficLightRegionOfIntrest, trafficLightGrey, COLOR_RGB2GRAY);
	equalizeHist(trafficLightGrey, trafficLightGrey);
	trafficLightCascade.detectMultiScale(trafficLightGrey, object);
	
	for( int i = 0; i < trafficLight.size(); i++) {
		
		Point p1(trafficLight[i].x, trafficLight[i].y);
		Point p2(trafficLight[i].x + trafficLight[i].width, trafficLight[i].y + trafficLight[i].height);
		rectangle(trafficLightRegionOfIntrest, p1, p2, Scalar(0, 0, 255), 2);
		putText(trafficLightRegionOfIntrest, "Traffic Light", p1, FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255, 255), 2);
		
		// y = mx+c
		distanceToTrafficLight = (-1)*((-1.07)*(p2.x-p1.x) + 102.597);
		
		stringStream.str(" ");
		stringStream.clear();
		
		stringStream<<"Distance: "<<p2.x-p1.x<<"pixels";
		putText(trafficLightFrame, stringStream.str(), Point2f(1, 130), 0, 1, Scalar(255, 165, 0), 1);
	}
}


int main(int argc,char **argv) {
	
    // camera init
	setup(argc, argv, camera);
	cout<<"Connecting to chuyiamera..."<<endl;
	if (!camera.open()) {
		cout<<"Failed to connect!!!"<<endl;
    }
     
    cout<<"Camera Id = "<<camera.getId()<<endl;
		wiringPiSetup();
		pinMode(21, OUTPUT);
		pinMode(22, OUTPUT);
		pinMode(23, OUTPUT);
		pinMode(24, OUTPUT);
     
    while(1) {
		
		auto start = std::chrono::system_clock::now();
		
		
		
		// function calls
		capture();
		perspective();
		threshold();
		histogram();
		laneFinder();
		centerOfRoad();
		stopDetection();
		//objectDetection();
		//trafficLightDetection();
		
		
		// detection
		
		// change distance based on what works
		if(distanceToStopsign < 0 || distanceToStopsign > 0) {
			digitalWrite(21, 1);
			digitalWrite(22, 1);
			digitalWrite(23, 1);
			digitalWrite(24, 0);
			cout<<"Stop Sign"<<endl;
			distanceToStopsign = 0;
			
			goto stopSign;
		}
		
		if(distanceToObject < 20 && distanceToObject > 5) {
			digitalWrite(21, 0);
			digitalWrite(22, 0);
			digitalWrite(23, 0);
			digitalWrite(24, 1);
			cout<<"Object"<<endl;
			distanceToObject = 0;
			
			goto objectInTheWay;
		}
		
		if(distanceToTrafficLight < 20 && distanceToTrafficLight > 5) {
			digitalWrite(21, 1);
			digitalWrite(22, 0);
			digitalWrite(23, 0);
			digitalWrite(24, 1);
			cout<<"Traffic Light"<<endl;
			distanceToTrafficLight = 0;
			
			goto trafficLightInTheWay;
		}
		
		
		// steering
		if (deviationFromCenterLine > -10 && deviationFromCenterLine < 10)
		{
			digitalWrite(21, 0);
			digitalWrite(22, 0);
			digitalWrite(23, 0);
			digitalWrite(24, 0);
			cout<<"Forward"<<endl;
		}
		else if (deviationFromCenterLine >= 10 && deviationFromCenterLine < 20)
		{
			digitalWrite(21, 1);
			digitalWrite(22, 0);
			digitalWrite(23, 0);
			digitalWrite(24, 0);
			cout<<"Right1"<<endl;
		}
		else if (deviationFromCenterLine >= 20 && deviationFromCenterLine < 30)
		{
			digitalWrite(21, 0);
			digitalWrite(22, 1);
			digitalWrite(23, 0);
			digitalWrite(24, 0);
			cout<<"Right2"<<endl;
		}
		else if (deviationFromCenterLine >= 30)
		{
			digitalWrite(21, 1);
			digitalWrite(22, 1);
			digitalWrite(23, 0);
			digitalWrite(24, 0);
			cout<<"Right3"<<endl;
		}
		else if (deviationFromCenterLine <= -10 && deviationFromCenterLine > -20)
		{
			digitalWrite(21, 0);
			digitalWrite(22, 0);
			digitalWrite(23, 1);
			digitalWrite(24, 0);
			cout<<"Left1"<<endl;
		}
		else if (deviationFromCenterLine <= -20 && deviationFromCenterLine > -30)
		{
			digitalWrite(21, 1);
			digitalWrite(22, 0);
			digitalWrite(23, 1);
			digitalWrite(24, 0);
			cout<<"Left2"<<endl;
		}
		else if (deviationFromCenterLine <= -30)
		{
			digitalWrite(21, 0);
			digitalWrite(22, 1);
			digitalWrite(23, 1);
			digitalWrite(24, 0);
			cout<<"Left3"<<endl;
		}

		stopSign:
		objectInTheWay:
		trafficLightInTheWay:
		
		// displaying the images
		stringStream.str(" ");
		stringStream.clear();
		
		stringStream<<"Deviation: "<< deviationFromCenterLine;
		putText(frame, stringStream.str(), Point2f(1, 30), 0, 1, Scalar(255, 165, 0), 1);
		
		namedWindow("Original view", WINDOW_KEEPRATIO);
		moveWindow("Original view", 0, 150);
		resizeWindow("Original view", 640, 480);
		imshow("Original view", frame);
		
		namedWindow("Perspective", WINDOW_KEEPRATIO);
		moveWindow("Perspective", 640, 150);
		resizeWindow("Perspective", 640, 480);
		imshow("Perspective", framPers);
		
		namedWindow("Lane detection", WINDOW_KEEPRATIO);
		moveWindow("Lane detection", 1280, 150);
		resizeWindow("Lane detection", 640, 480);
		imshow("Lane detection", frameFinal);
		
		namedWindow("Stop detection", WINDOW_KEEPRATIO);
		moveWindow("Stop detection", 1280, 580);
		resizeWindow("Stop detection", 480, 360);
		imshow("Stop detection", stopSignRegionOfIntrest);
		
		namedWindow("Object detection", WINDOW_KEEPRATIO);
		moveWindow("Object detection", 1280, 580);
		resizeWindow("Object detection", 480, 360);
		imshow("Object detection", objectRegionOfIntrest);
		
		namedWindow("Traffic light detection", WINDOW_KEEPRATIO);
		moveWindow("Traffic light detection", 1280, 580);
		resizeWindow("Traffic light detection", 480, 360);
		imshow("Traffic light detection", trafficLightRegionOfIntrest);
	   
		
	   
	    // print FPS to the terminal
		waitKey(1);
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-start;
		
		float t = elapsed_seconds.count();
		int FPS = 1/t;
		cout<<"FPS = "<<FPS<<endl;
    }

    return 0;  
}
