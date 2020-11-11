#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glut.h>
#include <vector>
#include <iostream>
#include <opencv2/highgui.hpp>
//macros
#define WINDOW_TITLE "Final Project"
#define MAX_FRAME (150)

//namespace
using namespace std;
using namespace cv;
using namespace dnn;

//fucntions
void init_capture(int id);
void init_gl(int argc, char * argv[]);
void init();
void init_dnn();
void init_cv();
void set_callback_functions();

void glut_display();
void glut_keyboard(unsigned char key, int x, int y);
void glut_mouse(int button, int state, int x, int y);
void glut_motion(int x, int y);
void glut_idle();
void glut_timer(int x);

GLuint MatToTexture(cv::Mat image);

void draw_background();

//cv related global varibles
cv::VideoCapture cap;
cv::Mat frame;
cv::Mat handframe,handmask;
cv::Mat faceframe,facehandmask,facemask;
cv::Mat facehandresultframe;
cv::CascadeClassifier cascade;
cv::Size frameSize;
//gl related global variables
GLuint g_bgTextureHandle = 0;
int bg_flag = 1;


//DNN related global variables

cv::dnn::Net net;
float thresh = 0.8;

namespace {
    const char* about = "OpenCV & OpenGL Final Project";
    const char* keys = "{cap | -1 | Camera id }";
    const char* protoFile = "res/pose_deploy.prototxt";
    const char* weightsFile = "res/pose_iter_102000.caffemodel";
    const char* cascadeName = "haarcascade_frontalface_alt.xml";


    const int POSE_PAIRS[20][2] =
    {
        {0,1}, {1,2}, {2,3}, {3,4},         // thumb
        {0,5}, {5,6}, {6,7}, {7,8},         // index
        {0,9}, {9,10}, {10,11}, {11,12},    // middle
        {0,13}, {13,14}, {14,15}, {15,16},  // ring
        {0,17}, {17,18}, {18,19}, {19,20}   // small
    };
    int width = 640;
    int height = 480;
    int nPoints = 22;
}
int main(int argc, char *argv[]){
    cv::CommandLineParser parser(argc,argv,keys);
    parser.about("Test");
    int cap_id = parser.get<int>("cap");
    printf("%d\n",cap_id);
    init_capture(cap_id);
    init_dnn();
    init_gl(argc,argv);
    init_cv();
    init();

    set_callback_functions();
    glutMainLoop();
    return 0;
}

void init(){
    glGenTextures(1, &g_bgTextureHandle);
    glClearColor(0.0,0.0,0.0,0.0);
    glClearDepth(1.0);
    glEnable(GL_DEPTH_TEST);
}
void init_cv(){
    //cv::namedWindow("handmask",1);
    //cv::namedWindow("facemask",1);
    if(!cascade.load(cascadeName)){
        printf("ERROR: cascadeFile not found\n");
        exit(-1);
    }
}
void init_capture(int id){
    cap.open(id);
    if(!cap.isOpened()){
        printf("No Input Video\n");
        exit(0);
    }
    width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    frameSize = cv::Size(width,height);
    facehandresultframe.create(frameSize,CV_8UC3);
}

void init_dnn(){
    net = cv::dnn::readNetFromCaffe(protoFile,weightsFile);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}
void init_gl(int argc, char * argv[]){
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width,height);
    glutCreateWindow(WINDOW_TITLE);
}
void set_callback_functions(){
	glutDisplayFunc(glut_display);
	glutKeyboardFunc(glut_keyboard);
	glutIdleFunc(glut_idle);
    glutTimerFunc(0,glut_timer,1);
}

void glut_display(){
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (GLfloat)width / (GLfloat)height, 1.0, 100.0);
    switch(bg_flag){
        //DEBUG
        case 1:
            g_bgTextureHandle = MatToTexture(frame);
            break;
        case 2:
            g_bgTextureHandle = MatToTexture(handmask);
            break;
        case 3:
            g_bgTextureHandle = MatToTexture(facemask);
            break;
        case 4:
            g_bgTextureHandle = MatToTexture(facehandmask);
            break;
        case 5:
            g_bgTextureHandle = MatToTexture(facehandresultframe);
            break;
        default:
            g_bgTextureHandle = MatToTexture(frame);
            break;
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glColor3f(1.0f,1.0f,1.0f);
    glBindTexture(GL_TEXTURE_2D,g_bgTextureHandle);
    glPushMatrix();
    glTranslatef(0,0,-9.0f);
    draw_background();
    glPopMatrix();
    
    glutSwapBuffers();
}

void draw_background(){
    float x = width/100.0;
    float y = height/100.0;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0,1.0); glVertex3f(-x,-y,0.0);
    glTexCoord2f(1.0,1.0); glVertex3f(x,-y,0.0);
    glTexCoord2f(1.0,0.0); glVertex3f(x,y,0.0);
    glTexCoord2f(0.0,0.0); glVertex3f(-x,y,0.0);
    glEnd();
}

void glut_keyboard(unsigned char key, int x, int y){
    printf("%c\n",key);
	switch(key){
	case 'q':
	case 'Q':
	case '\033':
		exit(0);
    //DEBUG
    case '1':
        bg_flag = 1;
        break;
    case '2':
        bg_flag = 2;
        break;
    case '3':
        bg_flag = 3;
        break;
    case '4':
        bg_flag = 4;
        break;
    case '5':
        bg_flag = 5;
        break;
	}
	glutPostRedisplay();
}

void glut_timer(int x){
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    float aspect_ratio = width/(float)height;
    int inHeight = 360;
    int inWidth = (int(aspect_ratio*inHeight) * 8) / 8;
    cv::Mat inpBlob;
    cv::Mat output;
    if(x==1){
            //Video capture to frame
            cap >> frame;

            //AR marker detection
            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
            if(markerIds.size() > 0)
                cv::aruco::drawDetectedMarkers(frame,markerCorners,markerIds);
            //Face & Hand detection with color
            cv::cvtColor(frame,handframe,cv::COLOR_BGR2HSV);
            cv::Scalar lower = cv::Scalar(0,48,80);
            cv::Scalar upper = cv::Scalar(20,255,255);
            cv::inRange(handframe,lower,upper,handmask);
            //Face recognition & Remove
            cv::cvtColor(frame,faceframe,cv::COLOR_BGR2GRAY);
            cv::equalizeHist(faceframe,faceframe);
            facemask = cv::Mat::zeros(handmask.size(),CV_8U);
            std::vector<cv::Rect> faces;

            cascade.detectMultiScale(faceframe,faces,1.1,3,0,cv::Size(30,30));

            for(int i=0;i < faces.size();i++){
                facemask(cv::Rect(faces[i].x,faces[i].y,faces[i].width,faces[i].height)) = 255;
            }

            cv::bitwise_not(facemask,facemask);
            cv::bitwise_and(facemask,handmask,facehandmask);
            facehandresultframe = cv::Scalar(0);
            frame.copyTo(facehandresultframe,facehandmask);
            //Neural Network Hand Recognition
            inpBlob = cv::dnn::blobFromImage(frame, 1.0 / 255, cv::Size(inWidth, inHeight), cv::Scalar(0, 0, 0), false, false);
            net.setInput(inpBlob);
            output = net.forward();
            int H = output.size[2];
            int W = output.size[3];
            std::vector<cv::Point> points(nPoints);
            for (int n=0; n < nPoints; n++)
            {
                // Probability map of corresponding body's part.
                cv::Mat probMap(H, W, CV_32F, output.ptr(0,n));
                cv::resize(probMap, probMap, cv::Size(width, height));

                cv::Point maxLoc;
                double prob;
                minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
                if (prob > thresh)
                {
                    circle(frame, cv::Point((int)maxLoc.x, (int)maxLoc.y), 8, cv::Scalar(0,255,255), -1);
                    cv::putText(frame, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

                }
                points[n] = maxLoc;
            }
            int nPairs = sizeof(POSE_PAIRS)/sizeof(POSE_PAIRS[0]);

            for (int n = 0; n < nPairs; n++)
            {
                // lookup 2 connected body/hand parts
                Point2f partA = points[POSE_PAIRS[n][0]];
                Point2f partB = points[POSE_PAIRS[n][1]];

                if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
                    continue;

                line(frame, partA, partB, Scalar(0,255,255), 8);
                circle(frame, partA, 8, Scalar(0,0,255), -1);
                circle(frame, partB, 8, Scalar(0,0,255), -1);
            }

            //Convert to RGB in order to display on OpenGL
            cv::cvtColor(frame,frame,cv::COLOR_BGR2RGB);
            //DEBUG
            cv::cvtColor(handmask,handmask,cv::COLOR_GRAY2BGR);
            cv::cvtColor(handmask,handmask,cv::COLOR_BGR2RGB);
            cv::cvtColor(facemask,facemask,cv::COLOR_GRAY2BGR);
            cv::cvtColor(facemask,facemask,cv::COLOR_BGR2RGB);
            cv::cvtColor(facehandmask,facehandmask,cv::COLOR_GRAY2BGR);
            cv::cvtColor(facehandmask,facehandmask,cv::COLOR_BGR2RGB);
            cv::cvtColor(facehandresultframe,facehandresultframe,cv::COLOR_BGR2RGB);
            //Redisplay OpenGL
            glutPostRedisplay();

            //Do this again every 1000/MAX_FRAME(ms)
            glutTimerFunc(1000/MAX_FRAME,glut_timer,1);
    }
}

void glut_idle(){
    return;
}

GLuint MatToTexture(cv::Mat image)
{
    if (image.empty())  return -1;
 
    GLuint textureID;
    glGenTextures(1, &textureID);

    glBindTexture(GL_TEXTURE_2D, textureID); 
 
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows,
        0, GL_RGB, GL_UNSIGNED_BYTE, image.ptr());
 
    return textureID;
}