//
//  ofxCvInverseCalibration.cpp
//
//  Created by Yuya Hanai on 10/10/14.
//
//

#include "ofxCvInverseCalibration.h"

using namespace cv;
using namespace ofxCv;

class FixScaleUndistortion : public ofxCv::Calibration
{
public:
    void updateUndistortionFixScale()
    {
        Mat undistortedCameraMatrix = distortedIntrinsics.getCameraMatrix();
        initUndistortRectifyMap(distortedIntrinsics.getCameraMatrix(), distCoeffs, Mat(), undistortedCameraMatrix, distortedIntrinsics.getImageSize(), CV_16SC2, undistortMapX, undistortMapY);
        undistortedIntrinsics.setup(undistortedCameraMatrix, distortedIntrinsics.getImageSize());
    }
};

class CustomFlagCalibration : public ofxCv::Calibration
{
public:
    bool calibrateWithIntrinsicGuess(const Mat& inCameraMatrix)
    {
        if(size() < 1) {
            ofLog(OF_LOG_ERROR, "Calibration::calibrate() doesn't have any image data to calibrate from.");
            if(ready) {
                ofLog(OF_LOG_ERROR, "Calibration::calibrate() doesn't need to be called after Calibration::load().");
            }
            return ready;
        }
        
        Mat cameraMatrix = inCameraMatrix;
        distCoeffs = Mat::zeros(8, 1, CV_64F);
        
        updateObjectPoints();
        
        int calibFlags = CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_USE_INTRINSIC_GUESS;
        float rms = calibrateCamera(objectPoints, imagePoints, addedImageSize, cameraMatrix, distCoeffs, boardRotations, boardTranslations, calibFlags);
        ofLog(OF_LOG_VERBOSE, "calibrateCamera() reports RMS error of " + ofToString(rms));
        
        ready = checkRange(cameraMatrix) && checkRange(distCoeffs);
        
        if(!ready) {
            ofLog(OF_LOG_ERROR, "Calibration::calibrate() failed to calibrate the camera");
        }
        
        distortedIntrinsics.setup(cameraMatrix, addedImageSize);
        updateReprojectionError();
        updateUndistortion();
        
        return ready;
    }
};

//========================================================================

void ofxCvInverseCalibration::generateCalibrationImage(ofFbo& fbo, ofImage& chessBoard, float chessBoardScale,
                                         ofxCv::Calibration& calib, ofCamera& cam, float pan, float tilt, ofPixels& outPix) {
        ofMatrix4x4 transformMat;
        transformMat.preMultTranslate(ofVec3f(0,0,100));
        transformMat.postMultRotate(pan, 0, 1, 0);
        transformMat.postMultRotate(tilt, 1, 0, 0);
        cam.setTransformMatrix(transformMat);
        cam.begin();
        cam.end();
        
        fbo.begin();
        
        // brief calculate scale
        float chessboardAspect = chessBoard.getWidth() / chessBoard.getHeight();
        float sz = 50;
        int trialCount = 0;
        float minxdiff = 0;
        float maxxdiff = 0;
        float minydiff = 0;
        float maxydiff = 0;
        while (trialCount < 10) {
            vector<ofVec3f> pts;
            pts.push_back(cam.worldToScreen(ofVec3f(sz * chessboardAspect, 0, 0)));
            pts.push_back(cam.worldToScreen(ofVec3f(0, sz, 0)));
            pts.push_back(cam.worldToScreen(ofVec3f(0, -sz, 0)));
            pts.push_back(cam.worldToScreen(ofVec3f(-sz * chessboardAspect, 0, 0)));
            pts.push_back(cam.worldToScreen(ofVec3f(sz * chessboardAspect, sz, 0)));
            pts.push_back(cam.worldToScreen(ofVec3f(-sz * chessboardAspect, sz, 0)));
            pts.push_back(cam.worldToScreen(ofVec3f(sz * chessboardAspect,  -sz, 0)));
            pts.push_back(cam.worldToScreen(ofVec3f(-sz * chessboardAspect, -sz, 0)));
            
            float minX = FLT_MAX, maxX = FLT_MIN, minY = FLT_MAX, maxY = FLT_MIN;
            for (ofVec3f& pt : pts) {
                ofVec2f pt2(pt.x, pt.y);
                ofVec2f undistorted = pt2;//calib.undistort(pt2);
                if (undistorted.x < minX) minX = undistorted.x;
                if (undistorted.y < minY) minY = undistorted.y;
                if (undistorted.x > maxX) maxX = undistorted.x;
                if (undistorted.y > maxY) maxY = undistorted.y;
            }
            float cx = fbo.getWidth() * 0.5f;
            float cy = fbo.getHeight() * 0.5f;
            float dx = max(fabs(minX - cx), fabs(maxX - cx));
            float dy = max(fabs(minY - cy), fabs(maxY - cy));
            float scale = min(cx / dx, cy / dy);
            minxdiff = minX;
            maxxdiff = (maxX - fbo.getWidth());
            minydiff = minY;
            maxydiff = (maxY - fbo.getHeight());
            
            if (scale > 0.95 && scale < 1.05) {
                break;
            }
            sz *= scale;
            trialCount++;
        }
        float basew = sz * chessboardAspect * chessBoardScale;
        float baseh = sz * chessBoardScale;
        float pandiff = (fabs(minxdiff) > fabs(maxxdiff)) ? maxxdiff : minxdiff;
        if (fabs(fabs(minxdiff) - fabs(maxxdiff)) < max(fabs(minxdiff), fabs(maxxdiff)) * 0.1f) {
            pandiff = 0;
        }
        float tiltdiff = (fabs(minydiff) > fabs(maxydiff)) ? maxydiff : minydiff;
        if (fabs(fabs(minydiff) - fabs(maxydiff)) < max(fabs(minydiff), fabs(maxydiff)) * 0.1f) {
            tiltdiff = 0;
        }
        
        float panDeg = ofRadToDeg(atan2f(pandiff, calib.getDistortedIntrinsics().getFocalLength()));
        transformMat.preMultRotate(ofQuaternion(-panDeg * 0.5f, ofVec3f(0,1,0)));
        float tiltDeg = ofRadToDeg(atan2f(tiltdiff, calib.getDistortedIntrinsics().getFocalLength()));
        transformMat.preMultRotate(ofQuaternion(-tiltDeg * 0.5f, ofVec3f(1,0,0)));
        cam.setTransformMatrix(transformMat);
        
        ofClear(0);
        cam.begin();
        chessBoard.draw(-basew, -baseh, 2.0f*basew, 2.0f*baseh);
        ofPushStyle();
        ofSetColor(ofColor::red);
        ofCircle(ofVec3f(sz * chessboardAspect, sz, 0), 1);
        ofCircle(ofVec3f(-sz * chessboardAspect, sz, 0), 1);
        ofPopStyle();
        cam.end();
        fbo.end();
        
        fbo.readToPixels(outPix);
        Mat mat = toCv(outPix);
        calib.undistort(mat, CV_INTER_LINEAR);
}
    
void ofxCvInverseCalibration::calculateInverseOf(string chessBoardImagePath, float scale, const ofxCv::Calibration& calibIn,
                                                 Mat& outCameraMatrix, Mat& outDistCoeffs)
{
    ofImage chessBoard;
    float chessBoardScale;
    ofFbo fbo;
    
    chessBoard.loadImage(chessBoardImagePath);
    chessBoardScale = scale;
    fbo.allocate(calibIn.getDistortedIntrinsics().getImageSize().width, calibIn.getDistortedIntrinsics().getImageSize().height, GL_RGB);
    
    Intrinsics intr;
    intr.setup(calibIn.getDistortedIntrinsics().getCameraMatrix(), calibIn.getDistortedIntrinsics().getImageSize());
    Mat dists = calibIn.getDistCoeffs();
    
    CustomFlagCalibration calibOut;
    FixScaleUndistortion calib;
    calib.setIntrinsics(intr, dists);
    calib.updateUndistortionFixScale();
    
    const float focalLength = calibIn.getDistortedIntrinsics().getFocalLength();
    const float fov = ofRadToDeg(2.0f * atan2f(fbo.getHeight() * 0.5f, focalLength));
    
    ofCamera cam;
    cam.setFov(fov);
    
    ofPixels pix;
    for (float pan = -20; pan <= 20; pan += 10) {
        for (float tilt = -20; tilt <= 20; tilt += 10) {
            generateCalibrationImage(fbo, chessBoard, chessBoardScale, calib, cam, pan, tilt, pix);
            calibOut.add(toCv(pix));
        }
    }
    calibOut.calibrateWithIntrinsicGuess(calibIn.getDistortedIntrinsics().getCameraMatrix());
    
    ofLogVerbose() << "Reprojection Error : " << calibOut.getReprojectionError() << endl;;
    
    outCameraMatrix = calibOut.getDistortedIntrinsics().getCameraMatrix();
    outDistCoeffs = calibOut.getDistCoeffs();
}
