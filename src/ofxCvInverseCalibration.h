//
//  ofxCvInverseCalibration.h
//
//  Created by Yuya Hanai on 10/10/14.
//
//
#pragma once

#include "ofMain.h"
#include "ofxCv.h"

// This addon calculates inverse of lens distortion coefficients
// by generating random chessboard image with undistortion (in this case this means distortion)

class ofxCvInverseCalibration
{
public:
    // must be called from GL thread
    static void calculateInverseOf(string chessBoardImagePath, float scale, const ofxCv::Calibration& calibIn,
                                   cv::Mat& outCameraMatrix, cv::Mat& outDistCoeffs);
    
protected:
    static void generateCalibrationImage(ofFbo& fbo, ofImage& chessBoard, float chessBoardScale,
                                         ofxCv::Calibration& calib, ofCamera& cam, float pan, float tilt, ofPixels& outPix);
};
