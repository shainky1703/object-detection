import cv2
import numpy as np
MIN_MATCH_COUNT=30
detector=cv2.SIFT()
FLANN_INDEX_KDTREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
trainImg=cv2.imread('training_data/logo.jpg',0)
trainKp,trainDecs=detector.detectAndCompute(trainImg,None)

cam=cv2.VideoCapture(0)
while True:
    ret,QueryImgBGR=cam.read();
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY);
    queryKp,queryDecs=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDecs,trainDecs,k=2)
    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKp[m.trainIdx].pt)
            qp.append(queryKp[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainingBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainingBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        print(" match -%d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
        
    else:
        print("not enough match -%d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
    cv2.imshow('object_recognition',QueryImgBGR)
    cv2.waitKey(10)
