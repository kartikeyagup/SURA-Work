import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap = cv2.VideoCapture(0)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


objoverall =[]
imgoverall =[]

# images = glob.glob('*.jpg')
print "here"

ret1,frame = cap.read()

while (1):
    # print "in loop"
    # img = cv2.imread(fname)
    ret1,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)


        # print corners

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners,ret)
        cv2.imshow('imgchess',img)
        k= cv2.waitKey(50) & 0xff
        if k==27:
            break
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # print ret,dist
        print mtx
        # h,  w = img.shape[:2]
        # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        # print newcameramtx
        # objoverall.append(objpointsbj)
        # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # # crop the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        # cv2.imshow('imgcorrected', dst)
        # k= cv2.waitKey(50) & 0xff
        # if k==27:
        #     break
        
    else:
        cv2.imshow('imgnochess',img)
        k= cv2.waitKey(50) & 0xff
        if k==27:
            break
        
# [[ 495.87443844    0.          313.5218237 ]
#  [   0.          473.16996648  250.00670925]
#  [   0.            0.            1.        ]]

cv2.destroyAllWindows()
print "destroyed"