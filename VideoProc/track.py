import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# take first frame of the video
ret,frame = cap.read()

features = cv2.goodFeaturesToTrack(cv2.cvtColor(frame,cv2.cv.CV_RGB2GRAY), 5, .01, 150)
features = features.reshape((-1, 2))

# setup initial location of window
# track_window = (c,r,w,h)

trackingpoints=[]
# roi=[]
# hsv_roi=[]
# mask=[]
roi_hist=[]
for x,y in features:
    r,h,c,w = x,y,10,10  # simply hardcoded the values
    trackingpoints.append((x,y,10,10))
    # roi.append(frame[r:r+h, c:c+w])
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist1 = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist1,roi_hist1,0,255,cv2.NORM_MINMAX)
    roi_hist.append(roi_hist1)


# set up the ROI for tracking

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:

        # apply meanshift to get the new location
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i in xrange(len(trackingpoints)):
            dst = cv2.calcBackProject([hsv],[0],roi_hist[i],[0,180],1)
            ret, trackingpoints[i]  = cv2.meanShift(dst, trackingpoints[i], term_crit)
            x,y,w,h = trackingpoints[i]
            cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        # Draw it on image
        # print x,y,w,h
        # print frame.shape[1::-1]
        # print features
        # for x, y in features:
            # cv2.circle(frame, (x, y), 10, (0, 0, 255))
        # print img2.shape[1::-1]
        cv2.imshow('img2',frame)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        # else:
            # cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()