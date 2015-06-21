import numpy as np
import cv2

# cap = cv2.VideoCapture('slow.flv')
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,1920);
print cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,1440);
# cap = cv2.VideoCapture('../../../testvideo.mp4')

g,p = cap.read()


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
                       qualityLevel = 0.01,
                       minDistance = 1,
                       blockSize = 300 )

print feature_params

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

print lk_params

# Create some random c1olors
color = np.random.randint(0,255,(10000,3))

# a=cv2.cv.CreateImage((500,500), cv2.CV_8UC1,1)
# a=np.zeros(p)

xc, yc, r = 348, 317, 45
# size of the image
print p.shape
H, W,c  = p.shape
# x and y coordinates per every pixel of the image
x, y = np.meshgrid(np.arange(W), np.arange(H))
# squared distance from the center of the circle
d2 = (x - xc)**2 + (y - yc)**2
# mask is True inside of the circle
mask = d2 < r**2

d2 = np.zeros(p.shape)

print d2.dtype

# cv2.imshow("winname", a)
k = cv2.waitKey(30) & 0xff
if k==112:
    print st
# elif k == 27:

print cv2.CV_8UC1

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = d2, **feature_params)

print len(p0)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # p1, st, err = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    if len(good_old)<=5:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
        continue

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    added=cv2.add(frame,mask)

    cv2.imshow('frame',added)
    k = cv2.waitKey(30) & 0xff
    if k==112:
        print st
    elif k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    # print good_new
    p0 = good_new.reshape(-1,1,2)
    # print p0

cv2.destroyAllWindows()
cap.release()