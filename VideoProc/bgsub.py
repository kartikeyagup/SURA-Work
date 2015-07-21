import numpy as np
import cv2

#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>


cap = cv2.VideoCapture(0)

fgbg = cv2.BackgroundSubtractorMOG2()
# fgbg.nmixtures=3


while(1):
    ret, frame = cap.read()
    fgbg.getBackgroundImage()
    frame=fgbg.apply(frame)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.BackgroundSubtractorGMG()

# while(1):
#     ret, frame = cap.read()

#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()