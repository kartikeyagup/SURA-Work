import cv2
import cv2.cv as cv
import numpy as np

writer = cv2.VideoWriter('test1.avi',cv.CV_FOURCC('P','I','M','1'),25,(640,480))
for i in range(1000):
    x = np.random.randint(255,size=(480,640)).astype('uint8')
    x = np.repeat(x,3,axis=1)
    x = x.reshape(480, 640, 3)
    writer.write(x)