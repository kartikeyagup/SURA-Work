import numpy as np 
import cv2
import random

height=720
width=1280
# out = cv2.VideoWriter('output.avi',cv2.cv.FOURCC('X','V','I','D'), 30.0, (1280,720))


blank_image = np.zeros((height,width,3), np.uint8)

# # cv2.rectangle(blank_image, (10,10), (100,100), (0,0,255))

pointspos=[0]*5
vel=[0]*5
for i in xrange(5):
	pointspos[i]=(random.randint(0,width),random.randint(0,height))
	vel[i]=(random.randint(-10,10),random.randint(-10,10))
	# cv2.circle(blank_image, pointspos[i], 4, (0,255,0))

previd=0
for j in xrange(1000):
	# print j
	for i in xrange(5):
		if i==0:
			cv2.circle(blank_image,pointspos[i],4,(255,0,0),-1)
		elif i==1:
			cv2.circle(blank_image,pointspos[i],4,(0,255,0),-1)
		elif i==2:
			cv2.circle(blank_image,pointspos[i],4,(0,0,255),-1)
		elif i==3:
			cv2.circle(blank_image,pointspos[i],4,(100,100,100),-1)
		elif i==4:
			cv2.circle(blank_image,pointspos[i],4,(120,20,30),-1)
		if (pointspos[i][0]+vel[i][0]>=width):
			vel[i]=(-vel[i][0],vel[i][1])
		if (pointspos[i][0]+vel[i][0]<=0):
			vel[i]=(-vel[i][0],vel[i][1])
		if (pointspos[i][1]+vel[i][1]>=height):
			vel[i]=(vel[i][0],-vel[i][1])
		if (pointspos[i][1]+vel[i][1]<=0):
			vel[i]=(vel[i][0],-vel[i][1])

		pointspos[i]=((pointspos[i][0]+vel[i][0])%width,(pointspos[i][1]+vel[i][1])%height)

	cv2.imshow('frame',blank_image)
        cv2.imwrite('photos/photo_'+str(previd) + '.jpg', blank_image)
        previd+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

	blank_image=np.zeros((height,width,3),np.uint8)
	# out.write(blank_image)

# out.release()

# cv2.imshow("winname",blank_image)
# cv2.waitKey(0)

# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC(*'DIVX')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480),1)

# previd=0

# while(cap.isOpened()):
#     ret, frame = cap.read()
    
#     if ret==True:
#         # frame = (frame,0

#         # write the flipped frame
#         # out.write(frame)

#         cv2.imshow('frame',frame)
#         cv2.imwrite('photos/photo_'+str(previd) + '.jpg', frame)
#         previd+=1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# Release everything if job is finished
# out.release()
# cap.release()
cv2.destroyAllWindows()