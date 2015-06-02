import numpy as np
import cv2

def GetSum(image,type,i):
	s1=image.shape[1::-1]
	if type==0: #column
		ans=0
		for j in xrange(0,s1[1]):
			ans += image[j][i]
	else:
		ans=0
		for j in xrange(0,s1[0]):
			ans += image[i][j]
	return ans 

def find_car(image):
	""" Finds red blob (hopefully only one, the rc car) in an image
	"""
	size = image.shape[1::-1]
	# print size
	# cv2.crea
	#prepare memory
	# car = cv2.CreateImage(size, 8, 1)
	# red = cv2.CreateImage(size, 8, 1)
	# hsv = cv2.CreateImage(size, 8, 3)
	# sat = cv2.CreateImage(size, 8, 1)

	#split image into hsv, grab the sat
	hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	(hue,sat,val1)= cv2.split(hsv)
	#split image into rgb
	(blue1,green1,red1)=cv2.split(image)

	#find the car by looking for red, with high saturation
	ret,red=cv2.threshold(red1, 128, 255, cv2.THRESH_BINARY )
	ret,sat=cv2.threshold(sat, 128, 255, cv2.THRESH_BINARY )

	#AND the two thresholds, finding the car
	car=cv2.multiply(red, sat)

	#remove noise, highlighting the car
	kernel = np.ones((5,5),np.uint8)
	car=cv2.erode(car,kernel, iterations=2)
	car=cv2.dilate(car,kernel, iterations=5)
	# cv2.imshow('car',car)
	# cv2.waitKey(10000)

	# leftmost=0
	# rightmost=0
	# topmost=0
	# bottommost=0
	# temp=0
	# for i in range(size[0]):
	# 	if not(GetSum(car,0,i)==0.0):
	# 		rightmost=i
	# 		if temp==0:
	# 			leftmost=i
	# 			temp=1		
	# for i in range(size[1]):
	# 	if not(GetSum(car,1,i)==0.0):
	# 		bottommost=i
	# 		if temp==1:
	# 			topmost=i
	# 			temp=2	
	# print (leftmost,rightmost,topmost,bottommost)


	#return a bounding box
	image1, contours = cv2.findContours(car, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	# print image1, "contours"
	# cv.ShowImage('A', car)

	if len(image1)==0:
		return(0, 0, 0, 0)
	else:
		# print image1
		areas = [cv2.contourArea(c) for c in image1]
		max_index = np.argmax(areas)
		cnt=image1[max_index]
	
		x,y,h,w= cv2.boundingRect(cnt)
		return (x,x+w,y+h,y)
		# return(0,0,0,0)

	# x,y,w,h= cv2.boundingRect(car)
	# print x,y,w,h
	# return (leftmost,rightmost,topmost,bottommost)
	# return cv2.rectangle(image, (leftmost,bottommost), (rightmost,topmost), (0,255,0))



cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('samplevideo.mp4')
points=[]

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# print ret

	car_rect = find_car(frame)
	# print car_rect, " is found"

	middle = (((car_rect[0] + car_rect[1] )/ 2), (car_rect[2] + car_rect[3])/2)
	# if points == []:
	points.append(middle)
	# else:
		# if abs(points[-1][0] - middle[0]) > 5 and abs(points[-1][1] - middle[1]) > 10:
			# points.append(middle)

	cv2.rectangle(frame,(car_rect[0],car_rect[3]),(car_rect[1],car_rect[2]),(255,0,0),2)

	for point in points:
		cv2.circle(frame, point, 3, (0, 0, 255),-1)

	# cv.WriteFrame(writer, original)

	# cv.ShowImage('Analysed', frame)




	# Our operations on the frame come here
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	if (ret):
		cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()