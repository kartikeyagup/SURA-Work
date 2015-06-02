import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

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
	# kernel = np.ones((2,2),np.uint8)
	# car=cv2.erode(car,kernel, iterations=2)
	# car=cv2.dilate(car,kernel, iterations=2)
	cv2.imshow("processed", cv2.flip(cv2.transpose(car),1))
	# cv2.imshow('car',car)

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
		# print cnt
		# print areas
		# cv2.waitKey(100)
		x,y,h,w= cv2.boundingRect(cnt)
		return (x,x+w,y+h,y)
		# return(0,0,0,0)

	# x,y,w,h= cv2.boundingRect(car)
	# print x,y,w,h
	# return (leftmost,rightmost,topmost,bottommost)
	# return cv2.rectangle(image, (leftmost,bottommost), (rightmost,topmost), (0,255,0))



cap = cv2.VideoCapture(0)
# cap= cv2.VideoCapture("1433237712482vid.mp4")
# cap=cv2.VideoCapture("1433238177764vid.mp4")
# cap=cv2.VideoCapture("1433238967763vid.mp4")
cap=cv2.VideoCapture("1433239750654vid.mp4")
# cap= cv2.VideoCapture("drop.avi")

points=[]

# while True:
#     flag, frame = cap.read()
#     print flag
#     if flag == 0:
#         # break
#         x=5
#     else:
#     	cv2.imshow("Video", frame)
#     	key_pressed = cv2.waitKey(10)    #Escape to exit
#     	if key_pressed == 27:
#     	    break
t1=time.time()

fnumber=-1
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	fnumber+=1
	# print ret
	if not(ret):
		break

	car_rect = find_car(frame)
	# print car_rect, " is found"
	if (car_rect==(0,0,0,0)):
		continue

	middle = (((car_rect[0] + car_rect[1] )/ 2), (car_rect[2] + car_rect[3])/2)
	# if points == []:
	points.append([middle,fnumber])
	# else:
		# if abs(points[-1][0] - middle[0]) > 5 and abs(points[-1][1] - middle[1]) > 10:
			# points.append(middle)

	cv2.rectangle(frame,(car_rect[0],car_rect[3]),(car_rect[1],car_rect[2]),(255,0,0),2)

	for point in points:
		cv2.circle(frame, point[0], 3, (0, 0, 255),-1)

	# cv.WriteFrame(writer, original)

	# cv.ShowImage('Analysed', frame)
	# Our operations on the frame come here
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	if (ret):
		cv2.imshow('frame', cv2.flip(cv2.transpose(frame),1))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

t2=time.time()
print t2-t1

speedpoints=[[0,0]]
pointsx=[points[0][0][0]]
pointsy=[points[0][0][1]]
framearr=[points[0][1]]
for i in xrange(1,len(points)):
	elem = points[i][0]
	framearr.append(points[i][1])
	prev = points[i-1][0]
	pointsx.append(points[i][0][0])
	pointsy.append(points[i][0][1])
	a=[elem[0]-prev[0],elem[1]-prev[1]]
	speedpoints.append(a)
# print speedpoints

[velx,vely] = map(list,zip(*speedpoints))

# print velx
# print framearr
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

plt.figure(0)
plt.subplot(2,2,1)
plt.plot(pointsx)
# plt.plot(pointsx)

plt.subplot(2,2,2)
plt.plot(pointsy)

plt.subplot(2,2,3)
plt.plot(velx)

plt.subplot(2,2,4)
plt.plot(vely)

plt.show()