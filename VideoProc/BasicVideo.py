import numpy as np
import cv2
import time
import csv
import copy
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


def SolveEqn((A,B,C),(D,E,F)):
	# Ax + By + Cz =0
	# Dx + Ey + Fz = 0
	print str(A)+"x+"+str(B)+"y+"+str(C)+"z=0"
	print str("D")+"x+"+str(E)+"y+"+str(F)+"z=0"
	t1 = A*E - B*D
	t2 = B*F - E*C 
	t3 = D*C - F*A 
	alpha = t2
	beta = t3
	k = (alpha**2 + beta**2 + t1**2) **0.5
	if k==0:
		return (0,0,0)
	else:
	 	return (alpha/k,beta/k,t1/k)
	# z is always positive

def GenerateEquation(R,P1,P2):
	# P1 is in the 1st point (y1, y2)
	# P2 is in the 2nd frame (y1 dash,y2 dash)
	# R is the 1*9 representation of rotation matrix
	coeffx =  R[2]*P1[1]*P2[0] + R[5]*P1[1]*P2[1] + R[8]*P1[1] - R[1] - R[4] - R[7]
	coeffy = -R[2]*P1[0]*P2[0] - R[5]*P1[0]*P2[1] - R[8]*P1[0] + R[0] + R[3] + R[6]
	coeffz =  R[1]*P1[0]*P2[0] + R[4]*P1[0]*P2[1] + R[7]*P1[0] - R[0]*P1[1]*P2[0] -R[3]*P1[1]*P2[1] - R[6]*P1[1]
	return (coeffx,coeffy,coeffz)	

def GetRelativeRotation(R1,R2):
	a= np.matrix([R1[0:3],R1[3:6],R1[6:9]])
	b= np.matrix([R2[0:3],R2[3:6],R2[6:9]])
	b = b.transpose()
	c= a*b
	return sum(c.tolist(),[])


def DoEverything(Rmat1,Rmat2,P1i,P2i,P1f,P2f):
	# Takes in the rotation matrices at 2 frames and the coordinates of 2 points in both frames
	# Returns the Translation vector (unit) between the 2 frames
	RMatRel=GetRelativeRotation(Rmat1, Rmat2)	
	eq1= GenerateEquation(RMatRel, P1i, P1f)
	eq2= GenerateEquation(RMatRel, P2i, P2f)
	return SolveEqn(eq1,eq2)

def AddTup(X,(d,e,f)):
	if X==None:
		return (d,e,f) 
	else:
		return (X[0]+d,X[1]+e,X[2]+f)


def EverythingFor3Points(Rmat1,Rmat2,P1i,P2i,P3i,P1f,P2f,P3f):
	validp=[]
	if P1i != None and P1f!= None:
		validp.append([P1i,P1f])
	if P2i != None and P2f!= None:
		validp.append([P2i,P2f])
	if P3i != None and P3f!= None:
		validp.append([P3i,P3f])
	sofar=None
	for i in xrange(len(validp)):
		for j in xrange(i+1,len(validp)):
			ans1=DoEverything(Rmat1, Rmat2, validp[i][0], validp[j][0], validp[i][1], validp[j][1])
			sofar = AddTup(sofar,ans1) 
	if sofar!=None:
		sofar=(sofar[0]/len(validp),sofar[1]/len(validp),sofar[2]/len(validp))
	return sofar

def ProcessList(l):
	#Takes in the list of [fnumber,time, rotmat,pred,pblue,pgreen]
	#Returns a list of time and instantaneous displacement vectors
	ans=[]
	for i in xrange(1,len(l)):
		f1=l[i-1]
		f2=l[i]
		ans.append([(f1[1]+f2[1])/2,EverythingFor3Points(f1[2],f2[2],f1[3],f1[4],f1[5],f2[3],f2[4],f2[5])])
	return ans

# print GetRelativeRotation([1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8])
# a1= [1,2,3,4,5,6,7,8,9]

# print a1[0:3]
# print np.matrix(a1[0:3],a1[3:6])

def find_car(image,type1):
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

	if (type1==0):
		imcons=red1
	elif (type1==1):
		imcons=blue1
	elif (type1==2):
		imcons=green1


	#find the car by looking for red, with high saturation
	blackret, black = cv2.threshold(red1, 128, 255, cv2.THRESH_BINARY)

	ret,red=cv2.threshold(imcons, 110, 255, cv2.THRESH_BINARY )
	ret,sat=cv2.threshold(sat, 128, 255, cv2.THRESH_BINARY )
	#AND the two thresholds, finding the car
	car=cv2.multiply(red, sat)

	black= cv2.multiply(black, sat)
	#remove noise, highlighting the car
	kernel = np.ones((2,2),np.uint8)
	kernelexp = np.ones((1,1),np.uint8)
	car=cv2.erode(car,kernel, iterations=3)
	car=cv2.dilate(car,kernelexp, iterations=1)
	# cv2.imshow("processed", cv2.flip(cv2.transpose(car),1))

	cv2.imshow('car',black)

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



# cap= cv2.VideoCapture(0)
# cap= cv2.VideoCapture("1433237712482vid.mp4")
# cap=cv2.VideoCapture("1433238177764vid.mp4")
# cap=cv2.VideoCapture("1433238967763vid.mp4")
# cap=cv2.VideoCapture("1433239750654vid.mp4")
# cap= cv2.VideoCapture("drop.avi")
# cap = cv2.VideoCapture("1433412822895vid.mp4")
# cap = cv2.VideoCapture("1433413418567vid.mp4")
cap = cv2.VideoCapture("1433480886878vid.mp4")
# cap = cv2.VideoCapture("1433493031044vid.mp4")

filename= '1433480886878SensorFusion3.csv'
# filename='1433493031044SensorFusion3.csv'

fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)
timed =map(lambda x: map(float,x),fileread[1:])


[timearr,r0,r1,r2,r3,r4,r5,r6,r7,r8,wr0,wr1,wr2,wr3,wr4,wr5,wr6,wr7,wr8,ax,ay,az,gx,gy,gz,gyx,gyy,gyz,mgx,mgy,mgz,imid,gp0,gp1,gp2,gp3]=map(list, zip(*timed))

totalframes = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
# print a1/a2

rotmat=list(map(list,zip(*[r0,r1,r2,r3,r4,r5,r6,r7,r8])))
avgtime=timearr[-1]/len(timearr)

print avgtime, fps, len(timearr)

def getnthindex(framenum,avgtime,fps):
	# Assuming that the frames are at sync from t=0 and the sensor recording goes further than expected
	return int((framenum/fps)/avgtime)

pointsred=[]
pointsgreen=[]
pointsblue=[]
rotmatricesframes=[]

t1=time.time()

fnumber=-1
ftoload=0
while(ftoload<100):
	ret, frame = cap.read()
	# frame=cv2.imread("photos/photo_"+str(ftoload)+".jpg")
	fnumber+=1	
	# ftoload+=1
	# print ret,fnumber
	if not(ret):
		break

	car_rect_red = find_car(frame,0)
	car_rect_blue = find_car(frame,1)
	car_rect_green = find_car(frame,2)

	middlred = middleblue= middlegreen= None
	# print car_rect, " is found"
	if (car_rect_red != (0,0,0,0)):
		middlered = (((car_rect_red[0] + car_rect_red[1] )/ 2), (car_rect_red[2] + car_rect_red[3])/2)
		pointsred.append([middlered,fnumber])
	if (car_rect_blue != (0,0,0,0)):
		middleblue = (((car_rect_blue[0] + car_rect_blue[1] )/ 2), (car_rect_blue[2] + car_rect_blue[3])/2)
		pointsblue.append([middleblue,fnumber])
	if (car_rect_green != (0,0,0,0)):
		middlegreen = (((car_rect_green[0] + car_rect_green[1] )/ 2), (car_rect_green[2] + car_rect_green[3])/2)
		pointsgreen.append([middlegreen,fnumber])
	
	rotmatricesframes.append([fnumber,(fnumber*1.0/fps),rotmat[getnthindex(fnumber,avgtime,fps)],middlered,middleblue,middlegreen])

	print fnumber,fnumber/fps,fnumber/(fps*avgtime)

	# if points == []:
	# else:
		# if abs(points[-1][0] - middle[0]) > 5 and abs(points[-1][1] - middle[1]) > 10:
			# points.append(middle)

	# cv2.rectangle(frame,(car_rect[0],car_rect[3]),(car_rect[1],car_rect[2]),(255,0,0),2)

	for point in pointsred:
		cv2.circle(frame, point[0], 3, (0, 0, 255),-1)
	for point in pointsgreen:
		cv2.circle(frame, point[0], 3, (0,255, 0),-1)
	for point in pointsblue:
		cv2.circle(frame, point[0], 3, (255, 0, 0),-1)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# print rotmatricesframes

print ProcessList(rotmatricesframes)

t2=time.time()
# print t2-t1

speedpointsr=[[0,0]]
speedpointsb=[[0,0]]
speedpointsg=[[0,0]]

pointsxr=[pointsred[0][0][0]]
pointsyr=[pointsred[0][0][1]]
framearrr=[pointsred[0][1]]

pointsxb= [pointsblue[0][0][0]]
pointsyb= [pointsblue[0][0][1]]
framearrb=[pointsblue[0][1]]

pointsxg=[pointsgreen[0][0][0]]
pointsyg=[pointsgreen[0][0][1]]
framearrg=[pointsgreen[0][1]]


[velxr,velyr] = map(list,zip(*speedpointsr))
[velxg,velyg] = map(list,zip(*speedpointsg))
[velxb,velyb] = map(list,zip(*speedpointsb))


for i in xrange(1,len(pointsred)):
	elem = pointsred[i][0]
	framearrr.append(pointsred[i][1])
	prev = pointsred[i-1][0]
	pointsxr.append(pointsred[i][0][0])
	pointsyr.append(pointsred[i][0][1])
	a=[elem[0]-prev[0],elem[1]-prev[1]]
	speedpointsr.append(a)

for i in xrange(1,len(pointsblue)):
	elem = pointsblue[i][0]
	framearrb.append(pointsblue[i][1])
	prev = pointsblue[i-1][0]
	pointsxb.append(pointsblue[i][0][0])
	pointsyb.append(pointsblue[i][0][1])
	a=[elem[0]-prev[0],elem[1]-prev[1]]
	speedpointsb.append(a)

for i in xrange(1,len(pointsgreen)):
	elem = pointsgreen[i][0]
	framearrg.append(pointsgreen[i][1])
	prev = pointsgreen[i-1][0]
	pointsxg.append(pointsgreen[i][0][0])
	pointsyg.append(pointsgreen[i][0][1])
	a=[elem[0]-prev[0],elem[1]-prev[1]]
	speedpointsg.append(a)




# print rotmat

# # print velx
# # print framearr
# # When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()

# plt.figure(0)
# plt.subplot(2,2,1)
# plt.plot(pointsxr)
# # plt.plot(pointsx)

# plt.subplot(2,2,2)
# plt.plot(pointsyr)

# plt.subplot(2,2,3)
# plt.plot(velxr)

# plt.subplot(2,2,4)
# plt.plot(velyr)

# plt.figure(1)
# plt.subplot(2,2,1)
# plt.plot(pointsxg)
# # plt.plot(pointsx)

# plt.subplot(2,2,2)
# plt.plot(pointsyg)

# plt.subplot(2,2,3)
# plt.plot(velxg)

# plt.subplot(2,2,4)
# plt.plot(velyg)


# plt.figure(2)
# plt.subplot(2,2,1)
# plt.plot(pointsxb)
# # plt.plot(pointsx)

# plt.subplot(2,2,2)
# plt.plot(pointsyb)

# plt.subplot(2,2,3)
# plt.plot(velxb)

# plt.subplot(2,2,4)
# plt.plot(velyb)

# plt.show()