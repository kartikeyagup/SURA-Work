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
	# print str(A)+"x+"+str(B)+"y+"+str(C)+"z=0"
	# print str(D)+"x+"+str(E)+"y+"+str(F)+"z=0"
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

def GetDotProduct(a,b):
	# TODO: fix for 0,0,0 and 0,0,0
	# TODO: take None into account as well
	if (a==None) or (b==None):
		return -1
	else:
		return abs(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])

def ConvertVelocity((vx,vy,vz)):
	# Normalises the velocity to give its unit vector
	balance=(vx**2 + vy**2 + vz**2)**0.5
	if (balance==0):
		return (0,0,0)
	else:
		return (vx/balance,vy/balance,vz/balance)

#TODO: Match the directions of velocity for the motion and rest zones
#TODO: Plot graph between the angle

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


# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("1433237712482vid.mp4")
# cap = cv2.VideoCapture("1433238177764vid.mp4")
# cap = cv2.VideoCapture("1433238967763vid.mp4")
# cap = cv2.VideoCapture("1433239750654vid.mp4")
# cap = cv2.VideoCapture("drop.avi")
# cap = cv2.VideoCapture("1433412822895vid.mp4")
# cap = cv2.VideoCapture("1433413418567vid.mp4")
cap = cv2.VideoCapture("1433480886878vid.mp4")
# cap = cv2.VideoCapture("1433493031044vid.mp4")

filename = '1433480886878SensorFusion3.csv'
# filename ='1433493031044SensorFusion3.csv'

fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)
timed =map(lambda x: map(float,x),fileread[1:])


[timearr,r0,r1,r2,r3,r4,r5,r6,r7,r8,wr0,wr1,wr2,wr3,wr4,wr5,wr6,wr7,wr8,ax,ay,az,gx,gy,gz,gyx,gyy,gyz,mgx,mgy,mgz,imid,gp0,gp1,gp2,gp3]=map(list, zip(*timed))

filename = '1433480886878SensorFusion3data.csv'
fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

def funcConv(x):
	return x=='true'

timed =map(lambda x: map(float,x[:-9] + x[-6:]) + map(funcConv, x[-9:-6]),fileread[1:])
[sno,timearr1,imid,ax1,ay1,az1,vx,vy,vz,dx,dy,dz,rawax,raway,rawaz,pratx,praty,pratz,motionx,motiony,motionz]=map(list, zip(*timed))


def GetMotionZones(boolarr):
	# print boolarr
	ans=[]
	state=False
	prevval=0
	for i in xrange(len(boolarr)):
		if boolarr[i]:
			if not(state):
				state=True
				prevval=i
		else:
			if state:
				state=False
				ans.append([prevval,i])
	if state:
		ans.append([prevval,len(boolarr)-1])
	return ans

def GetVelocity(timearr,motionzonearr,acc,type):
	ans=[0.0]*len(acc)
	for elem in motionzonearr:
		totalacc=0.0
		prevtime=timearr[elem[0]]
		totaltime=timearr[elem[1]]-timearr[elem[0]]
		for j in xrange(elem[0],elem[1]):
			if (j>0):
				x= acc[j]*(timearr[j]-prevtime)
				totalacc +=x
				ans[j]= ans[j-1]+ x
				prevtime=timearr[j]
		prevtime =timearr[elem[0]]
		if (type==0):
			corrfact=totalacc/totaltime
			for j in xrange(elem[0],elem[1]):
				ans[j] -= corrfact
		elif (type==1):
			finalv= ans[elem[1]-1]
			m=finalv/totaltime
			for j in xrange(elem[0],elem[1]):
				ans[j] -= m*(timearr[j]-prevtime)
		elif (type==2):
			finalv= ans[elem[1]-1]
			m = 2*finalv/(totaltime**2)
			for j in xrange (elem[0],elem[1]):
				ans[j] -= 0.5*m*((timearr[j]-prevtime)**2)
	return ans

def FixVelocity(acc,timearr,motionzonearr):
	ans=[0.0]*len(acc)
	for elem in motionzonearr:
		totalacc=0.0
		prevtime=timearr[elem[0]]
		totaltime=timearr[elem[1]]-timearr[elem[0]]
		for j in xrange(elem[0],elem[1]):
			if (j>0):
				x= acc[j]*(timearr[j]-prevtime)
				totalacc +=x
				ans[j]= ans[j-1]+ x
				prevtime=timearr[j]
		prevtime =timearr[elem[0]]
		# ans2= copy.deepcopy(ans)
		# finalv= ans2[elem[1]-1]
		# m = 2*finalv/(totaltime**2)
		# for j in xrange (elem[0],elem[1]):
		# 	ans2[j] -= 0.5*m*((timearr[j]-prevtime)**2)
		# totalvel=0.0
		# print totalvel
		# for i in xrange(elem[0],elem[1]):
		# 	# print ans[i]
		# 	totalvel +=ans2[i]
		# print totalvel
		# if (totalvel>0):
		# 	if (totalacc>0):
		# 		print "upward graph extension needed"
		# 		# # Final speed positive and upward graph
		# 		# prevslope=(ans[elem[1]-1]-ans[elem[1]-10])/9
		# 		# prevslope= - abs(prevslope)
		# 		# curr=elem[1]
		# 		# while (curr<len(timearr)):
		# 		# 	ans[curr] = ans[curr-1]+prevslope
		# 		# 	if (ans[curr]<=0):
		# 		# 		ans[curr]=0.0 
		# 		# 		break;
		# 		# 	curr+=1
		# 	elif (totalacc<0):
		# 		print "upward graph truncation"
		# 		curr=elem[1] -1 
		# 		while (ans[curr]<0):
		# 			ans[curr] = -ans[curr]
		# 			curr -=1
		# 		finalv = ans[elem[1]-1]
		# 		m=finalv/(timearr[elem[1]]-timearr[curr])
		# 		for j in xrange(curr,elem[1]):
		# 			ans[j] -= m*(timearr[j]-timearr[curr])
		# elif (totalvel<0):
		# 	if (totalacc>0):
		# 		# Final speed positive and downward
		# 		print "downward graph truncation"
		# 		prevslope=(ans[elem[1]-1]-ans[elem[1]-5])/4
		# 		curr= elem[1] -1 
		# 		while (ans[curr]>0):
		# 			ans[curr]= -ans[curr]
		# 			curr -=1
		# 		finalv = ans[elem[1]-1]
		# 		m=finalv/(timearr[elem[1]]-timearr[curr])
		# 		for j in xrange(curr,elem[1]):
		# 			ans[j] -= m*(timearr[j]-timearr[curr])
		# 	elif (totalacc<0):
		# 		print "downward graph extension"
		# 		# prevslope=(ans[elem[1]-1]-ans[elem[1]-10])/9
		# 		# curr=elem[1]
		# 		# prevslope=abs(prevslope)
		# 		# while (curr<len(timearr)):
		# 		# 	ans[curr] = ans[curr-1]+prevslope
		# 		# 	if (ans[curr]>=0):
		# 		# 		ans[curr]=0.0 
		# 		# 		break;
		# 		# 	curr +=1
	return ans

def GetDistance(timearr,velarray):
	ans=[0.0]*len(timearr)
	prevtime=timearr[0]
	for i in xrange(1,len(timearr)):
		ans[i]=ans[i-1] + velarray[i]*(timearr[i]-prevtime)
		prevtime = timearr[i]
	# print "distance is: ", ans[-1]
	return ans




calcvx1=GetVelocity(timearr1,GetMotionZones(motionx),ax,1)
calcdx1=GetDistance(timearr1,calcvx1)
calcvx0=GetVelocity(timearr1,GetMotionZones(motionx),ax,0)
calcdx0=GetDistance(timearr1,calcvx0)

calcvy1=GetVelocity(timearr1,GetMotionZones(motiony),ay,1)
calcdy1=GetDistance(timearr1,calcvy1)
calcvy0=GetVelocity(timearr1,GetMotionZones(motiony),ay,0)
calcdy0=GetDistance(timearr1,calcvy0)

calcvz1=GetVelocity(timearr1,GetMotionZones(motionz),az,1)
calcdz1=GetDistance(timearr1,calcvz1)
calcvz0=GetVelocity(timearr1,GetMotionZones(motionz),az,0)
calcdz0=GetDistance(timearr1,calcvz0)


soccervx2=GetVelocity(timearr1,GetMotionZones(motionx),pratx,2)
soccerdx2=GetDistance(timearr1,soccervx2)
soccervx1=GetVelocity(timearr1,GetMotionZones(motionx),pratx,1)
soccerdx1=GetDistance(timearr1,soccervx1)
soccervx0=GetVelocity(timearr1,GetMotionZones(motionx),pratx,0)
soccerdx0=GetDistance(timearr1,soccervx0)

soccervy2=GetVelocity(timearr1,GetMotionZones(motiony),praty,2)
soccerdy2=GetDistance(timearr1,soccervy2)
soccervy1=GetVelocity(timearr1,GetMotionZones(motiony),praty,1)
soccerdy1=GetDistance(timearr1,soccervy1)
soccervy0=GetVelocity(timearr1,GetMotionZones(motiony),praty,0)
soccerdy0=GetDistance(timearr1,soccervy0)

soccervz2=GetVelocity(timearr1,GetMotionZones(motionz),pratz,2)
soccerdz2=GetDistance(timearr1,soccervz2)
soccervz1=GetVelocity(timearr1,GetMotionZones(motionz),pratz,1)
soccerdz1=GetDistance(timearr1,soccervz1)
soccervz0=GetVelocity(timearr1,GetMotionZones(motionz),pratz,0)
soccerdz0=GetDistance(timearr1,soccervz0)

fixedvx = FixVelocity(ax,timearr1, GetMotionZones(motionx))
fixedvy = FixVelocity(ay,timearr1, GetMotionZones(motiony))
fixedvz = FixVelocity(az,timearr1, GetMotionZones(motionz))
fixeddx = GetDistance(timearr1,fixedvx)
fixeddy = GetDistance(timearr1,fixedvy)
fixeddz = GetDistance(timearr1,fixedvz)

kgv2 = map(list, zip(*[vx,vy,vz]))
kgv1 = map(list, zip(*[calcvx1,calcvy1,calcvz1]))
kgv0 = map(list, zip(*[calcvx0,calcvy0,calcvz0]))

soccerv2 = map(list, zip(*[soccervx2,soccervy2,soccervz2]))
soccerv1 = map(list, zip(*[soccervx1,soccervy1,soccervz1]))
soccerv0 = map(list, zip(*[soccervx0,soccervy0,soccervz0]))


totalframes = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

rotmat=list(map(list,zip(*[r0,r1,r2,r3,r4,r5,r6,r7,r8])))
avgtime=timearr[-1]/len(timearr)

print avgtime, fps, len(timearr)

def getnthindex(framenum,avgtime,fps):
	# Assuming that the frames are at sync from t=0 and the sensor recording goes further than expected
	return int((framenum/fps)/avgtime)

def getnthindextime(timeframe,avgtime):
	return int(timeframe/avgtime)

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

	# print fnumber,fnumber/fps,fnumber/(fps*avgtime)

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

cv2.destroyAllWindows()

def GetVelList(processedflist,velocityarr,avgtimeofsig):
	a = [0]*len(processedflist)
	for i in xrange(len(processedflist)):
		a[i]=velocityarr[getnthindextime(processedflist[i][0],avgtimeofsig)]
	return a

processedlist= ProcessList(rotmatricesframes)
# print processedlist[0:50]

kg0vellist= map(ConvertVelocity,GetVelList(processedlist, kgv0, avgtime))
kg1vellist= map(ConvertVelocity,GetVelList(processedlist, kgv1, avgtime))
kg2vellist= map(ConvertVelocity,GetVelList(processedlist, kgv2, avgtime))

soccer0vellist= map(ConvertVelocity,GetVelList(processedlist, soccerv0, avgtime))
soccer1vellist= map(ConvertVelocity,GetVelList(processedlist, soccerv1, avgtime))
soccer2vellist= map(ConvertVelocity,GetVelList(processedlist, soccerv2, avgtime))

def MappingOverVelocities(proframedata,procveldata):
	ans=[0]*len(proframedata)
	for i in xrange(len(ans)):
		# print i, proframedata[i],procveldata[i]
		ans[i]=[proframedata[i][0],GetDotProduct(proframedata[i][1],procveldata[i])]
	return ans

mappedkg0 = MappingOverVelocities(processedlist, kg0vellist)
mappedkg1 = MappingOverVelocities(processedlist, kg1vellist)
mappedkg2 = MappingOverVelocities(processedlist, kg2vellist)

mappedsoccer0 = MappingOverVelocities(processedlist, soccer0vellist)
mappedsoccer1 = MappingOverVelocities(processedlist, soccer1vellist)
mappedsoccer2 = MappingOverVelocities(processedlist, soccer2vellist)

plt.figure(0)
plt.subplot(3,2,1)
plt.plot(map(lambda x: x[0],mappedkg0),map(lambda x: x[1],mappedkg0))

plt.subplot(3,2,2)
plt.plot(map(lambda x: x[0],mappedkg1),map(lambda x: x[1],mappedkg1))

plt.subplot(3,2,3)
plt.plot(map(lambda x: x[0],mappedkg2),map(lambda x: x[1],mappedkg2))

plt.subplot(3,2,4)
plt.plot(map(lambda x: x[0],mappedsoccer0),map(lambda x: x[1],mappedsoccer0))

plt.subplot(3,2,5)
plt.plot(map(lambda x: x[0],mappedsoccer1),map(lambda x: x[1],mappedsoccer1))

plt.subplot(3,2,6)
plt.plot(map(lambda x: x[0],mappedsoccer2),map(lambda x: x[1],mappedsoccer2))


plt.show()



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