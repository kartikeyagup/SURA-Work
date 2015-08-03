import numpy as np
import cv2
import time
import csv
import copy
import matplotlib.pyplot as plt

fnumber='1436275811192'
fnumber='1436350749688'
fnumber='1436435833417'
fnumber='1436509344762'
# fnumber='1437395006225'
# fnumber='1437460621901'
# fnumber='1437460593277'
# fnumber='1437469487121'
# fnumber='1437473687034'
fnumber='1437481365601'
fnumber='1437486540228'
fnumber='1437643001347'

filenamedataprocessed = fnumber+'SensorFusion3data.csv'
filenamedataraw = fnumber+ 'SensorFusion3.csv'
filenameframemap = fnumber + 'framemapping.csv'
filenametracking = fnumber+ '_track.csv'

fileread=[]
with open(filenamedataraw,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)
timed =map(lambda x: map(float,x),fileread[1:])

[timearr,r0,r1,r2,r3,r4,r5,r6,r7,r8,wr0,wr1,wr2,wr3,wr4,wr5,wr6,wr7,wr8,ax1,ay1,az1,gx,gy,gz,gyx,gyy,gyz,mgx,mgy,mgz,imid,gp0,gp1,gp2,gp3]=map(list, zip(*timed))

framesensorread=[]
with open(filenameframemap,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		framesensorread.append(row)

framesensor = map(lambda x: [int(x[0]),int(x[1])], framesensorread[1:])

rotmatrices = map(list, zip(r0,r1,r2,r3,r4,r5,r6,r7,r8))
# print rotmatrices
mappeddata=[]
with open(filenametracking,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		mappeddata.append(map(float,row))

fileread=[]
with open(filenamedataprocessed,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

def funcConv(x):
	return x=='true'

timed =map(lambda x: map(float,x[:-9] + x[-6:]) + map(funcConv, x[-9:-6]),fileread[1:])
[sno,timearr1,imid,ax,ay,az,vx,vy,vz,dx,dy,dz,rawax,raway,rawaz,pratx,praty,pratz,motionx,motiony,motionz]=map(list, zip(*timed))


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


def breakupintopoints(r1):
	ans=[]
	for i in xrange(len(r1)/2):
		ans.append((r1[2*i],r1[1+ 2*i]))
	return ans

def breakintopointpairs(bigrow):
	bigans=[]
	for i in xrange(len(bigrow)/2):
		bigans.append([breakupintopoints(bigrow[2*i]),breakupintopoints(bigrow[1+2*i])])
	return bigans

def MakeTx(t):
	a=[[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]
	return np.asarray(a)

def GetEssentialMatrix(r,t):
	return np.mat(MakeTx(t))*np.mat(r)

def MakeRMatrix(r):
	if r == [0,0,0,0,0,0,0,0,0]:
		r = [1,0,0,0,1,0,0,0,1]
	c1= [[r[0],r[1],r[2]],[r[3],r[4],r[5]],[r[6],r[7],r[8]]]
	c1 =[[-r[1],-r[0],-r[2]],[-r[4],-r[3],-r[5]],[-r[7],-r[6],-r[8]]]
	c1= [[r[3],r[4],r[5]],[-r[0],-r[1],-r[2]],[r[6],r[7],r[8]]]
	return c1

def normalize(a):
	normval = (a[0]**2 + a[1]**2 + a[2]**2)**0.5
	if normval==0:
		return [0,0,0]
	else:
		return [a[0]/normval,a[1]/normval,a[2]/normval]

def GetCompleteEMatrix(r1,t1,r2,t2):
	tnet= np.asarray(t2)-np.asarray(t1)
	tnet=normalize(tnet)
	# print r1,r2
	if r1 == [0,0,0,0,0,0,0,0,0]:
			r1 =[1,0,0,0,1,0,0,0,1]
	rnet= np.dot(np.linalg.inv(np.asarray(MakeRMatrix(r1))),np.asarray(MakeRMatrix(r2)))
	# print tnet,rnet
	# print np.linalg.det(np.linalg.inv(np.asarray(MakeRMatrix(r1)))),np.linalg.det(np.asarray(MakeRMatrix(r1))),np.linalg.det(np.asarray(MakeRMatrix(r2))),np.linalg.det(rnet)
	return np.asarray(GetEssentialMatrix(rnet,tnet))

def GetCorresSensorNumber(fno):
	return framesensor[fno][1]

def GetSensorNumbers():
	return map(lambda x: x[1], framesensor)

# for i in xrange(len(ax)):
# 	ax[i]=1000*ax[i]

# for i in xrange(len(ay)):
# 	ay[i]=1000*ay[i]

# for i in xrange(len(az)):
# 	az[i]=1000*az[i]

# for i in xrange(len(pratx)):
# 	pratx[i]=1000*pratx[i]

# for i in xrange(len(praty)):
# 	praty[i]=1000*praty[i]

# for i in xrange(len(pratz)):
# 	pratz[i]=1000*pratz[i]

# for i in xrange(len(vx)):
# 	vx[i]=1000*vx[i]
# 	vy[i]=1000*vy[i]
# 	vz[i]=1000*vz[i]
# 	dx[i]=1000*dx[i]
# 	dy[i]=1000*dy[i]
# 	dz[i]=1000*dz[i]

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

Kd0= map(list, zip(*[calcdx0,calcdy0,calcdz0]))
Kd1= map(list, zip(*[calcdx1,calcdy1,calcdz1]))
Kd2= map(list, zip(*[dx,dy,dz]))

Sd0= map(list, zip(*[soccerdx0,soccerdy0,soccerdz0]))
Sd1= map(list, zip(*[soccerdx1,soccerdy1,soccerdz1]))
Sd2= map(list, zip(*[soccerdx2,soccerdy2,soccerdz2]))

sensornumbers= GetSensorNumbers()
RT_KG1 = []
RT_Soccer1 = []
RT_KG0 = []
RT_Soccer0 = []
RT_KG2 = []
RT_Soccer2 = []

for elem in sensornumbers:
	# print elem, len(rotmatrices)
	Rot = rotmatrices[elem]
	RT_KG0.append([Rot,Kd0[elem]])
	RT_KG1.append([Rot,Kd1[elem]])
	RT_KG2.append([Rot,Kd2[elem]])
	RT_Soccer0.append([Rot,Sd0[elem]])
	RT_Soccer1.append([Rot,Sd1[elem]])
	RT_Soccer2.append([Rot,Sd2[elem]])

print "Done with sensor processing"

# EKG0=[]
# EKG1=[]
# EKG2=[]
# ES0=[]
# ES1=[]
# ES2=[]

# for i in xrange(1,len(RT_KG0)):
# 	# EKG0.append(GetCompleteEMatrix(RT_KG0[i-1][0],RT_KG0[i-1][1],RT_KG0[i][0],RT_KG0[i][1]))
# 	EKG1.append(GetCompleteEMatrix(RT_KG1[i-1][0],RT_KG1[i-1][1],RT_KG1[i][0],RT_KG1[i][1]))
# 	# EKG2.append(GetCompleteEMatrix(RT_KG2[i-1][0],RT_KG2[i-1][1],RT_KG2[i][0],RT_KG2[i][1]))
# 	# ES0.append(GetCompleteEMatrix(RT_Soccer0[i-1][0],RT_Soccer0[i-1][1],RT_Soccer0[i][0],RT_Soccer0[i][1]))
# 	# ES1.append(GetCompleteEMatrix(RT_Soccer1[i-1][0],RT_Soccer1[i-1][1],RT_Soccer1[i][0],RT_Soccer1[i][1]))
# 	# ES2.append(GetCompleteEMatrix(RT_Soccer2[i-1][0],RT_Soccer2[i-1][1],RT_Soccer2[i][0],RT_Soccer2[i][1]))

# print "Number of frames", len(RT_KG1)

K=[[  1.15137655e+03,   0.00000000e+00,   6.35646935e+02], [  0.00000000e+00,   1.14984595e+03,   3.36169128e+02], [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
K[0][2]=0
K[1][2]=0
K= np.asarray(K)
Kinv = np.linalg.inv(K)


def ObtainFundament(r1,r2,t1,t2):
	KinvT = Kinv.T
	rnet= np.dot(np.linalg.inv(np.asarray(MakeRMatrix(r2))),(np.asarray(MakeRMatrix(r1))))
	rentT = rnet.T
	# print "value1",np.dot(rnet,rentT)
	tnet = np.asarray(t1)-np.asarray(t2)
	# tnet= np.asarray(normalize(tnet))
	tnet = np.dot(np.linalg.inv(np.asarray(MakeRMatrix(r2))),tnet.T)
	# tnet = (np.asarray(normalize(tnet.T))).T
	RtT = np.dot(rentT,tnet)
	RtT= MakeTx(RtT)
	# Ematrix = np.dot(MakeTx(tnet),rnet)
	# fmat = np.dot(Ematrix,Kinv)
	# fmat = np.dot(KinvT,fmat)
	rbig = np.dot(rnet,RtT)
	ans = np.dot(KinvT,rbig)
	ans = np.dot(ans,Kinv)
	# ansterm=ans[2][2]
	# ans /= ansterm
	# normfact = fmat[2][2]
	# fmat /= normfact
	# return fmat
	return ans



brokenpointpairs= breakintopointpairs(mappeddata)
# print brokenpointpairs[0][0]
# print brokenpointpairs[0][1]

# print len(brokenpointpairs)
# print min(map(lambda x: len(x[0]),brokenpointpairs))
# print len(x[0][0])
# print len(x[0][1])

# print mappeddata[0]

def findElem(x,y,data):
	for i in xrange(len(data)/2):
		if(x==data[2*i] and y==data[2*i+1]):
			return 2*i
	return -1


def getCorrPoints(data):
	ans=[]
	if(len(data)>0):
		for i in xrange(len(data[0])/2):
			x1=data[0][2*i]
			y1=data[0][2*i+1]
			state=1
			pos=2*i
			for j in xrange(1,len(data)/2):
				x2=data[2*j-1][pos]
				y2=data[2*j-1][pos+1]
				pos=findElem(x2,y2,data[2*j])
				if(pos<0):
					state=0
					break

			if(state==1):
				ans.append([x1,y1,data[len(data)-1][pos],data[len(data)-1][pos+1]])
	return ans

def GetError(F,p1arr,p2arr):
	errterm = 0
	for i in xrange(len(p1arr)):
		homogp1 = (np.asarray([p1arr[i][0],p1arr[i][1],1])).T
		homogp2 = (np.asarray([p2arr[i][0],p2arr[i][1],1]))
		valerror = np.dot(homogp2,F)
		valerror = np.dot(valerror,homogp1)
		# print valerror
		errterm += valerror
	return errterm/len(p1arr)

def CCTN(p):
	return (640+p[0],360-p[1])

def drawlines(img1,img2,lines1,lines2,pts1,pts2):
	c=640
	d=-640
	for r,m,pt1,pt2 in zip(lines1,lines2,pts1,pts2):
		color= tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int,[0,-r[2]/r[1]])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		x2,y2 = map(int, [d, -(r[2]+r[0]*d)/r[1] ])
		ax0,ay0 = map(int,[0,-m[2]/m[1]])
		ax1,ay1 = map(int, [c, -(m[2]+m[0]*c)/m[1] ])
		ax2,ay2 = map(int, [d, -(m[2]+m[0]*d)/m[1] ])
		cv2.line(img1,CCTN((x1,y1)),CCTN((x2,y2)),color,1)
		cv2.line(img2,CCTN((ax1,ay1)),CCTN((ax2,ay2)),color,1)
		cv2.circle(img1,CCTN((int(pt1[0]),int(pt1[1]))),5,color,-1)
		cv2.circle(img2,CCTN((int(pt2[0]),int(pt2[1]))),5,color,-1)
	return img1,img2


print"************"
correspoints = getCorrPoints(mappeddata)
points1 = map(lambda x: (x[0],x[1]), correspoints)
points2 = map(lambda x: (x[2],x[3]), correspoints)

dimx = 1280
dimy = 720

for i in xrange(len(points1)):
	points1[i]=((points1[i][0]-dimx/2),(dimy/2-points1[i][1]))
	points2[i]=((points2[i][0]-dimx/2),(dimy/2-points2[i][1]))
initimage= cv2.imread('0_image.jpg')

# cv2.imshow("winnameinit", initimage)
# cv2.waitKey(1)

for p in points1:
	cv2.circle(initimage, (int(dimx/2+p[0]),int(dimy/2-p[1])), 3, (255,0,0),-1)
cv2.imshow("winnameinit", initimage)
cv2.waitKey(1)

finitimage= cv2.imread('1_image.jpg')

# cv2.imshow("winnameinit", initimage)
# cv2.waitKey(1)

for p in points2:
	cv2.circle(finitimage, (int(dimx/2+p[0]),int(dimy/2-p[1])), 3, (255,0,0),-1)
cv2.imshow("winnameinitf", finitimage)
cv2.waitKey(1)

initimage1=cv2.resize(initimage, (0,0), fx=0.5, fy=0.5)
finitimage1 = cv2.resize(finitimage, (0,0), fx=0.5, fy=0.5)


vis = np.concatenate((initimage1, finitimage1), axis=1)

for i in xrange(0,len(points1)/10):
	cv2.line(vis, (int((dimx/2+points1[i][0])/2),int((dimy/2-points1[i][1]))/2), (int((dimx/2+points2[i][0])/2)+640,int((dimy/2-points2[i][1])/2)), (0,0,255))

print "################"
# print points2
print "################"

cv2.imshow("merged", vis)
cv2.waitKey(1)



Rinit = RT_KG1[0][0]
Rfinal= RT_KG1[-1][0]
Tinit = RT_KG1[0][1]
Tfinal= RT_KG1[-1][1]
fundapython = cv2.findFundamentalMat(np.asarray(points1[0:len(points1)/1]), np.asarray(points2[0:len(points2)/1]))[0]
fundamentalmat = ObtainFundament(Rinit, Rfinal, Tinit, Tfinal)
l1= fundamentalmat.T
l1=l1.copy()
errterm = GetError(fundamentalmat, points1, points2)





pts1=np.asarray(points1[0:4])
pts2=np.asarray(points2[0:4])
lines1=cv2.cv.fromarray(np.zeros((4,3)))
lines2=cv2.cv.fromarray(np.zeros((4,3)))
lines3=cv2.cv.fromarray(np.zeros((4,3)))
lines4=cv2.cv.fromarray(np.zeros((4,3)))

cv2.cv.ComputeCorrespondEpilines(cv2.cv.fromarray(pts2), 2,cv2.cv.fromarray(l1),lines1)
cv2.cv.ComputeCorrespondEpilines(cv2.cv.fromarray(pts1), 1,cv2.cv.fromarray(l1),lines2)
cv2.cv.ComputeCorrespondEpilines(cv2.cv.fromarray(pts2), 2,cv2.cv.fromarray(fundapython),lines3)
cv2.cv.ComputeCorrespondEpilines(cv2.cv.fromarray(pts1), 1,cv2.cv.fromarray(fundapython),lines4)


def MulPointMat(points2,mat):
	a=points2[0]*mat[0][0]+points2[1]*mat[0][1]+1*mat[0][2]
	b=points2[0]*mat[1][0]+points2[1]*mat[1][1]+1*mat[1][2]
	c=points2[0]*mat[2][0]+points2[1]*mat[2][1]+1*mat[2][2]
	print "****"
	return [a,b,c]

manlines1=[]
manlines2=[]
manlines3=[]
manlines4=[]

for elem in pts2:
	a=MulPointMat(elem,l1)
	manlines1.append(a)
for elem in pts1:
	a=MulPointMat(elem,l1)
	manlines2.append(a)

for elem in pts2:
	a=MulPointMat(elem,fundapython)
	manlines3.append(a)
for elem in pts1:
	a=MulPointMat(elem,fundapython)
	manlines4.append(a)


lines1= np.asarray(lines1)
lines2= np.asarray(lines2)
lines3= np.asarray(lines3)
lines4= np.asarray(lines4)
print lines1
print lines3
MulPointMat(points2[0],fundapython)

img1,img2 = drawlines(initimage,finitimage,manlines1,manlines2,pts1,pts2)
# img3,img4 = drawlines(initimage,finitimage,lines3,lines4,pts1,pts2)

cv2.imshow("elines1",img1)
cv2.imshow("elines2",img2)




# print "our mat",fundamentalmat
print "python mat",fundapython
print "our error:",errterm
print "python err",GetError(fundapython, points1[0:len(points1)/1], points2[0:len(points2)/1])

# print Rinit,Rfinal
# print Tinit,Tfinal
# print len(getCorrPoints(mappeddata))
# print len(mappeddata[len(mappeddata)-1])
print"************"

def gethomogpoint(a):
	k= np.asarray([a[0],a[1],1]).T
	return np.dot(Kinv,k)

def getallhomog(arr):
	return map(lambda x: gethomogpoint(x), arr)


def GenerateCoeffs(R,p1,p2):
	tx= (+p1[0]*p2[2]*R[1][0]  -p1[0]*p2[1]*R[2][0] -p1[1]*p2[1]*R[2][1] + p1[1]*p2[2]*R[1][1] - p1[2]*p2[1]*R[2][2] + p1[2]*p2[2]*R[1][2])
	ty= (+p1[0]*p2[0]*R[2][0]  -p1[0]*p2[2]*R[0][0] +p1[1]*p2[0]*R[2][1] - p1[1]*p2[2]*R[0][1] + p1[2]*p2[0]*R[2][2] - p1[2]*p2[2]*R[0][2])
	tz= (-p1[0]*p2[0]*R[1][0]  +p1[0]*p2[1]*R[0][0] -p1[1]*p2[0]*R[2][1] + p1[1]*p2[1]*R[0][1] - p1[2]*p2[0]*R[1][2] + p1[2]*p2[1]*R[0][2])
	return (tx,ty,tz)

def SolveEquations(r1,r2,P1,P2):
	rnet= np.dot(np.linalg.inv(np.asarray(MakeRMatrix(r1))),np.asarray(MakeRMatrix(r2)))
	coeffs=[]
	for i in xrange(len(P1)):
		coeffs.append(GenerateCoeffs(rnet,P1[i],P2[i]))
	txcoeff=[]
	tycoeff=[]
	tzcoeff=[]
	for elem in coeffs:
		txcoeff.append(elem[0])
		tycoeff.append(elem[1])
		tzcoeff.append(-elem[2])
	A= np.vstack([txcoeff,tycoeff]).T
	solution=np.linalg.lstsq(A, tzcoeff) 
	tx1,ty1= solution[0]
	# print ((solution[1])**0.5)/len(A)
	norm = (1+ tx1**2 + ty1**2)**0.5
	transdir = [tx1/norm,ty1/norm,1/norm]
	transdir = np.asarray(transdir).T
	transdir_cor = np.dot(np.asarray(MakeRMatrix(r1)),transdir)
	return transdir_cor


homogpoint1 = map(lambda x: getallhomog(x[0]), brokenpointpairs)
homogpoint2 = map(lambda x: getallhomog(x[1]), brokenpointpairs)
# print SolveEquations(RT_KG1[0][0], RT_KG1[1][0] , homogpoint1[0], homogpoint2[0])

# print len(RT_KG1),len(homogpoint1),len(homogpoint2)

transmatrices = []
sensortrans = []
magmatrices = []
fundamentalmatrices=[]

# RT_KG1=RT_Soccer1

# dottedup=[]
for i in xrange(1,min(len(RT_KG1),len(homogpoint1))):
	# print i, len(RT_KG1),len(homogpoint1),len(homogpoint2)
	transmatrices.append(SolveEquations(RT_KG1[i-1][0],RT_KG1[i][0],homogpoint1[i],homogpoint2[i]))
	tnet = np.asarray(RT_KG1[i][1])-np.asarray(RT_KG1[i-1][1])
	magmatrices.append(tnet[0]**2 + tnet[1]**2 + tnet[2]**2)
	tnet = normalize(tnet)
	sensortrans.append(tnet)

for i in xrange(1,(len(RT_KG1))):
	# print i, len(RT_KG1),len(homogpoint1),len(homogpoint2)
	fundamentalmatrices.append(ObtainFundament(RT_KG1[i-1][0],RT_KG1[i][0],RT_KG1[i-1][1],RT_KG1[i][1]))
	

# print fundamentalmatrices

def GetMagnitude(trans1,trans2,magnit):
	answer=[]
	for i in xrange(len(trans1)):
		m1 = trans1[i][0]*trans2[i][0] +trans1[i][1]*trans2[i][1] +trans1[i][2]*trans2[i][2]
		answer.append([magnit[i],m1])
	return answer



point1 = map(lambda x: x[0], brokenpointpairs)
point2 = map(lambda x: x[1], brokenpointpairs)

print len(fundamentalmatrices),len(point1)

errterms=[]
for i in xrange(len(fundamentalmatrices)):
	errterms.append(GetError(fundamentalmatrices[i],point1[i],point2[i]))

print errterms

print len(point1),len(point2),len(fundamentalmatrices)

Magnited= GetMagnitude(sensortrans, transmatrices, magmatrices)

# print Magnited
# for elem in Magnited:
# 	print elem

boolvals = map(lambda x: x[0]>0, Magnited)
vals = map(lambda x: abs(x[1]), Magnited)

# print transmatrices
# print sensortrans
# print magmatrices
# fmatrices = map(lambda x: cv2.findFundamentalMat(np.asarray(x[0]),np.asarray(x[1]))[0], brokenpointpairs)

# ematrices = map(lambda x: K.T *x*K, fmatrices)
# print "number of e matrices", len(ematrices)

# ekg0 = map(lambda x: x[0][0], EKG1)
# print ekg0
# e0 = map(lambda x: x[0][0], ematrices)
# prin	t e0

# print ematrices
# print np.linalg.svd(ematrices[0])
# print K, np.linalg.inv(K)
# print fmatrices


# print filename
# print 

print "kg 0: ", (calcdx0[-1]**2+ calcdy0[-1]**2 + calcdz0[-1]**2)**0.5
print "kg 1: ", (calcdx1[-1]**2+ calcdy1[-1]**2 + calcdz1[-1]**2)**0.5
print "kg 2: ", (dx[-1]**2+ dy[-1]**2 + dz[-1]**2)**0.5

print "soccer 0: ", (soccerdx0[-1]**2+ soccerdy0[-1]**2 + soccerdz0[-1]**2)**0.5
print "soccer 1: ", (soccerdx1[-1]**2+ soccerdy1[-1]**2 + soccerdz1[-1]**2)**0.5
print "soccer 2: ", (soccerdx2[-1]**2+ soccerdy2[-1]**2 + soccerdz2[-1]**2)**0.5
# print "soccer 2: ", (dx[-1]**2+ dy[-1]**2 + dz[-1]**2)**0.5

print "fixed: ", (fixeddx[-1]**2+ fixeddy[-1]**2 + fixeddz[-1]**2)**0.5


plt.figure(2)
plt.subplot(1,1,1)
plt.plot(vals,color='blue')
plt.plot(boolvals,color='red')

# plt.figure(3)
# plt.subplot(3,1,1)
# plt.plot(map(lambda x: (x[0]), transmatrices),color='green')
# plt.plot(map(lambda x: x[0], sensortrans),color='blue')
# plt.plot(boolvals, color='red')

# plt.subplot(3,1,2)
# plt.plot(map(lambda x: (x[1]), transmatrices),color='green')
# plt.plot(map(lambda x: x[1], sensortrans),color='blue')
# plt.plot(boolvals, color='red')

# plt.subplot(3,1,3)
# plt.plot(map(lambda x: (x[2]), transmatrices),color='green')
# plt.plot(map(lambda x: x[2], sensortrans),color='blue')
# plt.plot(boolvals, color='red')

# # print RT_KG1

# plt.figure(4)
# plt.subplot(3,3,1)
# plt.plot(map(lambda x:x[0][0],RT_KG1))

# plt.subplot(3,3,2)
# plt.plot(map(lambda x:x[0][1],RT_KG1))

# plt.subplot(3,3,3)
# plt.plot(map(lambda x:x[0][2],RT_KG1))

# plt.subplot(3,3,4)
# plt.plot(map(lambda x:x[0][3],RT_KG1))

# plt.subplot(3,3,5)
# plt.plot(map(lambda x:x[0][4],RT_KG1))

# plt.subplot(3,3,6)
# plt.plot(map(lambda x:x[0][5],RT_KG1))

# plt.subplot(3,3,7)
# plt.plot(map(lambda x:x[0][6],RT_KG1))

# plt.subplot(3,3,8)
# plt.plot(map(lambda x:x[0][7],RT_KG1))

# plt.subplot(3,3,9)
# plt.plot(map(lambda x:x[0][8],RT_KG1))


# plt.figure(1)


# plt.subplot(3,3,1)
# plt.ylabel('ax')
# plt.plot(rawax,color='yellow')
# plt.plot(ax)
# plt.plot(pratx)

# plt.subplot(3,3,2)
# plt.ylabel('ay')
# plt.plot(raway,color='yellow')
# plt.plot(ay)
# plt.plot(praty)

# plt.subplot(3,3,3)
# plt.ylabel('az')
# plt.plot(rawaz,color='yellow')
# plt.plot(az)
# plt.plot(pratz)


# plt.subplot(3,3,4)
# plt.ylabel('vx')
# plt.plot(calcvx0)
# plt.plot(calcvx1)
# plt.plot(vx)
# plt.plot(fixedvx)

# plt.subplot(3,3,5)
# plt.ylabel('vy')
# plt.plot(calcvy0)
# plt.plot(calcvy1)
# plt.plot(vy)
# plt.plot(fixedvy)

# plt.subplot(3,3,6)
# plt.ylabel('vz')
# plt.plot(calcvz0)
# plt.plot(calcvz1)
# plt.plot(vz)
# plt.plot(fixedvz)


# plt.subplot(3,3,7)
# plt.ylabel('dx')
# plt.plot(calcdx0)
# plt.plot(calcdx1)
# plt.plot(dx)
# plt.plot(fixeddx)

# plt.subplot(3,3,8)
# plt.ylabel('dy')
# plt.plot(calcdy0)
# plt.plot(calcdy1)
# plt.plot(dy)
# plt.plot(fixeddy)

# plt.subplot(3,3,9)
# plt.ylabel('dz')
# plt.plot(calcdz0)
# plt.plot(calcdz1)
# plt.plot(dz)
# plt.plot(fixeddz)

plt.show()