import numpy as np
import copy
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm

#File input part
# filename='1432883613575SensorFusion3data.csv'
# filename='1432883832602SensorFusion3data.csv'
filename='1432881097788SensorFusion3data.csv'
filename='1432884052928SensorFusion3data.csv'
filename='1432884114117SensorFusion3data.csv'
filename='1432885404575SensorFusion3data.csv'
filename='1432885522031SensorFusion3data.csv'
filename='1432885913255SensorFusion3data.csv'
filename='1432886079901SensorFusion3data.csv'
# Vertical 30
# filename='1432889188319SensorFusion3data.csv'
# filename='1432889234657SensorFusion3data.csv'
# filename='1432889312861SensorFusion3data.csv'
#slant 30
# filename='1432889794526SensorFusion3data.csv'
# filename='1432889825777SensorFusion3data.csv'
# filename='1432889856606SensorFusion3data.csv'
# filename='1432890233299SensorFusion3data.csv'
#redmi
# filename='1432894467420SensorFusion3data.csv'
# filename='1432895270143SensorFusion3data.csv'
# nexus
filename='1432895787434SensorFusion3data.csv'
filename='1432895836475SensorFusion3data.csv'
filename='1432895904865SensorFusion3data.csv'
filename='1432896359064SensorFusion3data.csv'
filename='1432896471375SensorFusion3data.csv'
# filename='1432896755115SensorFusion3data.csv'
filename='1432897681787SensorFusion3data.csv'
# filename='1432897957228SensorFusion3data.csv' 
# filename='1432898279391SensorFusion3data.csv'
# filename='1432898327107SensorFusion3data.csv'
# filename='1432900102716SensorFusion3data.csv'
# filename='1432900964264SensorFusion3data.csv'
filename='1433158602549SensorFusion3data.csv'
filename='1433158662786SensorFusion3data.csv'
filename='1433158704380SensorFusion3data.csv'
filename='1433161336159SensorFusion3data.csv'

 
filename='1433160480109SensorFusion3data.csv'
filename='1433160544422SensorFusion3data.csv'
filename='1433160512502SensorFusion3data.csv'
filename='1433161395113SensorFusion3data.csv'
filename='1433161451561SensorFusion3data.csv'

# filename='1433158775075SensorFusion3data.csv'
# filename='1433158867822SensorFusion3data.csv'
# filename='1433158898907SensorFusion3data.csv'
filename= '1433493580541SensorFusion3data.csv'
fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

def funcConv(x):
	return x=='true'

# print fileread[0:5]
timed =map(lambda x: map(float,x[:-9] + x[-6:]) + map(funcConv, x[-9:-6]),fileread[1:])
# starttime=timed[0][0]
# timed=map(lambda x: [x[0]-starttime] + x[1:], timed)

# print "Number or entries before the dupilicates removal: ", len(timed)

# #Removing the vlaues with same timestamp
# prevtime=timed[0][0]
# timecorrected=[timed[0]]
# for i in xrange(len(timed)):
# 	if not(timed[i][0]==prevtime):
# 		prevtime=timed[i][0]
# 		timecorrected.append(timed[i])

# print "Number of values after duplicate removal: ", len(timecorrected)

# timed=timecorrected

# startindex,done=0,False
# while not(done):
# 	if np.prod(timed[startindex])>0:
# 		done=True
# 	else:
# 		startindex+=1
# g++ -ggdb `pkg-config --cflags opencv` -o `basename VideoTracking.cpp .cpp` VideoTracking.cpp `pkg-config --libs opencv`
# starttime=timed[startindex][0]
# timed=map(lambda x: [x[0]-starttime] + x[1:], timed[startindex:])

# def Numberchanges(arr):
# 	counter=1
# 	prev=arr[0]
# 	for elem in arr:
# 		if not(elem==prev):
# 			counter+=1
# 			prev=elem
# 	return counter

# print "Number of values after initialisation: ", len(timed)

[sno,timearr,imid,ax,ay,az,vx,vy,vz,dx,dy,dz,rawax,raway,rawaz,pratx,praty,pratz,motionx,motiony,motionz]=map(list, zip(*timed))

# rawax=raway=rawaz=[]

# print max(Numberchanges(wx),Numberchanges(wy),Numberchanges(wz)),"gyro"
# print max(Numberchanges(ax),Numberchanges(ay),Numberchanges(az)),"acc"
# print max(Numberchanges(gravx),Numberchanges(gravy),Numberchanges(gravz)),"grav"
# print max(Numberchanges(magx),Numberchanges(magy),Numberchanges(magz)),"mag"

stringtime=""
for elem in timearr:
	stringtime+= str(elem) + ","

stringx=""
for elem in rawax:
	stringx+= str(elem) + ","

stringy=""
for elem in raway:
	stringy+= str(elem) + ","

stringz=""
for elem in rawaz:
	stringz+= str(elem) + ","

a=open('xdata.txt','w')
a.write(stringx[:-1])
a.close()

a=open('tdata.txt','w')
a.write(stringtime[:-1])
a.close()

a=open('ydata.txt','w')
a.write(stringy[:-1])
a.close()

a=open('zdata.txt','w')
a.write(stringz[:-1])
a.close()

a=open('smoothxxx4.txt','r')
matax1=map(lambda x: float(x[:-1]),a.readlines())
a.close()

a=open('smoothxxx3.txt','r')
matax2=map(lambda x: float(x[:-1]),a.readlines())
a.close()

a=open('coeffs.txt','r')
matax3=map(lambda x: float(x[:-1]),a.readlines())
a.close()

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

# lowx= sm.nonparametric.lowess(rawax, timearr, frac = 0.0125)
# lowx,lowy=zip(*lowx)

# print lowx

# wx=map(lambda x: x*180.0/3.14,wx)
# wy=map(lambda x: x*180.0/3.14,wy)
# wz=map(lambda x: x*180.0/3.14,wz)

# print GetMotionZones(motionx)
calcvx1=GetVelocity(timearr,GetMotionZones(motionx),ax,1)
calcdx1=GetDistance(timearr,calcvx1)
calcvx0=GetVelocity(timearr,GetMotionZones(motionx),ax,0)
calcdx0=GetDistance(timearr,calcvx0)

calcvy1=GetVelocity(timearr,GetMotionZones(motiony),ay,1)
calcdy1=GetDistance(timearr,calcvy1)
calcvy0=GetVelocity(timearr,GetMotionZones(motiony),ay,0)
calcdy0=GetDistance(timearr,calcvy0)

calcvz1=GetVelocity(timearr,GetMotionZones(motionz),az,3)
calcdz1=GetDistance(timearr,calcvz1)
calcvz0=GetVelocity(timearr,GetMotionZones(motionz),az,0)
calcdz0=GetDistance(timearr,calcvz0)


soccervx2=GetVelocity(timearr,GetMotionZones(motionx),pratx,2)
soccerdx2=GetDistance(timearr,soccervx2)
soccervx1=GetVelocity(timearr,GetMotionZones(motionx),pratx,1)
soccerdx1=GetDistance(timearr,soccervx1)
soccervx0=GetVelocity(timearr,GetMotionZones(motionx),pratx,0)
soccerdx0=GetDistance(timearr,soccervx0)

soccervy2=GetVelocity(timearr,GetMotionZones(motiony),praty,2)
soccerdy2=GetDistance(timearr,soccervy2)
soccervy1=GetVelocity(timearr,GetMotionZones(motiony),praty,1)
soccerdy1=GetDistance(timearr,soccervy1)
soccervy0=GetVelocity(timearr,GetMotionZones(motiony),praty,0)
soccerdy0=GetDistance(timearr,soccervy0)

soccervz2=GetVelocity(timearr,GetMotionZones(motionz),pratz,2)
soccerdz2=GetDistance(timearr,soccervz2)
soccervz1=GetVelocity(timearr,GetMotionZones(motionz),pratz,1)
soccerdz1=GetDistance(timearr,soccervz1)
soccervz0=GetVelocity(timearr,GetMotionZones(motionz),pratz,0)
soccerdz0=GetDistance(timearr,soccervz0)

fixedvx = FixVelocity(ax,timearr, GetMotionZones(motionx))
fixedvy = FixVelocity(ay,timearr, GetMotionZones(motiony))
fixedvz = FixVelocity(az,timearr, GetMotionZones(motionz))
fixeddx = GetDistance(timearr,fixedvx)
fixeddy = GetDistance(timearr,fixedvy)
fixeddz = GetDistance(timearr,fixedvz)

print filename
print 

print "kg 0: ", (calcdx0[-1]**2+ calcdy0[-1]**2 + calcdz0[-1]**2)**0.5
print "kg 1: ", (calcdx1[-1]**2+ calcdy1[-1]**2 + calcdz1[-1]**2)**0.5
print "kg 2: ", (dx[-1]**2+ dy[-1]**2 + dz[-1]**2)**0.5

print "soccer 0: ", (soccerdx0[-1]**2+ soccerdy0[-1]**2 + soccerdz0[-1]**2)**0.5
print "soccer 1: ", (soccerdx1[-1]**2+ soccerdy1[-1]**2 + soccerdz1[-1]**2)**0.5
print "soccer 2: ", (soccerdx2[-1]**2+ soccerdy2[-1]**2 + soccerdz2[-1]**2)**0.5
# print "soccer 2: ", (dx[-1]**2+ dy[-1]**2 + dz[-1]**2)**0.5

print "fixed: ", (fixeddx[-1]**2+ fixeddy[-1]**2 + fixeddz[-1]**2)**0.5



# plt.figure(2)
# plt.subplot(1,1,1)
# plt.ylabel('ax')
# plt.plot(rawax,color='yellow')
# plt.plot(ax,color='black')
# plt.plot(pratx,color='blue')
# # plt.plot(matax3,color='red')
# # plt.plot(lowy,color='green')

plt.figure(1)

# print timearr[-1]
# print np.trapz(wx,timearr)/timearr[-1]
# print np.trapz(wy,timearr)/timearr[-1]
# print np.trapz(wz,timearr)/timearr[-1]


plt.subplot(3,3,1)
plt.ylabel('ax')
plt.plot(rawax,color='yellow')
plt.plot(ax)
# plt.plot(pratx)

plt.subplot(3,3,2)
plt.ylabel('ay')
plt.plot(raway,color='yellow')
plt.plot(ay)
# plt.plot(praty)

plt.subplot(3,3,3)
plt.ylabel('az')
plt.plot(rawaz,color='yellow')
plt.plot(az)
# plt.plot(pratz)


plt.subplot(3,3,4)
plt.ylabel('vx')
plt.plot(calcvx0)
plt.plot(calcvx1)
plt.plot(vx)
plt.plot(fixedvx)

plt.subplot(3,3,5)
plt.ylabel('vy')
plt.plot(calcvy0)
plt.plot(calcvy1)
plt.plot(vy)
plt.plot(fixedvy)

plt.subplot(3,3,6)
plt.ylabel('vz')
plt.plot(calcvz0)
plt.plot(calcvz1)
plt.plot(vz)
plt.plot(fixedvz)

plt.subplot(3,3,7)
plt.ylabel('dx')
plt.plot(calcdx0)
plt.plot(calcdx1)
plt.plot(dx)
plt.plot(fixeddx)

plt.subplot(3,3,8)
plt.ylabel('dy')
plt.plot(calcdy0)
plt.plot(calcdy1)
plt.plot(dy)
plt.plot(fixeddy)

plt.subplot(3,3,9)
plt.ylabel('dz')
plt.plot(calcdz0)
plt.plot(calcdz1)
plt.plot(dz)
plt.plot(fixeddz)

# plt.subplot(4,3,10)
# plt.ylabel('mx (m)')
# # plt.xlabel('Time : (100=1s)')
# plt.plot(mx)

# plt.subplot(4,3,11)
# plt.ylabel('my (m)')
# # plt.xlabel('Time : (100=1s)')
# plt.plot(my)

# plt.subplot(4,3,12)
# plt.ylabel('mz (m)')
# # plt.xlabel('Time : (100=1s)')
# plt.plot(mz)

plt.figure(2)

# print timearr[-1]
# print np.trapz(wx,timearr)/timearr[-1]
# print np.trapz(wy,timearr)/timearr[-1]
# print np.trapz(wz,timearr)/timearr[-1]


plt.subplot(3,3,1)
plt.ylabel('ax')
# plt.plot(ax)
plt.plot(rawax,color='yellow')
plt.plot(pratx)

plt.subplot(3,3,2)
plt.ylabel('ay')
# plt.plot(ay)
plt.plot(raway)
plt.plot(praty)

plt.subplot(3,3,3)
plt.ylabel('az')
# plt.plot(az)
plt.plot(rawaz)
plt.plot(pratz)


plt.subplot(3,3,4)
plt.ylabel('vx')
plt.plot(soccervx0)
plt.plot(soccervx1)
plt.plot(soccervx2)
plt.plot(fixedvx)

plt.subplot(3,3,5)
plt.ylabel('vy')
plt.plot(soccervy0)
plt.plot(soccervy1)
plt.plot(soccervy2)
plt.plot(fixedvy)

plt.subplot(3,3,6)
plt.ylabel('vz')
plt.plot(soccervz0)
plt.plot(soccervz1)
plt.plot(soccervz2)
plt.plot(fixedvz)

plt.subplot(3,3,7)
plt.ylabel('dx')
plt.plot(soccerdx0)
plt.plot(soccerdx1)
plt.plot(soccerdx2)
plt.plot(fixeddx)

plt.subplot(3,3,8)
plt.ylabel('dy')
plt.plot(soccerdy0)
plt.plot(soccerdy1)
plt.plot(soccerdy2)
plt.plot(fixeddy)

plt.subplot(3,3,9)
plt.ylabel('dz')
plt.plot(soccerdz0)
plt.plot(soccerdz1)
plt.plot(soccerdz2)
plt.plot(fixeddz)

# plt.subplot(4,3,10)
# plt.ylabel('mx (m)')
# # plt.xlabel('Time : (100=1s)')
# plt.plot(mx)

# plt.subplot(4,3,11)
# plt.ylabel('my (m)')
# # plt.xlabel('Time : (100=1s)')
# plt.plot(my)

# plt.subplot(4,3,12)
# plt.ylabel('mz (m)')
# # plt.xlabel('Time : (100=1s)')
# plt.plot(mz)

# plt.show()

plt.show()
