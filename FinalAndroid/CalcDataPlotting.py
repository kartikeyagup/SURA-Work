import numpy as np
import copy
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm

#File input part
filename='1432810578853SensorFusion3data.csv'
fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

print fileread[0:5]
timed =map(lambda x: map(float,x[:-9] + x[-6:]),fileread[1:])
# starttime=timed[0][0]
# timed=map(lambda x: [x[0]-starttime] + x[1:], timed)

print "Number or entries before the dupilicates removal: ", len(timed)

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

[sno,timearr,imid,ax,ay,az,vx,vy,vz,dx,dy,dz,rawax,raway,rawaz,pratx,praty,pratz]=map(list, zip(*timed))

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


lowx= sm.nonparametric.lowess(rawax, timearr, frac = 0.0125)
lowx,lowy=zip(*lowx)

# print lowx

# wx=map(lambda x: x*180.0/3.14,wx)
# wy=map(lambda x: x*180.0/3.14,wy)
# wz=map(lambda x: x*180.0/3.14,wz)

plt.figure(2)
plt.subplot(1,1,1)
plt.ylabel('ax')
plt.plot(rawax,color='yellow')
plt.plot(ax,color='black')
plt.plot(pratx,color='blue')
# plt.plot(matax3,color='red')
plt.plot(lowy,color='green')

plt.figure(1)

# print timearr[-1]
# print np.trapz(wx,timearr)/timearr[-1]
# print np.trapz(wy,timearr)/timearr[-1]
# print np.trapz(wz,timearr)/timearr[-1]


plt.subplot(3,3,1)
plt.ylabel('ax')
plt.plot(ax)
plt.plot(rawax)
plt.plot(pratx)

plt.subplot(3,3,2)
plt.ylabel('ay')
plt.plot(ay)
plt.plot(raway)
plt.plot(praty)

plt.subplot(3,3,3)
plt.ylabel('az')
plt.plot(az)
plt.plot(rawaz)
plt.plot(pratz)


plt.subplot(3,3,4)
plt.ylabel('vx')
plt.plot(vx)

plt.subplot(3,3,5)
plt.ylabel('vy')
plt.plot(vy)

plt.subplot(3,3,6)
plt.ylabel('vz')
plt.plot(vz)

plt.subplot(3,3,7)
plt.ylabel('dx')
plt.plot(dx)

plt.subplot(3,3,8)
plt.ylabel('dy')
plt.plot(dy)

plt.subplot(3,3,9)
plt.ylabel('dz')
plt.plot(dz)

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

plt.show()