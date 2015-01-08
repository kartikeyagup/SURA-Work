import numpy as np
import copy
import csv
import matplotlib.pyplot as plt

#File input part
filename='Jan 7, 2015 65300 PM.csv'
fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

print fileread[0:5]
timed =map(lambda x: map(float,x),fileread[1:])
# starttime=timed[0][0]
# timed=map(lambda x: [x[0]-starttime] + x[1:], timed)

print "Number or entries before the dupilicates removal: ", len(timed)

#Removing the vlaues with same timestamp
prevtime=timed[0][0]
timecorrected=[timed[0]]
for i in xrange(len(timed)):
	if not(timed[i][0]==prevtime):
		prevtime=timed[i][0]
		timecorrected.append(timed[i])

print "Number of values after duplicate removal: ", len(timecorrected)

timed=timecorrected

startindex,done=0,False
while not(done):
	if np.prod(timed[startindex])>0:
		done=True
	else:
		startindex+=1

starttime=timed[startindex][0]
timed=map(lambda x: [x[0]-starttime] + x[1:], timed[startindex:])

def Numberchanges(arr):
	counter=1
	prev=arr[0]
	for elem in arr:
		if not(elem==prev):
			counter+=1
			prev=elem
	return counter

print "Number of values after initialisation: ", len(timed)

[timearr,wx,wy,wz,ax,ay,az,gravx,gravy,gravz,magx,magy,magz,r0,r1,r2,r3,r4,r5,r6,r7,r8]=map(list, zip(*timed))

print max(Numberchanges(wx),Numberchanges(wy),Numberchanges(wz)),"gyro"
print max(Numberchanges(ax),Numberchanges(ay),Numberchanges(az)),"acc"
print max(Numberchanges(gravx),Numberchanges(gravy),Numberchanges(gravz)),"grav"
print max(Numberchanges(magx),Numberchanges(magy),Numberchanges(magz)),"mag"



# wx=map(lambda x: x*180.0/3.14,wx)
# wy=map(lambda x: x*180.0/3.14,wy)
# wz=map(lambda x: x*180.0/3.14,wz)

plt.figure(1)

# print timearr[-1]
# print np.trapz(wx,timearr)/timearr[-1]
# print np.trapz(wy,timearr)/timearr[-1]
# print np.trapz(wz,timearr)/timearr[-1]


plt.subplot(4,3,1)
plt.ylabel('wx (rad/s)')
plt.plot(wx)

plt.subplot(4,3,2)
plt.ylabel('wy (rad/s)')
plt.plot(wy)

plt.subplot(4,3,3)
plt.ylabel('wz (rad/s)')
plt.plot(wz)

plt.subplot(4,3,4)
plt.ylabel('Ax')
plt.plot(ax)

plt.subplot(4,3,5)
plt.ylabel('Ay')
plt.plot(ay)

plt.subplot(4,3,6)
plt.ylabel('Az')
plt.plot(az)

plt.subplot(4,3,7)
plt.ylabel('gx (m/s)')
plt.plot(gravx)

plt.subplot(4,3,8)
plt.ylabel('gy (m/s)')
plt.plot(gravy)

plt.subplot(4,3,9)
plt.ylabel('gz (m/s)')
plt.plot(gravz)

plt.subplot(4,3,10)
plt.ylabel('mx (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(magx)

plt.subplot(4,3,11)
plt.ylabel('my (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(magy)

plt.subplot(4,3,12)
plt.ylabel('mz (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(magz)

plt.show()