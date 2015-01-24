import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import csv
import math

#File reading part
fileread=[]
with open('2014-12-28_18-02-14.csv','rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

print fileread[0]

timed =map(lambda x: map(float,x),fileread[1:])
timinit=timed[0][0]
timed=map(lambda x: [x[0]-timinit] + x[1:],timed)

print "Number or entries before the dupilicates removal: ", len(timed)

#Removing the vlaues with same timestamp
prevtime=timed[0][0]
timecorrected=[timed[0]]
for i in xrange(len(timed)):
	if not(timed[i][0]==prevtime):
		prevtime=timed[i][0]
		timecorrected.append(timed[i])

timed=timecorrected
print "Number of values after duplicate removal: ", len(timecorrected)

[timearr,r0,r1,r2,r3,r4,r5,r6,r7,r8,wx,wy,wz,mx,my,mz,gx,gy,gz,ax,ay,az]=map(list, zip(*timed))

#Functions part
def findstaticbias(accelarrayx,lim=10):
	#Finds the static bias considering the the 1st lim readings and last lim readings default is 3 
	sum1=sum(accelarrayx[0:lim]) +sum(accelarrayx[len(accelarrayx)-lim:len(accelarrayx)])
	return sum1/(2*lim)

def getlimitingvalue(accelarrayx,lim=10):
	l1=map(abs,accelarrayx[0:lim])
	l2=map(abs,accelarrayx[len(accelarrayx)-lim:len(accelarrayx)])
	return max(max(l1),max(l2))

def fixstaticbias(accelarrayx):
	staticbias=findstaticbias(accelarrayx)				#calclulating the static bias
	applied=map(lambda x: x-staticbias, accelarrayx)	#Applying it now	
	limitvalue=getlimitingvalue(applied)
	print "limitvalus is",limitvalue
	for i in xrange(len(accelarrayx)):
		if abs(applied[i])<limitvalue:
			applied[i]=0
	return applied

def getvelocity(accelarrayx,timearray):
	fixed=fixstaticbias(accelarrayx)
	#TODO: Make this order n instead of n*n
	velarray=[0.0]*(len(accelarrayx))
	for i in xrange(len(fixed)):
		velarray[i]=np.trapz(fixed[0:1+i],timearray[0:i+1])
	return velarray

def getdisplacement(velarrayx,timearray):
	#TODO: Make this of order n instead of n*n
	disparrayx=[0.0]*len(velarrayx)
	for i in xrange(len(velarrayx)):
		disparrayx[i]=np.trapz(velarrayx[0:1+i],timearray[0:i+1])
	return disparrayx

def getRotationMatrix(thetap,thetar,thetay):
	#For Rzxy
	#thetap=theta pitch, (X axis)
	#thetar=theta roll, (Yaxis)
	#thetay=theta yaw, (Z axis)
	Cp,Cr,Cy=math.cos(thetap),math.cos(thetar),math.cos(thetay)
	Sp,Sr,Sy=math.sin(thetap),math.sin(thetar),math.sin(thetay)
	R=[0.0]*9
	R[0]=Cr*Cy+Sr*Sp*Sy
	R[1]=Cp*Sy
	R[2]=Cr*Sp*Sy -Sr*Cy
	R[3]=Sr*Sp*Cy -Cr*Sy
	R[4]=Cp*Cy
	R[5]=Sr*Sy + Cr*Sp*Cy
	R[6]=Sr*Cp
	R[7]=-Sp
	R[8]=Cr*Cp
	return R


def getRotMatrix(g,m):
	Hx = m[1]*g[2] - m[2]*g[1]
	Hy = m[2]*g[0] - m[0]*g[2]
	Hz = m[0]*g[1] - m[1]*g[0]
	h=[Hx,Hy,Hz]
	H=math.sqrt(Hx*Hx + Hy*Hy + Hz*Hz)
	invH=1.0 / H
	h=map(lambda x: x*invH, h)
	invA = 1.0/ math.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])
	g=map(lambda x: x*invA, g)
	Mx = g[1]*h[2] - g[2]*h[1]
	My = g[2]*h[0] - g[0]*h[2]
	Mz = g[0]*h[1] - g[1]*h[0]
	R=[i for i in h]+[Mx,My,Mz]+[j for j in g]
	return R

def invertMatrix(input_array):
	dimension=int(len(input_array)**0.5)
	matrixform=[0]*dimension
	for i in xrange(dimension):
		matrixform[i]=input_array[dimension*i:dimension*(i+1)]
	matrixform=np.matrix(matrixform)
	inverse=inv(matrixform)
	listform=inverse.tolist()
	ans=[]
	for elem in listform:
		ans+=elem
	return ans

# print invertMatrix([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0])

def applyRotationMatrix(rot_matrix_array,acc_array):
	acc_output=[0.0]*3
	acc_output[0]=rot_matrix_array[0]*acc_array[0] + rot_matrix_array[1]*acc_array[1] + rot_matrix_array[2]*acc_array[2]
	acc_output[1]=rot_matrix_array[3]*acc_array[0] + rot_matrix_array[4]*acc_array[1] + rot_matrix_array[5]*acc_array[2]
	acc_output[2]=rot_matrix_array[6]*acc_array[0] + rot_matrix_array[7]*acc_array[1] + rot_matrix_array[8]*acc_array[2]
	return acc_output


#Applying functions


accelerationxyz=list(zip(*[ax,ay,az]))
rotmatrices=list(zip(*[r0,r1,r2,r3,r4,r5,r6,r7,r8]))

accelerationXYZ=map(ApplyRotationMatrix,rotmatrices,accelerationxyz)

[ax,ay,az]=map(list,zip(*accelerationXYZ))


correctedax=ApplyCorrections(ax)
correcteday=ApplyCorrections(ay)
correctedaz=ApplyCorrections(az)

velocityx=GetVeleocityArray(ax,timearr)
velocityy=GetVeleocityArray(ay,timearr)
velocityz=GetVeleocityArray(az,timearr)

displacementx=GetDisplacementArray(velocityx,timearr)
displacementy=GetDisplacementArray(velocityy,timearr)
displacementz=GetDisplacementArray(velocityz,timearr)


#PLOTTING PART
plt.figure(1)

plt.subplot(4,3,1)
plt.ylabel('Ax (m/s2)')
plt.plot(ax)

plt.subplot(4,3,2)
plt.ylabel('Ay (m/s2)')
plt.plot(ay)

plt.subplot(4,3,3)
plt.ylabel('Az (m/s2)')
plt.plot(az)

plt.subplot(4,3,4)
plt.ylabel('Corr Ax')
plt.plot(correctedax)

plt.subplot(4,3,5)
plt.ylabel('Corr Ay')
plt.plot(correcteday)

plt.subplot(4,3,6)
plt.ylabel('Corr Az')
plt.plot(correctedaz)

plt.subplot(4,3,7)
plt.ylabel('Vx (m/s)')
plt.plot(velocityx)

plt.subplot(4,3,8)
plt.ylabel('Vy (m/s)')
plt.plot(velocityy)

plt.subplot(4,3,9)
plt.ylabel('Vz (m/s)')
plt.plot(velocityz)

plt.subplot(4,3,10)
plt.ylabel('x (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementx)

plt.subplot(4,3,11)
plt.ylabel('y (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementy)

plt.subplot(4,3,12)
plt.ylabel('z (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementz)


plt.show()
