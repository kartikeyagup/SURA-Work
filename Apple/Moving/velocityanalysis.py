import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import csv
import math

#File reading part
fileread=[]
with open('2014-12-28_17-53-15.csv','rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

# print fileread[0]
# required=map(lambda x: x[18:22], fileread)
# acceldata=map(lambda x: map(float,x),required[1:])

# starttime=acceldata[0][0]
# timecorrected=map(lambda x: [x[0]-starttime]+map(lambda y: y*9.81,x[1:4]),acceldata )

# [timearr,ax,ay,az] = map(list,zip(*timecorrected))

required = map(lambda x: x[26:30]+ x[33:36], fileread)
alldata =map(lambda x: map(float,x),required[1:])
starttime=alldata[0][0]
timed=map(lambda x: [x[0]-starttime] + x[1:4] + map(lambda y: y*9.81,x[4:7]),alldata)

print "Number or entries before the dupilicates removal: ", len(timed)

#Removing the vlaues with same timestamp
prevtime=timed[0][0]
timecorrected=[timed[0]]
for i in xrange(len(timed)):
	if not(timed[i][0]==prevtime):
		prevtime=timed[i][0]
		timecorrected.append(timed[i])
print "Number of values after duplicate removal: ", len(timecorrected)

[timearr,ty,tr,tp,ax,ay,az]=map(list, zip(*timecorrected))


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
rotmatrices=map(getRotationMatrix,tp,tr,ty)
# invrotmatrices=map(invertMatrix,rotmatrices)
accelerationxyz=list(zip(*[ax,ay,az]))
accelerationXYZ=map(applyRotationMatrix,rotmatrices,accelerationxyz)



# fixedbiasz=fixstaticbias(az)

[ax,ay,az]=map(list,zip(*accelerationXYZ))
velocityx=getvelocity(ax,timearr)
velocityy=getvelocity(ay,timearr)
velocityz=getvelocity(az,timearr)

displacementx=getdisplacement(velocityx,timearr)
displacementy=getdisplacement(velocityy,timearr)
displacementz=getdisplacement(velocityz,timearr)



#PLOTTING PART
plt.figure(1)

plt.subplot(331)
plt.ylabel('Ax (m/s2)')
# plt.plot(xaxis,ax,'r--',xaxis,lx,'b--')
plt.plot(ax)


plt.subplot(332)
plt.ylabel('Ay (m/s2)')
plt.plot(ay)

plt.subplot(333)
plt.ylabel('Az (m/s2)')
plt.plot(az)

plt.subplot(334)
plt.ylabel('Vx (m/s)')
plt.plot(velocityx)


plt.subplot(335)
plt.ylabel('Vy (m/s)')
plt.plot(velocityy)


plt.subplot(336)
plt.ylabel('Vx (m/s)')
plt.plot(velocityz)

plt.subplot(337)
plt.ylabel('x (m)')
plt.xlabel('Time : (100=1s)')
plt.plot(displacementx)

plt.subplot(338)
plt.ylabel('y (m)')
plt.xlabel('Time : (100=1s)')
plt.plot(displacementy)

plt.subplot(339)
plt.ylabel('z (m)')
plt.xlabel('Time : (100=1s)')
plt.plot(displacementz)

plt.show()
