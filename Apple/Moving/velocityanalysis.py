import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import csv

#File reading part
fileread=[]
with open('2014-12-26_22-37-07.csv','rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)
print fileread[0][35]
required=map(lambda x: x[18:22], fileread)
# required=map(lambda x: x[32:36], fileread)
acceldata=map(lambda x: map(float,x),required[1:])

starttime=acceldata[0][0]
timecorrected=map(lambda x: [x[0]-starttime]+map(lambda y: y*9.81,x[1:4]),acceldata )

[timearr,ax,ay,az] = map(list,zip(*timecorrected))


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
fixedbias=fixstaticbias(az)
# print np.trapz(fixedbias)
velocity=getvelocity(az,timearr)
displacement=getdisplacement(velocity,timearr)

#PLOTTING PART
plt.figure(1)

plt.subplot(411)
plt.ylabel('Ax (m/s2)')
# plt.plot(xaxis,ax,'r--',xaxis,lx,'b--')
plt.plot(az)


plt.subplot(412)
plt.ylabel('Corrected Ax (m/s2)')
# plt.plot(xaxis,ay,'r--',xaxis,lx,'b--')
plt.plot(fixedbias)


plt.subplot(413)
plt.ylabel('Vx (m/s)')
plt.plot(velocity)

plt.subplot(414)
plt.ylabel('x (m)')
plt.xlabel('Time : (100=1s)')
plt.plot(displacement)

plt.show()
